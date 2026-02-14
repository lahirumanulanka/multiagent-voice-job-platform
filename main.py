import os
import json
import time
from typing import Dict, Any, List, Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response, JSONResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, ConversationRelay

from openai import OpenAI

app = FastAPI()

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# For demo/viva: in-memory store. Replace with DB later.
SESSIONS: Dict[str, Dict[str, Any]] = {}


def now() -> int:
    return int(time.time())


def https_to_wss(url: str) -> str:
    return url.replace("https://", "wss://")


# ----------------------------
# What "complete" means for your intake (tune as you like)
# ----------------------------
REQUIRED_FIELDS = [
    "full_name",
    "phone_number",
    "preferred_language",
    "desired_job_title",
    "skills",
    "experience_years",
    "location",
    "expected_salary",
    "availability",
    "education",
]


SYSTEM_PROMPT = """
You are a REAL-TIME voice AI agent for a Sri Lanka job-seeking platform.

Goal:
1) Talk with the caller naturally.
2) Collect job seeker details to create a structured Job Description JSON.
3) Ask ONE short question at a time.
4) When all required details are collected, do:
   - Thank the caller
   - Tell them: "We will create the job post and publish it on our website"
   - Say goodbye
   - End the call

Keep replies short and phone-friendly.
"""


def empty_job() -> Dict[str, Any]:
    return {
        "full_name": None,
        "phone_number": None,
        "preferred_language": None,  # Sinhala/English/Tamil
        "desired_job_title": None,
        "skills": [],
        "experience_years": None,
        "location": None,
        "expected_salary": None,
        "availability": None,
        "education": None,
        "notes": None,
    }


def is_complete(job: Dict[str, Any]) -> bool:
    # skills must be non-empty list
    if not isinstance(job.get("skills"), list) or len(job.get("skills")) == 0:
        return False

    for k in REQUIRED_FIELDS:
        v = job.get(k)
        if v is None:
            return False
        if isinstance(v, str) and not v.strip():
            return False
    return True


async def llm_reply(history: List[Dict[str, str]], job_state: Dict[str, Any]) -> str:
    """
    Main agent reply. Uses job_state context so LLM knows what's missing.
    """
    if client is None:
        return "OPENAI_API_KEY is not set. Please set it in Railway Variables. What job are you looking for?"

    missing = []
    for k in REQUIRED_FIELDS:
        v = job_state.get(k)
        if k == "skills":
            if not isinstance(v, list) or len(v) == 0:
                missing.append("skills")
        else:
            if v is None or (isinstance(v, str) and not v.strip()):
                missing.append(k)

    context = {
        "current_job_json": job_state,
        "missing_fields": missing,
        "instruction": "Ask the next best ONE question to fill missing fields. If none missing, say closing message and goodbye.",
    }

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Context (JSON): " + json.dumps(context, ensure_ascii=False),
            },
            *history,
        ],
        max_output_tokens=220,
    )

    out = []
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    out.append(c.text)

    return ("\n".join(out)).strip() or "Sorry, could you repeat that?"


async def llm_update_job_json(
    history: List[Dict[str, str]], current: Dict[str, Any]
) -> Dict[str, Any]:
    """
    After each user turn, update/merge job JSON using LLM extraction.
    Returns a full JSON object (same keys).
    """
    if client is None:
        return current

    prompt = f"""
Update the job JSON using the latest conversation.
Return ONLY valid JSON with these keys exactly:
full_name, phone_number, preferred_language, desired_job_title, skills, experience_years,
location, expected_salary, availability, education, notes.

Rules:
- skills must be an array of strings.
- experience_years must be a number if known, else null.
- If unknown, keep null (or [] for skills).
- Merge with existing JSON (do not delete existing values unless clearly corrected).
Existing JSON:
{json.dumps(current, ensure_ascii=False)}
"""

    convo = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "You extract and update structured data."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Conversation:\n" + convo},
        ],
        max_output_tokens=600,
    )

    text = ""
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    text += c.text

    text = text.strip()
    try:
        updated = json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b != -1 and b > a:
            updated = json.loads(text[a : b + 1])
        else:
            raise

    # harden types
    if not isinstance(updated.get("skills"), list):
        updated["skills"] = []

    return updated


async def ws_send_text(ws: WebSocket, text: str):
    # Twilio will speak this text
    await ws.send_text(json.dumps({"type": "text", "token": text, "last": True}))


async def ws_end_call(ws: WebSocket, handoff: Optional[Dict[str, Any]] = None):
    # End the ConversationRelay session.
    payload = {"type": "end"}
    if handoff is not None:
        payload["handoffData"] = json.dumps(handoff, ensure_ascii=False)
    await ws.send_text(json.dumps(payload))


@app.get("/health")
def health():
    return {"ok": True, "ts": now()}


@app.post("/voice/incoming")
async def voice_incoming(request: Request):
    raw = (await request.body()).decode("utf-8", errors="ignore")
    form = {k: v[0] for k, v in parse_qs(raw).items()}

    call_sid = form.get("CallSid", "unknown")
    from_num = form.get("From", "unknown")
    to_num = form.get("To", "unknown")
    print(f"📞 Incoming: callSid={call_sid} from={from_num} to={to_num}")

    if not PUBLIC_BASE_URL:
        vr = VoiceResponse()
        vr.say("Configuration error. PUBLIC_BASE_URL not set.")
        vr.hangup()
        return Response(str(vr), media_type="text/xml")

    relay_wss = https_to_wss(PUBLIC_BASE_URL) + "/ws/relay"

    SESSIONS[call_sid] = {
        "callSid": call_sid,
        "from": from_num,
        "to": to_num,
        "createdAt": now(),
        "history": [],
        "job_json": empty_job(),
        "finalized": False,
    }

    vr = VoiceResponse()
    connect = Connect(action=f"{PUBLIC_BASE_URL}/voice/ended")

    connect.append(
        ConversationRelay(
            url=relay_wss,
            welcomeGreeting="Hello! I am your job assistant. What job are you looking for?",
            interruptible="any",
            dtmfDetection=True,
            language="en-US",
        )
    )
    vr.append(connect)
    return Response(str(vr), media_type="text/xml")


@app.post("/voice/ended")
async def voice_ended(request: Request):
    # This is called after <Connect> ends.
    raw = (await request.body()).decode("utf-8", errors="ignore")
    form = {k: v[0] for k, v in parse_qs(raw).items()}
    call_sid = form.get("CallSid", "unknown")
    print(f"☎️ Connect ended: callSid={call_sid}")
    return Response("OK", media_type="text/plain")


@app.get("/jobs/{call_sid}")
async def get_job(call_sid: str):
    sess = SESSIONS.get(call_sid)
    if not sess:
        return JSONResponse({"error": "unknown callSid"}, status_code=404)

    return JSONResponse(
        {
            "callSid": call_sid,
            "from": sess.get("from"),
            "createdAt": sess.get("createdAt"),
            "finalized": sess.get("finalized"),
            "job_json": sess.get("job_json"),
            "history": sess.get("history"),
        }
    )


@app.websocket("/ws/relay")
async def ws_relay(ws: WebSocket):
    await ws.accept()
    print("✅ Relay WS connected")

    call_sid: Optional[str] = None

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            mtype = data.get("type")

            if mtype == "setup":
                call_sid = data.get("callSid")
                print(f"🧩 setup callSid={call_sid}")

            elif mtype == "prompt":
                if not call_sid:
                    continue

                user_text = (data.get("voicePrompt") or "").strip()
                if not user_text:
                    continue

                print(f"🗣 user: {user_text}")

                sess = SESSIONS.get(call_sid)
                if not sess:
                    sess = SESSIONS.setdefault(
                        call_sid,
                        {
                            "callSid": call_sid,
                            "createdAt": now(),
                            "history": [],
                            "job_json": empty_job(),
                            "finalized": False,
                        },
                    )

                # If already finalized, ignore extra prompts
                if sess.get("finalized"):
                    continue

                # store user message
                sess["history"].append({"role": "user", "content": user_text})

                # update job JSON every turn
                try:
                    sess["job_json"] = await llm_update_job_json(
                        sess["history"], sess["job_json"]
                    )
                except Exception as e:
                    print("❌ JSON update failed:", repr(e))

                # generate assistant reply
                try:
                    reply = await llm_reply(sess["history"], sess["job_json"])
                except Exception as e:
                    print("❌ LLM failed:", repr(e))
                    reply = "Sorry, I had a small system issue. Please repeat your last sentence."

                print(f"🤖 assistant: {reply}")
                sess["history"].append({"role": "assistant", "content": reply})

                # speak reply
                await ws_send_text(ws, reply)

                # If completed, say closing message via GPT and end call
                if is_complete(sess["job_json"]):
                    sess["finalized"] = True

                    closing = (
                        "Thank you. I have collected your job details. "
                        "We will create your job post and publish it on our website soon. "
                        "Have a nice day. Goodbye."
                    )

                    # Speak closing
                    print("✅ Completed intake. Sending closing + ending call.")
                    sess["history"].append({"role": "assistant", "content": closing})
                    await ws_send_text(ws, closing)

                    # End session (Twilio will close the connect)
                    await ws_end_call(
                        ws,
                        handoff={
                            "callSid": call_sid,
                            "job_json": sess["job_json"],
                            "createdAt": sess["createdAt"],
                        },
                    )
                    break

            elif mtype in ("end", "end_session"):
                print(f"🏁 end callSid={call_sid}")
                break

            else:
                # dtmf / interrupt / etc
                pass

    except Exception as e:
        # 1000 close is normal
        print("❌ WS error:", repr(e))

    finally:
        # If disconnect happened before finalize, still try to create best JSON
        if call_sid and call_sid in SESSIONS:
            sess = SESSIONS[call_sid]
            if sess.get("job_json") is None:
                sess["job_json"] = empty_job()

        try:
            await ws.close()
        except Exception:
            pass
        print("✅ WS closed")
