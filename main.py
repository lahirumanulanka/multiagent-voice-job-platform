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

# Demo store (memory)
SESSIONS: Dict[str, Dict[str, Any]] = {}


def now() -> int:
    return int(time.time())


def https_to_wss(url: str) -> str:
    return url.replace("https://", "wss://")


# ✅ 1-minute demo fields only
REQUIRED_FIELDS = ["full_name", "expected_salary"]


def empty_job() -> Dict[str, Any]:
    return {
        "full_name": None,
        "expected_salary": None,
        "notes": "1-minute demo intake (only name + salary).",
    }


def is_complete(job: Dict[str, Any]) -> bool:
    for k in REQUIRED_FIELDS:
        v = job.get(k)
        if v is None:
            return False
        if isinstance(v, str) and not v.strip():
            return False
    return True


SYSTEM_PROMPT = """
You are a voice AI agent for a Sri Lanka job platform demo.
Goal: finish in ~1 minute.

You MUST collect ONLY:
- full_name
- expected_salary (LKR)

Rules:
- Ask ONE short question at a time.
- Do not ask extra questions.
- After you get both, say:
  "Thank you. We will create your job post and publish it on our website soon. Goodbye."
Then end the call.
"""


async def llm_reply(history: List[Dict[str, str]], job_state: Dict[str, Any]) -> str:
    """LLM decides next short question or closing message."""
    if client is None:
        # fallback (still works)
        if not job_state.get("full_name"):
            return "What is your name?"
        if not job_state.get("expected_salary"):
            return "What is your expected salary in Sri Lankan rupees?"
        return "Thank you. We will create your job post and publish it on our website soon. Goodbye."

    missing = [k for k in REQUIRED_FIELDS if not job_state.get(k)]
    context = {"job_json": job_state, "missing_fields": missing}

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Context JSON: " + json.dumps(context, ensure_ascii=False),
            },
            *history,
        ],
        max_output_tokens=120,
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
    """Extract/merge ONLY name + salary."""
    if client is None:
        return current

    prompt = f"""
Update the JSON using the conversation.
Return ONLY valid JSON with keys exactly:
full_name, expected_salary, notes

Rules:
- If unknown, use null.
- expected_salary should be a string like "100000 LKR" or "150000-200000 LKR".
- Merge with existing JSON (keep existing values unless corrected).

Existing JSON:
{json.dumps(current, ensure_ascii=False)}
"""

    convo = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "You extract structured data."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Conversation:\n" + convo},
        ],
        max_output_tokens=220,
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
        updated = (
            json.loads(text[a : b + 1]) if a != -1 and b != -1 and b > a else current
        )

    # ensure keys exist
    if "full_name" not in updated:
        updated["full_name"] = current.get("full_name")
    if "expected_salary" not in updated:
        updated["expected_salary"] = current.get("expected_salary")
    if "notes" not in updated:
        updated["notes"] = current.get("notes")

    return updated


async def ws_send_text(ws: WebSocket, text: str):
    await ws.send_text(json.dumps({"type": "text", "token": text, "last": True}))


async def ws_end_call(ws: WebSocket, handoff: Optional[Dict[str, Any]] = None):
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
        vr.say("Configuration error. PUBLIC BASE_URL not set.")
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
            welcomeGreeting="Hello! Quick demo. What is your name?",
            interruptible="any",
            dtmfDetection=True,
            language="en-US",
        )
    )
    vr.append(connect)
    return Response(str(vr), media_type="text/xml")


@app.post("/voice/ended")
async def voice_ended(request: Request):
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

                if sess.get("finalized"):
                    continue

                print(f"🗣 user: {user_text}")
                sess["history"].append({"role": "user", "content": user_text})

                # update job json quickly (name/salary only)
                try:
                    sess["job_json"] = await llm_update_job_json(
                        sess["history"], sess["job_json"]
                    )
                except Exception as e:
                    print("❌ JSON update failed:", repr(e))

                # ask next question or close
                try:
                    reply = await llm_reply(sess["history"], sess["job_json"])
                except Exception as e:
                    print("❌ LLM failed:", repr(e))
                    reply = "Sorry, please repeat that."

                print(f"🤖 assistant: {reply}")
                sess["history"].append({"role": "assistant", "content": reply})
                await ws_send_text(ws, reply)

                # if we have name + salary => close fast
                if is_complete(sess["job_json"]):
                    sess["finalized"] = True

                    closing = "Thank you. We will create your job post and publish it on our website soon. Goodbye."
                    print("✅ Completed (name+salary). Closing call.")
                    sess["history"].append({"role": "assistant", "content": closing})
                    await ws_send_text(ws, closing)

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

    except Exception as e:
        # 1000 close is normal
        print("❌ WS error:", repr(e))

    finally:
        try:
            await ws.close()
        except Exception:
            pass
        print("✅ WS closed")
