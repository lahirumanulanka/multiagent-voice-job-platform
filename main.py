import os
import json
import time
from typing import Dict, Any, List
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response, JSONResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, ConversationRelay

# OpenAI (text-only) for agent brain + final JSON extraction
from openai import OpenAI

app = FastAPI()

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip(
    "/"
)  # https://<app>.up.railway.app
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Demo memory store (OK for viva). Use DB later.
SESSIONS: Dict[str, Dict[str, Any]] = {}


def now() -> int:
    return int(time.time())


def https_to_wss(url: str) -> str:
    return url.replace("https://", "wss://")


SYSTEM_PROMPT = """
You are a voice AI agent for a Sri Lanka job-seeking platform.
Goal: talk with the caller and collect details to produce a Job Description JSON.

Collect these fields:
- full_name
- phone_number
- preferred_language (Sinhala/English/Tamil)
- desired_job_title
- skills (list)
- experience_years
- location (city)
- expected_salary (LKR range)
- availability (immediate / notice period)
- education
- notes

Rules:
- Ask ONE question at a time.
- If missing info, ask for it.
- Confirm critical fields (job title, skills, location).
- Keep responses short and natural for phone calls.
"""


async def llm_text(history: List[Dict[str, str]]) -> str:
    """Generate next agent reply (text)."""
    if client is None:
        return "OPENAI_API_KEY is not set. Please set it in Railway Variables. What job are you looking for?"

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        max_output_tokens=220,
    )

    out = []
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    out.append(c.text)

    return ("\n".join(out)).strip() or "Sorry, could you repeat that?"


async def llm_extract_json(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Create final Job JSON at end."""
    if client is None:
        return {
            "full_name": None,
            "phone_number": None,
            "preferred_language": None,
            "desired_job_title": None,
            "skills": [],
            "experience_years": None,
            "location": None,
            "expected_salary": None,
            "availability": None,
            "education": None,
            "notes": "OPENAI_API_KEY not set; extraction skipped.",
        }

    prompt = """Return ONLY valid JSON with keys exactly:
full_name, phone_number, preferred_language, desired_job_title, skills, experience_years,
location, expected_salary, availability, education, notes.
skills must be an array of strings. unknown values => null (or [] for skills)."""

    convo = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "You extract structured data."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Conversation:\n" + convo},
        ],
        max_output_tokens=500,
    )

    text = ""
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    text += c.text

    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # salvage JSON block if model adds extra text
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(text[a : b + 1])
        raise


@app.get("/health")
def health():
    return {"ok": True, "ts": now()}


@app.post("/voice/incoming")
async def voice_incoming(request: Request):
    # Twilio posts application/x-www-form-urlencoded
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

    # ConversationRelay WebSocket
    relay_wss = https_to_wss(PUBLIC_BASE_URL) + "/ws/relay"

    SESSIONS[call_sid] = {
        "callSid": call_sid,
        "from": from_num,
        "to": to_num,
        "createdAt": now(),
        "history": [],
        "job_json": None,
    }

    vr = VoiceResponse()
    connect = Connect(action=f"{PUBLIC_BASE_URL}/voice/ended")

    # ConversationRelay: Twilio handles STT + TTS, we handle text + LLM
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
            "job_json": sess.get("job_json"),
            "history": sess.get("history"),
        }
    )


@app.websocket("/ws/relay")
async def ws_relay(ws: WebSocket):
    await ws.accept()
    print("✅ Relay WS connected")

    call_sid = None

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            mtype = data.get("type")

            # Twilio sends setup first (contains callSid)
            if mtype == "setup":
                call_sid = data.get("callSid")
                print(f"🧩 setup callSid={call_sid}")

            # user transcript comes as a prompt
            elif mtype == "prompt":
                if not call_sid:
                    continue

                user_text = (data.get("voicePrompt") or "").strip()
                print(f"🗣 user: {user_text}")

                sess = SESSIONS.get(call_sid)
                if not sess:
                    SESSIONS[call_sid] = {
                        "callSid": call_sid,
                        "history": [],
                        "job_json": None,
                    }
                    sess = SESSIONS[call_sid]

                sess["history"].append({"role": "user", "content": user_text})

                reply = await llm_text(sess["history"])
                sess["history"].append({"role": "assistant", "content": reply})

                # send assistant response back (Twilio will speak it)
                await ws.send_text(
                    json.dumps({"type": "text", "token": reply, "last": True})
                )

            # session end -> generate JSON
            elif mtype in ("end", "end_session"):
                print(f"🏁 end callSid={call_sid}")
                if call_sid and call_sid in SESSIONS:
                    sess = SESSIONS[call_sid]
                    try:
                        sess["job_json"] = await llm_extract_json(sess["history"])
                        print("🧾 job_json created")
                    except Exception as e:
                        print("❌ job_json extraction failed:", repr(e))
                break

            else:
                # dtmf / interrupt / errors etc
                pass

    except Exception as e:
        print("❌ WS error:", repr(e))
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        print("✅ WS closed")
