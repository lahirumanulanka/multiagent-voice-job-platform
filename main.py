import os
import json
import time
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Start, Stream

app = FastAPI()

# Set this in Railway Variables:
# PUBLIC_BASE_URL = https://multiagent-voice-job-platform.up.railway.app
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


@app.get("/")
def root():
    return {"status": "running", "service": "twilio-media-streams-demo"}


@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


@app.post("/voice/incoming")
async def voice_incoming(request: Request):
    """
    Twilio sends application/x-www-form-urlencoded.
    We parse it without python-multipart using parse_qs.
    """
    body = (await request.body()).decode("utf-8", errors="ignore")
    params = parse_qs(body)

    def get_param(key: str, default: str = "unknown") -> str:
        return params.get(key, [default])[0]

    call_sid = get_param("CallSid")
    from_number = get_param("From")
    to_number = get_param("To")
    call_status = get_param("CallStatus", "unknown")
    caller_country = get_param("CallerCountry", "unknown")

    print(
        f"📞 Incoming call: CallSid={call_sid} From={from_number} To={to_number} "
        f"Status={call_status} Country={caller_country}"
    )

    if not PUBLIC_BASE_URL:
        vr = VoiceResponse()
        vr.say("Configuration error. Public base URL is not set.")
        vr.hangup()
        return Response(str(vr), media_type="text/xml")

    # Twilio Media Streams expects WSS
    wss_url = PUBLIC_BASE_URL.replace("https://", "wss://") + "/ws/audio"
    print(f"🔗 Streaming to: {wss_url}")

    vr = VoiceResponse()

    # Start streaming audio to our WebSocket endpoint
    start = Start()
    start.append(Stream(url=wss_url))
    vr.append(start)

    vr.say("Hello. You are connected to the real time streaming demo.")
    vr.pause(length=600)  # keep call alive for demo
    vr.hangup()

    return Response(str(vr), media_type="text/xml")


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    """
    Twilio Media Streams WebSocket.
    Receives JSON messages: start, media, stop.
    """
    await ws.accept()
    print("✅ WS connected")

    media_count = 0
    stream_sid = None
    call_sid = None
    started_at = time.time()

    try:
        while True:
            raw = await ws.receive_text()

            try:
                data = json.loads(raw)
            except Exception:
                print("⚠️ Non-JSON message received (ignored)")
                continue

            event = data.get("event")

            if event == "start":
                start_info = data.get("start", {})
                stream_sid = start_info.get("streamSid")
                call_sid = start_info.get("callSid")
                print(f"▶️ start: streamSid={stream_sid} callSid={call_sid}")

            elif event == "media":
                media_count += 1

                # base64 audio payload is in:
                # payload_b64 = data["media"]["payload"]

                if media_count % 50 == 0:
                    elapsed = int(time.time() - started_at)
                    print(
                        f"🎧 media frames: {media_count} (elapsed {elapsed}s) "
                        f"streamSid={stream_sid}"
                    )

            elif event == "stop":
                stop_info = data.get("stop", {})
                print(
                    f"⏹ stop: streamSid={stop_info.get('streamSid')} "
                    f"callSid={stop_info.get('callSid')}"
                )
                print(f"✅ Total media frames received: {media_count}")
                break

            else:
                if event:
                    print(f"ℹ️ event={event}")

    except Exception as e:
        print("❌ WS ended:", repr(e))