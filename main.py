import os
import json
import time
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Start, Stream

app = FastAPI()

# Railway variable you already added
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")  # e.g. https://multiagent-voice-job-platform.up.railway.app


@app.get("/")
def root():
    # So your base URL doesn't show {"detail":"Not Found"}
    return {"status": "running", "service": "twilio-media-streams-demo"}


@app.get("/health")
def health():
    # Railway healthcheck can call this
    return {"ok": True, "ts": int(time.time())}


@app.post("/voice/incoming")
async def voice_incoming(request: Request):
    """
    Twilio hits this when someone calls your Twilio phone number.
    We respond with TwiML that starts Media Streams to our WSS endpoint.
    """
    body = (await request.body()).decode("utf-8", errors="ignore")
    print("Raw body:", body)
    call_sid = form.get("CallSid", "unknown")
    from_number = form.get("From", "unknown")
    to_number = form.get("To", "unknown")

    print(f"📞 Incoming call: CallSid={call_sid} From={from_number} To={to_number}")

    if not PUBLIC_BASE_URL:
        vr = VoiceResponse()
        vr.say("Configuration error. Public base URL is not set.")
        vr.hangup()
        return Response(str(vr), media_type="text/xml")

    # Twilio needs WSS for media streams
    wss_url = PUBLIC_BASE_URL.replace("https://", "wss://") + "/ws/audio"
    print(f"🔗 Streaming to: {wss_url}")

    vr = VoiceResponse()

    # Start streaming audio to WebSocket
    start = Start()
    start.append(Stream(url=wss_url))
    vr.append(start)

    vr.say("Hello. You are connected to the real time streaming demo.")
    # Keep call alive for demo (seconds). Increase if you want longer calls.
    vr.pause(length=600)

    # Optional: you can remove hangup; pause ending will naturally end flow
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

                # For later: base64 audio payload is here
                # payload_b64 = data["media"]["payload"]

                # Print every 50 frames so logs don't explode
                if media_count % 50 == 0:
                    elapsed = int(time.time() - started_at)
                    print(f"🎧 media frames: {media_count} (elapsed {elapsed}s) streamSid={stream_sid}")

            elif event == "stop":
                stop_info = data.get("stop", {})
                print(f"⏹ stop: streamSid={stop_info.get('streamSid')} callSid={stop_info.get('callSid')}")
                print(f"✅ Total media frames received: {media_count}")
                break

            else:
                # Sometimes you may receive other events; log lightly
                if event:
                    print(f"ℹ️ event={event}")

    except Exception as e:
        print("❌ WS ended:", repr(e))