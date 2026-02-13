import os, json, time
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Start, Stream

app = FastAPI()

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")  # https://<app>.up.railway.app

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/voice/incoming")
async def voice_incoming(request: Request):
    if not PUBLIC_BASE_URL:
        vr = VoiceResponse()
        vr.say("Configuration error. PUBLIC BASE URL not set.")
        vr.hangup()
        return Response(str(vr), media_type="text/xml")

    wss_url = PUBLIC_BASE_URL.replace("https://", "wss://") + "/ws/audio"

    vr = VoiceResponse()
    start = Start()
    start.append(Stream(url=wss_url))
    vr.append(start)

    vr.say("Hello. You are connected to the real time streaming demo.")
    vr.pause(length=600)
    vr.hangup()

    return Response(str(vr), media_type="text/xml")

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    print("✅ WS connected")
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")
            if event == "start":
                print("▶️ start:", data.get("start", {}).get("streamSid"))
            elif event == "media":
                # base64 audio payload is in data["media"]["payload"]
                pass
            elif event == "stop":
                print("⏹ stop:", data.get("stop", {}).get("streamSid"))
            else:
                print("ℹ️ event:", event)
    except Exception as e:
        print("❌ WS ended:", repr(e))