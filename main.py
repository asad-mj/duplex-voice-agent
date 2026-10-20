import numpy as np
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from src.vad import AcousticVAD, VADMetrics
from src.state_machine import DuplexSessionController, SessionStepResult

app = FastAPI(
    title="Full-Duplex Voice Agent & Barge-In Gateway",
    description="Real-time WebSocket voice streaming service with spectral VAD and mid-sentence barge-in cancellation.",
    version="1.0.0"
)

vad_engine = AcousticVAD()

class AudioFrameRequest(BaseModel):
    samples: List[float] = Field(..., min_length=16, description="Normalized float32 PCM samples")
    agent_currently_speaking: bool = False

class FrameTelemetryResponse(BaseModel):
    vad: VADMetrics
    session: SessionStepResult

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "service": "duplex-voice-agent", "version": "1.0.0"}

@app.post("/analyze-frame", response_model=FrameTelemetryResponse, tags=["Audio Telemetry"])
def analyze_single_frame(payload: AudioFrameRequest):
    controller = DuplexSessionController(barge_in_frame_trigger=1)
    if payload.agent_currently_speaking:
        controller.start_agent_response()

    arr = np.array(payload.samples, dtype=np.float32)
    vad_res = vad_engine.analyze_frame(arr)
    session_res = controller.process_vad_decision(vad_res.is_speech)

    return FrameTelemetryResponse(vad=vad_res, session=session_res)

@app.websocket("/ws/duplex")
async def duplex_voice_socket(websocket: WebSocket):
    await websocket.accept()
    controller = DuplexSessionController(barge_in_frame_trigger=2)

    try:
        while True:
            message = await websocket.receive_json()
            command = message.get("command")

            if command == "start_agent_tts":
                controller.start_agent_response()
                await websocket.send_json({
                    "event": "agent_tts_started",
                    "state": controller.state
                })
                continue

            samples = message.get("samples", [])
            arr = np.array(samples, dtype=np.float32)
            vad_res = vad_engine.analyze_frame(arr)
            step_res = controller.process_vad_decision(vad_res.is_speech)

            await websocket.send_json({
                "event": "frame_processed",
                "vad": vad_res.model_dump(),
                "session": step_res.model_dump()
            })
    except WebSocketDisconnect:
        pass
