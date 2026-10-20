import numpy as np
from fastapi.testclient import TestClient
from main import app
from src.vad import AcousticVAD
from src.state_machine import DuplexSessionController, DuplexState
from src.synthesizer import StreamingPCMGenerator

def test_api_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

def test_vad_silence_vs_speech():
    vad = AcousticVAD()
    synth = StreamingPCMGenerator()

    silence = np.zeros(512, dtype=np.float32)
    sil_res = vad.analyze_frame(silence)
    assert sil_res.is_speech is False

    speech_wave = synth.generate_harmonic_frame(duration_ms=32, f0_hz=500.0, amplitude=0.45)
    sp_res = vad.analyze_frame(speech_wave)
    assert sp_res.is_speech is True
    assert sp_res.speech_band_ratio > 0.80

def test_barge_in_state_machine():
    controller = DuplexSessionController(barge_in_frame_trigger=2)
    controller.start_agent_response()
    assert controller.state == DuplexState.AGENT_SPEAKING

    # Frame 1 of user speech while agent speaks
    step1 = controller.process_vad_decision(is_speech=True)
    assert step1.barge_in_triggered is False

    # Frame 2 triggers immediate barge-in interruption
    step2 = controller.process_vad_decision(is_speech=True)
    assert step2.barge_in_triggered is True
    assert step2.cancel_agent_playback is True
    assert step2.state == DuplexState.INTERRUPTED
    assert step2.interruption_count == 1

def test_websocket_duplex_barge_in():
    synth = StreamingPCMGenerator()
    speech_samples = synth.generate_harmonic_frame(duration_ms=32, f0_hz=500.0, amplitude=0.5).tolist()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/duplex") as ws:
            ws.send_json({"command": "start_agent_tts"})
            ack = ws.receive_json()
            assert ack["state"] == "AGENT_SPEAKING"

            ws.send_json({"samples": speech_samples})
            _ = ws.receive_json()

            ws.send_json({"samples": speech_samples})
            ev2 = ws.receive_json()
            assert ev2["session"]["barge_in_triggered"] is True
            assert ev2["session"]["state"] == "INTERRUPTED"
