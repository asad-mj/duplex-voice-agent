import streamlit as st
import numpy as np
from src.vad import AcousticVAD
from src.state_machine import DuplexSessionController
from src.synthesizer import StreamingPCMGenerator

st.set_page_config(page_title="Full-Duplex Voice AI Workbench", page_icon="🎙️", layout="wide")
st.title("🎙️ Full-Duplex Voice Agent & Barge-In Telemetry Workbench")
st.markdown("Simulate real-time **Spectral Voice Activity Detection (VAD)** and **Mid-Sentence User Barge-In Interruption**.")

vad = AcousticVAD()
synth = StreamingPCMGenerator()

col1, col2, col3 = st.columns(3)
signal_mode = col1.selectbox("Incoming Audio Stream Type", ["Human Speech Harmonic (600 Hz)", "Background White Noise", "Digital Silence"])
agent_speaking = col2.checkbox("Agent Currently Speaking (Active TTS Stream)", value=True)
amplitude = col3.slider("Input Signal Amplitude", min_value=0.01, max_value=0.90, value=0.40, step=0.05)

if st.button("Simulate 32ms Full-Duplex Audio Frame", type="primary"):
    if signal_mode == "Human Speech Harmonic (600 Hz)":
        frame = synth.generate_harmonic_frame(duration_ms=32, f0_hz=600.0, amplitude=amplitude)
    elif signal_mode == "Background White Noise":
        rng = np.random.default_rng(42)
        frame = rng.normal(0.0, 0.01, size=512).astype(np.float32)
    else:
        frame = np.zeros(512, dtype=np.float32)

    controller = DuplexSessionController(barge_in_frame_trigger=1)
    if agent_speaking:
        controller.start_agent_response()

    vad_metrics = vad.analyze_frame(frame)
    step = controller.process_vad_decision(vad_metrics.is_speech)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VAD Speech Detected", "🟢 YES" if vad_metrics.is_speech else "⚪ NO")
    m2.metric("Session State", step.state.value)
    m3.metric("Barge-In Interruption", "⚡ TRIGGERED" if step.barge_in_triggered else "None")
    m4.metric("Speech Band Ratio", f"{vad_metrics.speech_band_ratio * 100:.1f}%")

    st.subheader("📈 32ms PCM Waveform (512 Samples @ 16kHz)")
    st.line_chart(frame[:200])
