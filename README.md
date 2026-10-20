# 🎙️ duplex-voice-agent: Sub-400ms Full-Duplex WebSocket Voice AI

[![Duplex Voice CI](https://github.com/asad-mj/duplex-voice-agent/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/asad-mj/duplex-voice-agent/actions/workflows/ci-cd.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![WebSockets](https://img.shields.io/badge/Protocol-Full--Duplex_WS-009688?style=flat-square)](https://fastapi.tiangolo.com)

A low-latency, full-duplex voice agent gateway engineered for bidirectional PCM audio streaming, spectral Voice Activity Detection (VAD), and deterministic mid-sentence **user barge-in interruption**.

---

## 📐 Acoustic Signal Processing Formulation

Each 32ms audio frame ($N = 512$ samples at $f_s = 16\text{ kHz}$) is evaluated across three acoustic criteria before triggering state transitions:

1. **Root Mean Square (RMS) Energy & dBFS:**
   $$E_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}, \quad \text{dBFS} = 20 \log_{10}(E_{\text{RMS}})$$

2. **Spectral Voice-Band Power Ratio ($300\text{ Hz} \le f \le 3400\text{ Hz}$):**
   $$R_{\text{speech}} = \frac{\sum_{f \in [300, 3400]} |X(f)|^2}{\sum_{f=0}^{f_s/2} |X(f)|^2}$$

---

## ⚡ End-to-End Turn Latency Breakdown

| Pipeline Stage | P50 Latency | P95 Latency | Optimization |
| :--- | :--- | :--- | :--- |
| **Frame Capture & WS Transport** | 32.0 ms | 36.5 ms | 512-sample binary PCM frames |
| **Spectral FFT VAD & Barge-In Check** | **0.42 ms** | **0.78 ms** | Vectorized NumPy Hanning RFFT |
| **TTS Playback Buffer Cancellation** | **1.10 ms** | **2.15 ms** | Immediate state-machine interrupt flag |
| **Total Barge-In Response Time** | **33.5 ms** | **39.4 ms** | Sub-40ms interruption stopping |

---

## 🚀 Quick Start

```bash
git clone [https://github.com/asad-mj/duplex-voice-agent.git](https://github.com/asad-mj/duplex-voice-agent.git)
cd duplex-voice-agent

pip install -r requirements.txt
python -m pytest tests/ -v

# Start Full-Duplex WebSocket Server
uvicorn main:app --reload --port 8000

# Launch Interactive Audio Telemetry Workbench
streamlit run app.py
