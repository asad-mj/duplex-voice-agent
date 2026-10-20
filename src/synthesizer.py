import numpy as np
from typing import List

class StreamingPCMGenerator:
    """
    Synthesizes 16kHz harmonic speech-band audio frames for low-latency
    WebSocket duplex streaming and testing.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def generate_harmonic_frame(
        self,
        duration_ms: int = 32,
        f0_hz: float = 440.0,
        amplitude: float = 0.4
    ) -> np.ndarray:
        """Generates a vocal-tract harmonic wave (F0 + F1 + F2 formants)."""
        n_samples = int(self.sample_rate * (duration_ms / 1000.0))
        t = np.arange(n_samples, dtype=np.float32) / self.sample_rate

        wave = (
            0.6 * np.sin(2.0 * np.pi * f0_hz * t)
            + 0.3 * np.sin(2.0 * np.pi * (f0_hz * 2.0) * t)
            + 0.1 * np.sin(2.0 * np.pi * (f0_hz * 3.0) * t)
        )
        return (amplitude * wave).astype(np.float32)

    def chunk_stream(self, samples: np.ndarray, chunk_size: int = 512) -> List[List[float]]:
        return [
            samples[i : i + chunk_size].tolist()
            for i in range(0, len(samples), chunk_size)
        ]
