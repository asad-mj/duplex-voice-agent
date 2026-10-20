import numpy as np
from pydantic import BaseModel

class VADMetrics(BaseModel):
    is_speech: bool
    rms_energy: float
    db_fs: float
    zero_crossing_rate: float
    speech_band_ratio: float

class AcousticVAD:
    """
    Real-time Voice Activity Detector using Short-Time Energy (RMS),
    Zero-Crossing Rate (ZCR), and FFT Spectral Voice-Band Ratio (300Hz - 3400Hz).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        rms_threshold: float = 0.025,
        speech_ratio_threshold: float = 0.45
    ):
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.speech_ratio_threshold = speech_ratio_threshold

    def analyze_frame(self, audio_samples: np.ndarray) -> VADMetrics:
        """
        Evaluates a 1D float32 audio frame in [-1.0, 1.0].
        """
        if audio_samples.size == 0:
            return VADMetrics(
                is_speech=False,
                rms_energy=0.0,
                db_fs=-100.0,
                zero_crossing_rate=0.0,
                speech_band_ratio=0.0
            )

        samples = audio_samples.astype(np.float32)

        # 1. Root Mean Square (RMS) Energy & dBFS
        rms = float(np.sqrt(np.mean(samples ** 2)))
        db_fs = float(20.0 * np.log10(max(rms, 1e-5)))

        # 2. Zero-Crossing Rate (ZCR)
        signs = np.sign(samples)
        signs[signs == 0] = -1
        zcr = float(np.mean(signs[:-1] != signs[1:])) if samples.size > 1 else 0.0

        # 3. FFT Spectral Speech Band Energy Ratio (300 Hz to 3400 Hz)
        windowed = samples * np.hanning(samples.size)
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)

        total_power = float(np.sum(spectrum))
        if total_power < 1e-9:
            speech_ratio = 0.0
        else:
            band_mask = (freqs >= 300.0) & (freqs <= 3400.0)
            speech_power = float(np.sum(spectrum[band_mask]))
            speech_ratio = speech_power / total_power

        is_speech = (rms >= self.rms_threshold) and (speech_ratio >= self.speech_ratio_threshold)

        return VADMetrics(
            is_speech=bool(is_speech),
            rms_energy=round(rms, 5),
            db_fs=round(db_fs, 2),
            zero_crossing_rate=round(zcr, 4),
            speech_band_ratio=round(speech_ratio, 4)
        )
