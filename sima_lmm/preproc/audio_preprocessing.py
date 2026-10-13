"""Shared file decoding to mono PCM; each model handles its own framing."""
import av
import numpy as np


def load_audio(file, sr: int = 16000, max_samples: int | None = None) -> np.ndarray:
    """Decode/resample to mono float32 PCM, preserving the actual sample count."""
    parts = []
    count = 0
    with av.open(str(file)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        for frame in container.decode(audio=0):
            frame.pts = None
            for resampled in resampler.resample(frame):
                samples = resampled.to_ndarray().flatten()
                count += len(samples)
                if max_samples is not None and count > max_samples:
                    raise ValueError("Audio exceeds the compiled token budget")
                parts.append(samples)
        for resampled in resampler.resample(None):
            samples = resampled.to_ndarray().flatten()
            count += len(samples)
            if max_samples is not None and count > max_samples:
                raise ValueError("Audio exceeds the compiled token budget")
            parts.append(samples)
    if not count:
        raise ValueError("Audio input is empty")
    return np.concatenate(parts).astype(np.float32) / 32768.0
