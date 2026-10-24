"""Shared audio decoding and Gemma 4's model-specific mel frontend."""

from itertools import chain

import av
import numpy as np

from sima_lmm.config.vlm_config import AudioPreprocessorConfig


def load_audio(file, sr: int = 16000, max_samples: int | None = None) -> np.ndarray:
    """Decode mono PCM, flushing delayed samples and preserving actual duration."""
    parts, count = [], 0
    with av.open(str(file)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        for frame in chain(container.decode(audio=0), (None,)):
            if frame is not None:
                frame.pts = None
            for resampled in resampler.resample(frame):
                values = resampled.to_ndarray().flatten()
                count += len(values)
                if max_samples is not None and count > max_samples:
                    raise ValueError("Audio exceeds the compiled token budget")
                parts.append(values)
    if not count:
        raise ValueError("Audio input is empty")
    return np.concatenate(parts).astype(np.float32) / 32768.0


def gemma4_audio_features(
    samples: np.ndarray, audio, frontend: AudioPreprocessorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded NHWC mel features and their exact per-frame validity mask."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite, nonempty mono samples")
    frame_length, hop_length = frontend.frame_length, frontend.hop_length
    if len(samples) > min(30 * frontend.sampling_rate, audio.num_frames * hop_length):
        raise ValueError("Audio exceeds the compiled token budget")
    left = frame_length // 2
    padded_length = (len(samples) + 127) // 128 * 128
    waveform = np.pad(samples, (left, padded_length - len(samples)))
    count = (len(waveform) - frame_length - 1) // hop_length + 1
    if count <= 0:
        raise ValueError("Audio is too short for a Gemma analysis frame")
    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_length + 1)[::hop_length, :-1]
    spectrum = np.abs(np.fft.rfft(frames * np.asarray(frontend.window, np.float32), n=frontend.fft_length))
    mel = np.log(spectrum @ np.asarray(frontend.mel_filters) + frontend.mel_floor)
    if frontend.per_bin_mean is not None:
        mel -= np.asarray(frontend.per_bin_mean)
    if frontend.per_bin_stddev is not None:
        mel /= np.asarray(frontend.per_bin_stddev)
    valid = np.arange(count) * hop_length + frame_length < left + len(samples)
    if not valid[::4].any():
        raise ValueError("Audio is too short to produce a valid audio token")
    # The sample budget already bounds valid frames; discard trailing padding.
    count = min(count, audio.num_frames)
    features = np.zeros((1, 1, audio.num_frames, frontend.num_mel_bins), np.float32)
    mask = np.zeros((1, 1, audio.num_frames, 1), np.float32)
    features[0, 0, :count] = mel[:count].astype(np.float32) * valid[:count, None]
    mask[0, 0, :count, 0] = valid[:count]
    return features, mask
