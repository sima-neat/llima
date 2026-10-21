"""Shared audio decoding and Gemma 4's model-specific mel frontend."""

from itertools import chain

import av
import numpy as np


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


def audio_feature_constants(extractor) -> dict:
    """Export the original processor's coefficients for the C++ frontend."""
    if extractor.dither or extractor.preemphasis or extractor.input_scale_factor != 1:
        raise ValueError("Gemma audio currently requires zero dither/preemphasis and unit input scale")
    return {
        "window": extractor.window.tolist(),
        "mel_filters": extractor.mel_filters.tolist(),
        "mel_floor": float(extractor.mel_floor),
        "per_bin_mean": None if extractor.per_bin_mean is None else extractor.per_bin_mean.reshape(-1).tolist(),
        "per_bin_stddev": None if extractor.per_bin_stddev is None else extractor.per_bin_stddev.reshape(-1).tolist(),
    }


def gemma4_audio_features(samples: np.ndarray, audio, constants: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return padded NHWC mel features and their exact per-frame validity mask."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite, nonempty mono samples")
    if len(samples) > min(30 * audio.sampling_rate, audio.num_frames * audio.hop_length):
        raise ValueError("Audio exceeds the compiled token budget")
    left = audio.frame_length // 2
    padded_length = (len(samples) + 127) // 128 * 128
    waveform = np.pad(samples, (left, padded_length - len(samples)))
    count = (len(waveform) - audio.frame_length - 1) // audio.hop_length + 1
    if count <= 0:
        raise ValueError("Audio is too short for a Gemma analysis frame")
    frames = np.lib.stride_tricks.sliding_window_view(waveform, audio.frame_length + 1)[::audio.hop_length, :-1]
    spectrum = np.abs(np.fft.rfft(frames * np.asarray(constants["window"], np.float32), n=audio.fft_length))
    mel = np.log(spectrum @ np.asarray(constants["mel_filters"]) + constants["mel_floor"])
    if constants["per_bin_mean"] is not None:
        mel -= np.asarray(constants["per_bin_mean"])
    if constants["per_bin_stddev"] is not None:
        mel /= np.asarray(constants["per_bin_stddev"])
    valid = np.arange(count) * audio.hop_length + audio.frame_length < left + len(samples)
    if not valid[::4].any():
        raise ValueError("Audio is too short to produce a valid audio token")
    # The sample budget already bounds valid frames; discard trailing padding.
    count = min(count, audio.num_frames)
    features = np.zeros((1, 1, audio.num_frames, audio.num_mel_bins), np.float32)
    mask = np.zeros((1, 1, audio.num_frames, 1), np.float32)
    features[0, 0, :count] = mel[:count].astype(np.float32) * valid[:count, None]
    mask[0, 0, :count, 0] = valid[:count]
    return features, mask
