"""Aligned image windows and self-attention masks for Gemma 4 Unified."""
import numpy as np

from sima_lmm.utils import round_up_to
from sima_lmm.preproc.audio_preprocessing import load_audio


def audio_frames(path, config) -> np.ndarray:
    """Keep the zero-padded final waveform frame valid for the Unified decoder."""
    if config.samples_per_token <= 0 or config.sampling_rate != 16000:
        raise ValueError("Unified audio requires a positive frame size and 16-kHz samples")
    samples = load_audio(path, config.sampling_rate, config.samples_per_token * config.max_tokens)
    return np.pad(samples, (0, (-len(samples)) % config.samples_per_token)).reshape(-1, config.samples_per_token)


def image_windows(spans: list[tuple[int, int]], group_size: int) -> list[tuple[int, int]]:
    """Merge overlapping chunk-aligned spans; touching windows remain separate."""
    windows = []
    previous_end = 0
    for begin, end in spans:
        if begin < previous_end or end <= begin:
            raise ValueError("Image spans must be nonempty, ordered and disjoint")
        previous_end = end
        begin, end = begin // group_size * group_size, round_up_to(end, group_size)
        if windows and begin < windows[-1][1]:
            windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
        else:
            windows.append((begin, end))
    return windows


def attention_mask(query_begin: int, rows: int, key_begin: int, capacity: int,
                   valid_end: int, spans: list[tuple[int, int]], window: int = 0) -> np.ndarray:
    """Causal attention plus future keys in the query's own image block."""
    q = np.arange(query_begin, query_begin + rows)[:, None]
    k = np.arange(key_begin, key_begin + capacity)[None, :]
    allowed = k <= q
    if window:
        allowed &= k > q - window
    for begin, end in spans:
        allowed |= (q >= begin) & (q < end) & (k >= begin) & (k < end)
    allowed &= (k < valid_end) & (q < valid_end)
    # Padded query outputs are discarded, but softmax still needs a finite row.
    allowed[q[:, 0] >= valid_end, 0] = True
    return np.where(allowed, 0, -3.3895313892515355e38).astype(np.float32)[None, None]
