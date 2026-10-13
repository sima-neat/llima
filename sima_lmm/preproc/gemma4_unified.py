"""Aligned image windows and self-attention masks for Gemma 4 Unified."""
from math import sqrt

import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms.v2 import functional as tvF

from sima_lmm.config.vlm_config import VisionModelConfig
from sima_lmm.utils import round_up_to
from sima_lmm.preproc.audio_preprocessing import load_audio


def resize_image(image: Image.Image, config: VisionModelConfig) -> Image.Image:
    """Downscale to the token ceiling and align to merged patches without upscaling."""
    side = config.patch_size * config.spatial_merge_size
    budget = config.max_soft_tokens
    if side <= 0 or budget <= 0:
        raise ValueError("Unified images require a positive patch size and token budget")
    image = image.convert("RGB")
    width, height = image.size
    if width < side or height < side:
        image = ImageOps.expand(image, (0, 0, max(0, side - width), max(0, side - height)), fill=0)
        width, height = image.size
    factor = min(1.0, sqrt(budget * side * side / (width * height)))
    target_h = max(side, int(height * factor / side) * side)
    target_w = max(side, int(width * factor / side) * side)
    if target_h == side:
        target_w = min(target_w, budget * side)
    if target_w == side:
        target_h = min(target_h, budget * side)
    if image.size == (target_w, target_h):
        return image
    pixels = tvF.resize(tvF.pil_to_tensor(image), [target_h, target_w],
                       interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return tvF.to_pil_image(pixels)


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


def cache_view(query_begin: int, rows: int, context: int, window: int,
               spans: list[tuple[int, int]]) -> tuple[int, int]:
    """Smallest aligned KV bucket covering causal and same-image attention."""
    begin = max(0, query_begin + 1 - window)
    end = query_begin + rows
    for image_begin, image_end in spans:
        if image_begin < query_begin + rows and image_end > query_begin:
            end = max(end, image_end)
    capacity = round_up_to(end - begin, rows)
    return min(begin, context - capacity), capacity


def attention_mask(query_begin: int, rows: int, key_begin: int, capacity: int,
                   valid_end: int, spans: list[tuple[int, int]], window: int = 0) -> np.ndarray:
    """Causal attention; sliding layers also attend within the same image."""
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
