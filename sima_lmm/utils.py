import numpy as np


mla_row_size: int = 16
mla_num_tiles: int = 100
mla_max_num_rows_per_tile: int = 4096
mla_max_num_rows: int = mla_max_num_rows_per_tile * mla_num_tiles


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def ceil_div_row(x: int) -> int:
    return ceil_div(x, mla_row_size)


def round_up_to(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def round_up_to_row(x: int) -> int:
    return ceil_div_row(x) * mla_row_size


def _rope_scaling_llama3(
    inv_freq: np.ndarray, factor: float, low_freq_factor: float, high_freq_factor: float,
    original_max_position_embeddings: int
) -> np.ndarray:
    """
    Computes LLAMA3 RoPE scaling, which is "llama3" for rope_type in HuggingFace.

    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L385.

    Args:
        inv_freq: Precomputed RoPE inverse frequencies without scaling.
        factor: The scaling factor. The new context length is the original context length
            times the scaling factor.
        low_freq_factor: The scaling factor for low frequencies.
        high_freq_factor: The scaling factor for high frequencies.
        original_max_position_embeddings: The context length used in training.

    Returns:
        Scaled inverse frequencies for the RoPE embeddings.
    """
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor

    wavelen = 2 * np.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (
        original_max_position_embeddings / wavelen - low_freq_factor
    ) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (
        (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    )
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def _rope_scaling_longrope(
    inv_freq: np.ndarray, long_factor: list[float], short_factor: list[float],
    max_position_embeddings: int, original_max_position_embeddings: int
) -> tuple[np.ndarray, float]:
    """
    Computes "longrope" scaling for RoPE.

    The scaling factor is computed based on "max_position_embeddings" in inference and
    "original_max_position_embeddings" in pretraining.

    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L322.

    Args:
        inv_freq: Precomputed RoPE inverse frequencies without scaling.
        long_factor: The scaling factor for long context length.
        short_factor: The scaling factor for short context length.
        max_position_embeddings: The context length used in inferencing.
        original_max_position_embeddings: The context length used in training.

    Returns:
        Scaled inverse frequencies for the RoPE embeddings and post-processing scaling factor
        applied to the computed cos/sin.  The inverse frequencies have the same length as
        long_factor or short_factor.
    """
    n_freqs = len(inv_freq)
    assert len(long_factor) <= n_freqs
    assert len(short_factor) <= n_freqs

    factor = max_position_embeddings / original_max_position_embeddings
    if factor <= 1.0:
        attention_factor = 1.0
    else:
        attention_factor = np.sqrt(1 + np.log(factor) / np.log(original_max_position_embeddings))

    if max_position_embeddings > original_max_position_embeddings:
        ext_factors = np.array(long_factor, dtype=np.float32)
    else:
        ext_factors = np.array(short_factor, dtype=np.float32)
    inv_freq = inv_freq[:len(ext_factors)] / ext_factors

    return inv_freq, attention_factor


def _rope_scaling_yarn(inv_freq, factor, beta_fast, beta_slow,
                       original_max_position_embeddings, base, dim, truncate,
                       attention_factor=None):
    """YaRN: ramp inv_freq between extrapolation and interpolation; returns (inv_freq, mscale).

    attention_factor overrides the default mscale when the config declares one.
    """
    def find_correction_dim(num_rotations):
        return (dim * np.log(original_max_position_embeddings / (num_rotations * 2 * np.pi))) \
            / (2 * np.log(base))

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low, high = np.floor(low), np.ceil(high)
    low = max(low, 0.0)
    high = min(high, dim - 1)
    if high == low:
        high += 0.001

    n = len(inv_freq)  # dim // 2
    ramp = np.clip((np.arange(n, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
    extrapolation_factor = 1.0 - ramp
    interpolation = inv_freq / factor
    scaled = interpolation * (1.0 - extrapolation_factor) + inv_freq * extrapolation_factor

    if attention_factor is None:
        attention_factor = 1.0 if factor <= 1.0 else 0.1 * np.log(factor) + 1.0
    return scaled.astype(np.float32), attention_factor


def calc_freq_real_imag(
    max_num_tokens: int, rope_type: str, theta: float, head_dim: int, scaling_cfg: dict,
    idx_base: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    inv_freq = inv_freq.astype(np.float32)
    factor = scaling_cfg.get("factor")
    attention_factor = 1.0  # YaRN scales cos/sin by mscale; 1.0 for every other type.
    match rope_type:
        case "" | "default":
            pass
        case "linear":
            inv_freq /= factor
        case "llama3":
            inv_freq = _rope_scaling_llama3(
                inv_freq, factor,
                scaling_cfg.get("low_freq_factor"),
                scaling_cfg.get("high_freq_factor"),
                scaling_cfg.get("original_max_position_embeddings")
            )
        case "longrope":
            inv_freq, _ = _rope_scaling_longrope(
                inv_freq,
                scaling_cfg.get("long_factor"),
                scaling_cfg.get("short_factor"),
                max_num_tokens + idx_base,
                scaling_cfg.get("original_max_position_embeddings")
            )
        case "yarn":
            inv_freq, attention_factor = _rope_scaling_yarn(
                inv_freq, factor,
                scaling_cfg.get("beta_fast", 32.0),
                scaling_cfg.get("beta_slow", 1.0),
                scaling_cfg.get("original_max_position_embeddings"),
                theta, head_dim,
                scaling_cfg.get("truncate", True),
                scaling_cfg.get("attention_factor"),
            )
        case _:
            raise ValueError(f"{rope_type} rope type is not supported.")

    n_scaled_freq = len(inv_freq)
    t = np.arange(idx_base, max_num_tokens + idx_base, dtype=np.float32)
    inv_freq = np.outer(inv_freq, t).reshape(n_scaled_freq, max_num_tokens)

    # Return in HWC layout. Perform a copy so that the last dimension is contiguous.
    inv_freq = np.expand_dims(inv_freq.transpose(1, 0), axis=(0, 1)).copy()
    re = (np.cos(inv_freq, dtype=np.float64) * attention_factor).astype(np.float32)
    im = (np.sin(inv_freq, dtype=np.float64) * attention_factor).astype(np.float32)

    # Insert 1 for indices that are not scaled
    if n_scaled_freq < head_dim // 2:
        n_unscaled_freq = head_dim // 2 - n_scaled_freq
        re = np.pad(re, ((0, 0), (0, 0), (0, 0), (0, n_unscaled_freq)), constant_values=1)
        im = np.pad(im, ((0, 0), (0, 0), (0, 0), (0, n_unscaled_freq)), constant_values=1)
    return re, im
