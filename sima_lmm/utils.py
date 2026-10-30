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
) -> np.ndarray:
    """
    Computes "longrope" scaling for RoPE.

    Which factor applies is decided by "max_position_embeddings" in inference against
    "original_max_position_embeddings" in pretraining.

    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L322.

    The companion post-processing scale on cos/sin is not computed here: RoPEConfig derives
    it once as rope_scaling.attention_factor, and both calc_freq_real_imag and the runtime's
    rope_utils.cpp read it from there so host and DevKit cannot disagree.

    Args:
        inv_freq: Precomputed RoPE inverse frequencies without scaling.
        long_factor: The scaling factor for long context length.
        short_factor: The scaling factor for short context length.
        max_position_embeddings: The context length used in inferencing.
        original_max_position_embeddings: The context length used in training.

    Returns:
        Scaled inverse frequencies for the RoPE embeddings, the same length as long_factor
        or short_factor.
    """
    n_freqs = len(inv_freq)
    assert len(long_factor) <= n_freqs
    assert len(short_factor) <= n_freqs

    if max_position_embeddings > original_max_position_embeddings:
        ext_factors = np.array(long_factor, dtype=np.float32)
    else:
        ext_factors = np.array(short_factor, dtype=np.float32)
    return inv_freq[:len(ext_factors)] / ext_factors


def rope_channel_layout(
    rope_dimension_count: int, head_dim: int, proportional: bool = False
) -> tuple[int, int]:
    """
    Channel layout of a rotation over head_dim channels: (half_rope_dim, imag_start).

    The rotation pairs channel i with channel imag_start + i for i < half_rope_dim and leaves
    every other channel unchanged. Partial RoPE rotates a prefix of the head and passes the
    tail through; proportional RoPE (Gemma4 full-attention layers) keeps the imaginary half
    at head_dim // 2, so an unrotated segment sits between the halves.

    The frequency tables the language pre models consume are laid out to match: each
    frequency appears on both channels of its pair, with the sin of the real channel i negated,
    and unrotated channels hold the neutral rotation of cos 1 and sin 0. This is the compiler
    side of that contract; the runtime builds the same layout in rope_utils.cpp
    rope_channel_layout.
    """
    half_rope_dim = rope_dimension_count // 2
    imag_start = head_dim // 2 if proportional else half_rope_dim
    assert imag_start + half_rope_dim <= head_dim
    return half_rope_dim, imag_start


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
    max_num_tokens: int, rope_type: str, theta: float, rope_dimension_count: int,
    head_dim: int, scaling_cfg: dict, idx_base: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the head_dim-wide cos and sin tables that the language pre models consume.

    Returns two (1, 1, max_num_tokens, head_dim) arrays in HWC layout.

    This is the compiler side of the frequency-table contract. The runtime side is
    rope_utils.cpp calc_freq_real_imag, which the DevKit uploads into the same freq_real /
    freq_imag inputs; the two must stay numerically identical, so every rope_type either
    both accept or both reject.
    """
    # Proportional RoPE (Gemma4 full-attention layers) spaces the active frequencies over the
    # full head dim; every other variant spaces them over the active rotary dim itself.
    proportional = rope_type == "proportional"
    frequency_denominator = head_dim if proportional else rope_dimension_count
    inv_freq = 1.0 / (
        theta ** (
            np.arange(0, rope_dimension_count, 2, dtype=np.float64) / frequency_denominator
        )
    )
    inv_freq = inv_freq.astype(np.float32)
    # LongRoPE and YaRN scale cos/sin; 1.0 for every other type.
    attention_factor = 1.0
    factor = scaling_cfg.get("factor")
    match rope_type:
        # "mrope" and "proportional" select a channel layout and a frequency spacing, not a
        # frequency rescaling, so the inverse frequencies stay unscaled here. mRoPE's
        # per-section token indices are a separate, later step: the DevKit applies them in
        # rope_utils.cpp calc_mrope_with_image, and host evaluation uses this text-only
        # table as it stands.
        case "" | "default" | "mrope" | "proportional":
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
            inv_freq = _rope_scaling_longrope(
                inv_freq,
                scaling_cfg.get("long_factor"),
                scaling_cfg.get("short_factor"),
                max_num_tokens,
                scaling_cfg.get("original_max_position_embeddings")
            )
            attention_factor = scaling_cfg.get("attention_factor") or 1.0
        case "yarn":
            inv_freq, attention_factor = _rope_scaling_yarn(
                inv_freq, factor,
                scaling_cfg.get("beta_fast", 32.0),
                scaling_cfg.get("beta_slow", 1.0),
                scaling_cfg.get("original_max_position_embeddings"),
                theta, rope_dimension_count,
                scaling_cfg.get("truncate", True),
                scaling_cfg.get("attention_factor"),
            )
        case _:
            raise ValueError(f"{rope_type} rope type is not supported.")

    half_rope_dim, imag_start = rope_channel_layout(
        rope_dimension_count, head_dim, proportional
    )
    n_scaled_freq = min(len(inv_freq), half_rope_dim)
    t = np.arange(idx_base, max_num_tokens + idx_base, dtype=np.float32)
    angles = np.outer(inv_freq[:n_scaled_freq], t).reshape(n_scaled_freq, max_num_tokens)

    # Transpose to HWC layout, then copy so that the last dimension is contiguous.
    angles = np.expand_dims(angles.transpose(1, 0), axis=(0, 1)).copy()

    # cos and sin stay in float32, attention factor included, to match the runtime, which
    # evaluates and scales them in float32 on the float32 angles; the tables are consumed as
    # bfloat16, so a wider intermediate would buy nothing and cost bit-exactness against the
    # DevKit.
    #
    # LongRoPE and YaRN scale the rotated channels; the neutral channels below must stay at
    # cos 1 and sin 0, so apply it here rather than to the finished tables.
    cos = np.cos(angles) * np.float32(attention_factor)
    sin = np.sin(angles) * np.float32(attention_factor)

    # Each frequency lands on both channels of its rotation pair. The pre model multiplies sin
    # by a pure channel swap of the data, so the sin of the real channel carries the minus
    # sign of the rotation: out[i] = x[i] * cos - x[imag_start + i] * sin. Channels the
    # rotation does not reach keep the neutral rotation of cos 1 and sin 0: the tail of a
    # partial rotation, the middle segment of a proportional one, and any frequency that the
    # scaling dropped.
    re = np.ones((1, 1, max_num_tokens, head_dim), dtype=np.float32)
    im = np.zeros((1, 1, max_num_tokens, head_dim), dtype=np.float32)
    for start, sign in ((0, -1), (imag_start, 1)):
        re[..., start:start + n_scaled_freq] = cos
        im[..., start:start + n_scaled_freq] = sign * sin
    return re, im
