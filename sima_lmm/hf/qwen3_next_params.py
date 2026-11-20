"""Qwen3-Next weight remapping onto the Qwen3.5 Gated DeltaNet parameter names.

Qwen3-Next stores the DeltaNet input projections fused and grouped per key head:
``in_proj_qkvz`` rows are ``[q(hk), k(hk), v(r * hv), z(r * hv)]`` for each of the
``num_k_heads`` groups, and ``in_proj_ba`` rows are ``[b(r), a(r)]`` per group, where
``r = num_v_heads / num_k_heads``. This mirrors ``fix_query_key_value_ordering`` in
transformers' ``modeling_qwen3_next``.

The llima linear-attention graph (``LanguageLinearModel``) consumes the Qwen3.5
layout: ``in_proj_qkv`` with all query heads, then all key heads, then all value heads,
and separate ``in_proj_z``, ``in_proj_b`` and ``in_proj_a``. Value heads keep the
group-major order ``g * r + j``, which matches the graph's ``repeat_interleave`` key
sharing (value head ``h`` uses key head ``h // r``).
"""

from dataclasses import dataclass

import numpy as np

_QKVZ_SUFFIX = ".linear_attn.in_proj_qkvz.weight"
_BA_SUFFIX = ".linear_attn.in_proj_ba.weight"
_DERIVED_SUFFIXES = {
    ".linear_attn.in_proj_qkv.weight": (_QKVZ_SUFFIX, "qkv"),
    ".linear_attn.in_proj_z.weight": (_QKVZ_SUFFIX, "z"),
    ".linear_attn.in_proj_b.weight": (_BA_SUFFIX, "b"),
    ".linear_attn.in_proj_a.weight": (_BA_SUFFIX, "a"),
}


@dataclass(frozen=True)
class Qwen3NextLinearDims:
    """Gated DeltaNet head dimensions read from a Qwen3-Next config."""

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int

    @staticmethod
    def from_config(config: dict) -> "Qwen3NextLinearDims":
        return Qwen3NextLinearDims(
            num_k_heads=config["linear_num_key_heads"],
            num_v_heads=config["linear_num_value_heads"],
            head_k_dim=config["linear_key_head_dim"],
            head_v_dim=config["linear_value_head_dim"],
        )

    @property
    def ratio(self) -> int:
        assert self.num_v_heads % self.num_k_heads == 0
        return self.num_v_heads // self.num_k_heads


def derived_source(name: str) -> tuple[str, str] | None:
    """Return (fused source name, component) for a derived Qwen3.5-style name, else None."""
    for suffix, (src_suffix, component) in _DERIVED_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)] + src_suffix, component
    return None


def split_qkvz(weight: np.ndarray, dims: Qwen3NextLinearDims) -> dict[str, np.ndarray]:
    """Split a grouped ``in_proj_qkvz`` weight (out, in) into flat qkv and z weights."""
    g, hk, hv, r = dims.num_k_heads, dims.head_k_dim, dims.head_v_dim, dims.ratio
    per_group = 2 * hk + 2 * r * hv
    assert weight.shape[0] == g * per_group, (weight.shape, dims)
    grouped = weight.reshape(g, per_group, weight.shape[1])
    q = grouped[:, :hk]
    k = grouped[:, hk:2 * hk]
    v = grouped[:, 2 * hk:2 * hk + r * hv]
    z = grouped[:, 2 * hk + r * hv:]
    flat = lambda t: t.reshape(-1, weight.shape[1])
    return {
        "qkv": np.ascontiguousarray(np.concatenate([flat(q), flat(k), flat(v)], axis=0)),
        "z": np.ascontiguousarray(flat(z)),
    }


def split_ba(weight: np.ndarray, dims: Qwen3NextLinearDims) -> dict[str, np.ndarray]:
    """Split a grouped ``in_proj_ba`` weight (out, in) into b and a weights."""
    g, r = dims.num_k_heads, dims.ratio
    assert weight.shape[0] == g * 2 * r, (weight.shape, dims)
    grouped = weight.reshape(g, 2 * r, weight.shape[1])
    return {
        "b": np.ascontiguousarray(grouped[:, :r].reshape(-1, weight.shape[1])),
        "a": np.ascontiguousarray(grouped[:, r:].reshape(-1, weight.shape[1])),
    }


def derive_param(name: str, load_fused, dims: Qwen3NextLinearDims) -> np.ndarray:
    """Load the fused source of a derived name and return the requested component."""
    source = derived_source(name)
    assert source is not None, name
    src_name, component = source
    fused = load_fused(src_name)
    if isinstance(fused, tuple):
        raise NotImplementedError(f"{src_name}: pre-quantized Qwen3-Next weights are not supported")
    parts = split_qkvz(fused, dims) if src_name.endswith(_QKVZ_SUFFIX) else split_ba(fused, dims)
    return parts[component]
