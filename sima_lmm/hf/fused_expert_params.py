"""Per-expert views of fused MoE expert tensors (Qwen3.5-MoE).

Qwen3.5-MoE checkpoints store all routed experts of a layer in two 3-D tensors, as in
transformers' ``Qwen3_5MoeExperts``:

- ``<prefix>.mlp.experts.gate_up_proj`` with shape (num_experts, 2 * intermediate, hidden);
  expert ``e`` computes ``gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)``, so the
  first ``intermediate`` rows are ``gate_proj`` and the rest ``up_proj``.
- ``<prefix>.mlp.experts.down_proj`` with shape (num_experts, hidden, intermediate).

llima's expert graphs and the expert record generator use the per-expert names
``<prefix>.mlp.experts.<e>.{gate_proj,up_proj,down_proj}.weight`` (Qwen3-Next, OLMoE layout).
This module maps those names onto the fused tensors and slices a single expert, so a request
reads only that expert's rows from the safetensors shard.
"""

import re
from collections.abc import Callable

import numpy as np

_EXPERT_NAME = re.compile(
    r"(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight"
)
_FUSED_SOURCE = {"gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "down_proj": "down_proj"}


def fused_expert_source(name: str) -> tuple[str, int, str] | None:
    """Return (fused tensor name, expert index, projection) for a per-expert name, else None."""
    match = _EXPERT_NAME.fullmatch(name)
    if match is None:
        return None
    proj = match["proj"]
    return f"{match['prefix']}.{_FUSED_SOURCE[proj]}", int(match["expert"]), proj


def load_fused_expert_param(
    name: str, load_expert_slice: Callable[[str, int], np.ndarray]
) -> np.ndarray:
    """Load one expert's projection weight (out, in) from its fused tensor.

    Args:
        name: Per-expert parameter name, e.g. ``...mlp.experts.3.up_proj.weight``.
        load_expert_slice: Returns ``fused[expert]`` for a fused tensor name and expert index.

    Returns:
        The projection weight in the nn.Linear (out_features, in_features) layout.
    """
    source = fused_expert_source(name)
    assert source is not None, name
    fused_name, expert, proj = source
    weight = load_expert_slice(fused_name, expert)
    if proj == "down_proj":
        return weight
    assert weight.shape[0] % 2 == 0, (name, weight.shape)
    intermediate = weight.shape[0] // 2
    rows = slice(0, intermediate) if proj == "gate_proj" else slice(intermediate, None)
    return np.ascontiguousarray(weight[rows])
