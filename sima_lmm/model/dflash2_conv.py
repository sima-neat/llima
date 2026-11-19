"""DFlash 2 grouped dynamic causal convolutions over a draft block's positions."""
import numpy as np

from sima_lmm.model.model_graph import ModelGraph, NodeOrHandle, tensor_type


class DFlash2Conv:
    """One DFlash 2 convolution: a prepare stage before a sublayer and a finish stage after it.

    Stage ``s`` computes ``out[t] = sum_o (base[s, o] + dynamic[t, s, o, g]) * x[t - o]`` with
    ``x[t - o] = 0`` before the block start. ``dynamic`` comes from ``kernel_projection`` applied to
    the sublayer input and holds one weight per group of ``group_size`` adjacent channels.
    """

    def __init__(self, graph: ModelGraph, prefix: str, kernel_size: int, group_size: int):
        self.graph = graph
        self.prefix = prefix
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.base = np.asarray(graph.parameter(f"{prefix}.base_kernel"), dtype=np.float32)
        self.dynamic = None

    def project(self, hidden: NodeOrHandle) -> None:
        """Project the per-group kernels of both stages from the sublayer input ``hidden``."""
        self.dynamic = self.graph.linear(f"{self.prefix}.kernel_projection", hidden)

    def prepare(self, hidden: NodeOrHandle) -> NodeOrHandle:
        """Project the per-group kernels from ``hidden`` and convolve it with stage zero."""
        self.project(hidden)
        return self._convolve(hidden, 0)

    def finish(self, hidden: NodeOrHandle) -> NodeOrHandle:
        """Convolve the sublayer output with stage one of the kernels prepare projected."""
        assert self.dynamic is not None, "prepare must run before finish"
        return self._convolve(hidden, 1)

    def _convolve(self, hidden: NodeOrHandle, stage: int) -> NodeOrHandle:
        graph = self.graph
        _, _, tokens, hidden_size = tensor_type(hidden).shape
        groups = hidden_size // self.group_size
        assert self.base.shape == (2, self.kernel_size, hidden_size), self.base.shape
        zero_row = None
        output = None
        for offset in range(self.kernel_size):
            if offset == 0:
                values = hidden
            else:
                if zero_row is None:
                    first = graph.slice(hidden, start=0, stop=1, axis=2)
                    zero_row = graph.sub(first, first)
                shifted = graph.slice(hidden, start=0, stop=tokens - offset, axis=2)
                values = graph.concat([zero_row] * offset + [shifted], 2)
            base = graph.constant(self.base[stage, offset].reshape(1, 1, 1, hidden_size))
            start = (stage * self.kernel_size + offset) * groups
            weights = graph.slice(self.dynamic, start=start, stop=start + groups, axis=3)
            grouped = graph.mul(
                graph.reshape(values, [1, tokens, groups, self.group_size]),
                graph.reshape(weights, [1, tokens, groups, 1]),
            )
            term = graph.add(
                graph.mul(values, base), graph.reshape(grouped, [1, 1, tokens, hidden_size])
            )
            output = term if output is None else graph.add(output, term)
        return output
