"""Normal ONNX exporter for the Qwen3-TTS codec-decoder tail.

The caller provides the architecture-specific 4D wrapper at compile time.  This
module only partitions that wrapper into fixed-shape micro stages and delegates
quantization and MLA compilation to :class:`BaseModel`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file
import torch
import torch.nn as nn

from sima_lmm.model.base import BaseModel, TensorTessellateParameters


class _Qwen3TTSTailMicroStage(nn.Module):
    """One executable micro stage from a caller-supplied ``Tail4DWrapper``."""

    def __init__(self, modules: list[nn.Module], clamp_output: bool = False):
        super().__init__()
        self.modules = nn.ModuleList(modules)
        self.clamp_output = clamp_output

    def forward(self, hidden_4d: torch.Tensor) -> torch.Tensor:
        output = hidden_4d
        for module in self.modules:
            output = module(output)
        if self.clamp_output:
            output = output.clamp(min=-1, max=1)
        return output


@dataclass
class Qwen3TTSCodecTailModel(BaseModel):
    """One normal ONNX/quantize/compile micro stage of the Qwen3-TTS tail."""

    part_idx: int = 0
    qwen3tts_tail_wrapper: Callable[..., object] | None = field(default=None, kw_only=True)

    def _hf_cache_path(self) -> Path:
        if self.hf_model is None:
            raise RuntimeError("Qwen3-TTS codec-tail export requires a Hugging Face checkpoint")
        return Path(self.hf_model.hf_cache)

    def _tail_config(self) -> dict:
        return json.loads((self._hf_cache_path() / "config.json").read_text())

    def _build_tail_wrapper(self) -> nn.Module:
        if self.qwen3tts_tail_wrapper is None:
            raise RuntimeError(
                "Qwen3-TTS codec-tail compilation requires "
                "--qwen3tts-tail-wrapper MODULE:ATTRIBUTE."
            )

        # The wrapper remains a caller-provided compile-time dependency.  Only
        # Qwen's decoder checkpoint class is loaded here to supply its weights.
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2DecoderConfig,
        )
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Decoder,
        )

        hf_path = self._hf_cache_path()
        config = Qwen3TTSTokenizerV2DecoderConfig.from_pretrained(hf_path)
        decoder = Qwen3TTSTokenizerV2Decoder._from_config(config)
        wrapper = self.qwen3tts_tail_wrapper(
            SimpleNamespace(decoder=decoder),
            require_host_upsampling=True,
            device_upsampling=True,
            fixed_input_width=int(self._tail_config()["frames"]),
        ).eval()
        if not isinstance(wrapper, nn.Module):
            raise TypeError("Qwen3-TTS tail wrapper must return torch.nn.Module")

        state = load_file(hf_path / "model.safetensors")
        expected = wrapper.state_dict()
        for name, value in list(state.items()):
            target = expected.get(name)
            if (
                name.endswith(".weight")
                and value.ndim == 4
                and target is not None
                and tuple(value.shape) != tuple(target.shape)
            ):
                converted = value.permute(1, 0, 2, 3).flip(-1).contiguous()
                if tuple(converted.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"Unexpected Qwen3-TTS tail weight shape for {name}: "
                        f"{tuple(value.shape)} versus {tuple(target.shape)}"
                    )
                state[name] = converted
        missing, unexpected = wrapper.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Qwen3-TTS tail safetensors do not match the supplied Tail4DWrapper; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return wrapper

    @staticmethod
    def _micro_stages(wrapper: nn.Module) -> list[_Qwen3TTSTailMicroStage]:
        try:
            upsample = wrapper.upsample
            decoder = wrapper.decoder
        except AttributeError as exc:
            raise TypeError(
                "Qwen3-TTS tail wrapper must expose Tail4DWrapper.upsample and .decoder"
            ) from exc

        # Preserve the validated raw-runtime boundaries: each module in the
        # two upsample Sequential blocks is its own ELF stage.  Combining each
        # pair would reduce the contract from 27 stages to 25 and is therefore
        # incompatible with the deployed qwen3tts runtime.
        stages = [
            *(_Qwen3TTSTailMicroStage([module]) for module in upsample[0]),
            *(_Qwen3TTSTailMicroStage([module]) for module in upsample[1]),
            _Qwen3TTSTailMicroStage([decoder[0]]),
        ]
        for decoder_block_idx in range(1, 5):
            for block in decoder[decoder_block_idx].block:
                stages.append(_Qwen3TTSTailMicroStage([block]))
        stages.extend((
            _Qwen3TTSTailMicroStage([decoder[5]]),
            _Qwen3TTSTailMicroStage([decoder[6]], clamp_output=True),
        ))
        if len(stages) != 27:
            raise RuntimeError(
                f"Tail4DWrapper topology produced {len(stages)} micro stages, expected 27"
            )
        return stages

    def gen_onnx_files(self):
        tail_config = self._tail_config()
        wrapper = self._build_tail_wrapper()
        stages = self._micro_stages(wrapper)
        if not 0 <= self.part_idx < len(stages):
            raise ValueError(f"Invalid Qwen3-TTS tail micro-stage index {self.part_idx}")

        input_shape = tuple(int(v) for v in tail_config["input_shape"])
        if len(input_shape) != 4:
            raise ValueError(f"Qwen3-TTS tail input must be NCHW, got {input_shape}")
        input_tensor = torch.zeros(input_shape, dtype=torch.float32)
        with torch.inference_mode():
            for stage in stages[:self.part_idx]:
                input_tensor = stage(input_tensor)
            output_tensor = stages[self.part_idx](input_tensor)

        if input_tensor.ndim != 4 or output_tensor.ndim != 4:
            raise RuntimeError("Qwen3-TTS tail micro stages must use fixed 4D tensors")
        torch.onnx.export(
            stages[self.part_idx],
            (input_tensor,),
            str(self.onnx_file_name),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["qwen3tts_tail_input"],
            output_names=["qwen3tts_tail_output"],
            dynamic_axes=None,
        )

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        return {}
