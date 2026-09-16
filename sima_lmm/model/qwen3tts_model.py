"""Qwen3-TTS compiler parts and model-package contract.

The Qwen3-TTS package is one composite model.  Its backbone, code predictor,
codec decoder, decoder tail, codec prefix, and talker head are compiled as
named internal parts that collectively populate ``qwen3_model/mpk``.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.nn as nn

from afe.apis.defines import TensorTessellateParameters, gen2_target
from afe.ir.attributes import ConvAttrs
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.serializer import save_awesomenet
from afe.ir.sima_builder import SimaBuilder
from afe.ir.tensor_type import ScalarType, TensorType

from sima_lmm.model.base import (
    BaseModel,
    FileGenMode,
    FileGenPrecision,
    LayerConfiguration,
)


CODEC_PREFIX_MODEL_NAME = "qwen3tts_codec_prefix_dense_f50"
TALKER_HEAD_MODEL_NAME = "qwen3tts_talker_head_n1"
QWEN3TTS_CODEC_TAIL_FRAMES = 50


@dataclass(frozen=True)
class Qwen3TTSComponent:
    """A normal compiler component owned by the Qwen3-TTS package."""

    source_directory: str
    model_name: str
    is_codec_tail: bool = False


QWEN3TTS_COMPONENTS = (
    Qwen3TTSComponent("backbone", "backbone"),
    Qwen3TTSComponent("code_predictor", "code_predictor"),
    Qwen3TTSComponent("codec_decoder", "codec_decoder"),
    Qwen3TTSComponent(
        "codec_decoder", "codec_decoder_tail_full", is_codec_tail=True
    ),
)


@dataclass
class Qwen3TTSPartModel(BaseModel):
    """One fixed-shape Qwen3-TTS graph in the standard LLiMa lifecycle."""

    components_dir: Path = field(kw_only=True)
    part: str = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.part not in {"codec_prefix", "talker_head"}:
            raise ValueError(f"Unknown Qwen3-TTS part: {self.part}")
        if not self.components_dir.is_dir():
            raise FileNotFoundError(
                f"Qwen3-TTS components directory is missing: {self.components_dir}"
            )

    @property
    def raw_elf_name(self) -> str:
        return f"{self.model_name}_stage1_mla.elf"

    def _load_weights(
        self, relative_path: str, expected: dict[str, tuple[int, ...]]
    ) -> dict[str, np.ndarray]:
        source_path = self.components_dir / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Qwen3-TTS weight file is missing: {source_path}")
        with safe_open(source_path, framework="numpy") as source:
            result = {name: source.get_tensor(name) for name in expected}
        for name, shape in expected.items():
            value = result[name]
            if value.shape != shape or value.dtype != np.float32:
                raise RuntimeError(
                    f"Unexpected {name} in {source_path}: got {value.shape} {value.dtype}, "
                    f"expected {shape} float32"
                )
        return result

    @staticmethod
    def _conv(
        builder: SimaBuilder,
        input_node: object,
        weight: np.ndarray,
        bias: np.ndarray | None,
    ) -> object:
        if weight.ndim == 2:
            weight = weight[:, :, None]
        hwigo = np.ascontiguousarray(
            weight.transpose(2, 1, 0)[None, :, :, None, :]
        )
        input_value = get_expected_tensor_value(input_node.get_type().output)
        attrs = ConvAttrs(
            stride=(1, 1),
            dilation=(1, 1),
            padding=((0, 0), (0, 0)),
            output_padding=((0, 0), (0, 0)),
            is_transposed=False,
            weight_shape=hwigo.shape,
            reloc_name=None,
            input_spatial_shape=input_value.shape[1:-1],
            batch_size=1,
            input_type=input_value.scalar,
        )
        return builder.create_conv_node(input_node, hwigo, bias, attrs)

    @staticmethod
    def _assert_ops(net: object, expected: set[str]) -> None:
        operators = {
            type(node.ir.operation).__name__
            for node in net.nodes["MLA_0"].ir.nodes.values()
        }
        if operators != expected:
            raise RuntimeError(
                f"Unexpected Qwen3-TTS part operators: {operators}; "
                f"expected {expected}"
            )

    def _codec_prefix_net(self) -> object:
        weights = self._load_weights(
            "codec_decoder/model.safetensors",
            {
                "quantizer.rvq_first.output_proj.weight": (512, 256, 1),
                "quantizer.rvq_rest.output_proj.weight": (512, 256, 1),
                "pre_conv.conv.weight": (1024, 512, 3),
                "pre_conv.conv.bias": (1024,),
                "pre_transformer.input_proj.weight": (512, 1024),
                "pre_transformer.input_proj.bias": (512,),
            },
        )
        builder = SimaBuilder(Status.RELAY, gen2_target)
        input_shape = (1, 1, 52, 256)
        host_inputs = [
            builder.create_placeholder_node(
                name, TensorType(ScalarType.float32, input_shape)
            )
            for name in ("first_sum", "rest_sum")
        ]
        builder.begin_subnet(host_inputs)
        first, rest = [
            builder.create_placeholder_node(
                f"MLA_0/{name}", TensorType(ScalarType.float32, input_shape)
            )
            for name in ("first_sum", "rest_sum")
        ]
        first = self._conv(
            builder, first, weights["quantizer.rvq_first.output_proj.weight"], None
        )
        rest = self._conv(
            builder, rest, weights["quantizer.rvq_rest.output_proj.weight"], None
        )
        combined = builder.create_add_node(first, rest)
        convolved = self._conv(
            builder,
            combined,
            weights["pre_conv.conv.weight"],
            weights["pre_conv.conv.bias"],
        )
        output = self._conv(
            builder,
            convolved,
            weights["pre_transformer.input_proj.weight"],
            weights["pre_transformer.input_proj.bias"],
        )
        if get_expected_tensor_value(output.get_type().output).shape != (1, 1, 50, 512):
            raise RuntimeError("Unexpected Qwen3-TTS codec-prefix output shape")
        builder.finish_subnet("MLA_0")
        net = builder.finish(self.model_name)
        self._assert_ops(
            net, {"PlaceholderOp", "ConvAddActivationOp", "AddActivationOp"}
        )
        return net

    def _talker_head_net(self) -> object:
        weight = self._load_weights(
            "codec_head/codec_head.safetensors", {"weight": (3072, 1024)}
        )["weight"]
        builder = SimaBuilder(Status.RELAY, gen2_target)
        input_shape = (1, 1, 1, 1024)
        host = builder.create_placeholder_node(
            "hidden", TensorType(ScalarType.float32, input_shape)
        )
        builder.begin_subnet([host])
        hidden = builder.create_placeholder_node(
            "MLA_0/hidden", TensorType(ScalarType.float32, input_shape)
        )
        output = self._conv(builder, hidden, weight, None)
        if get_expected_tensor_value(output.get_type().output).shape != (1, 1, 1, 3072):
            raise RuntimeError("Unexpected Qwen3-TTS talker-head output shape")
        builder.finish_subnet("MLA_0")
        net = builder.finish(self.model_name)
        self._assert_ops(net, {"PlaceholderOp", "ConvAddActivationOp"})
        return net

    def gen_model_sdk_files_directly(
        self, layer_cfg: LayerConfiguration, log_level: int, quantizable: bool
    ) -> None:
        del log_level
        if not quantizable:
            raise ValueError(
                "Qwen3-TTS parts require SOURCE_TO_FP followed by FP_TO_QUANT"
            )
        if layer_cfg["precision"] != FileGenPrecision.BF16:
            raise ValueError("Qwen3-TTS parts require BF16 precision")
        net = (
            self._codec_prefix_net()
            if self.part == "codec_prefix"
            else self._talker_head_net()
        )
        save_awesomenet(net, f"{self.model_name}.fp32", str(self.sima_model_sdk_path))

    def get_mla_input_tessellate_params(
        self,
    ) -> dict[int, TensorTessellateParameters]:
        return {}

    def get_mla_output_tessellate_params(
        self,
    ) -> dict[int, TensorTessellateParameters]:
        return {}

    def _extract_raw_elf(self) -> None:
        if not self.mpk_file_name.is_file():
            raise FileNotFoundError(f"Missing generated MPK: {self.mpk_file_name}")
        with tarfile.open(self.mpk_file_name, "r:gz") as archive:
            matches = [
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).name == self.raw_elf_name
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {self.raw_elf_name} in {self.mpk_file_name}, "
                    f"found {len(matches)}"
                )
            source = archive.extractfile(matches[0])
            if source is None:
                raise RuntimeError(
                    f"Cannot read {self.raw_elf_name} from {self.mpk_file_name}"
                )
            destination = self.sima_mpk_path / self.raw_elf_name
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    def gen_files(
        self,
        gen_mode: FileGenMode,
        *,
        layer_cfg: LayerConfiguration | None = None,
        log_level: int = 0,
        resume: bool = False,
    ) -> bool:
        if (
            resume
            and gen_mode == FileGenMode.MODEL_SDK_COMPILE
            and self.mpk_file_name.is_file()
        ):
            if not (self.sima_mpk_path / self.raw_elf_name).is_file():
                self._extract_raw_elf()
            return False
        generated = super().gen_files(
            gen_mode, layer_cfg=layer_cfg, log_level=log_level, resume=resume
        )
        if generated and gen_mode == FileGenMode.MODEL_SDK_COMPILE:
            self._extract_raw_elf()
        return generated

    def gen_mpk_files(self, log_level: int):
        """Use the normal BaseModel MPK stage with Palette SDK compatibility."""
        from afe.backends.mla.afe_to_n2a_compiler import n2a_compiler_operations

        original_config = n2a_compiler_operations.CompilerConfig

        def compatible_config(*args: object, **options: object) -> object:
            if options.pop("layout_search_effort_level", 0) != 0:
                raise RuntimeError(
                    "Qwen3-TTS part compilation does not support layout search effort"
                )
            return original_config(*args, **options)

        n2a_compiler_operations.CompilerConfig = compatible_config
        try:
            return super().gen_mpk_files(log_level)
        finally:
            n2a_compiler_operations.CompilerConfig = original_config


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
        config = json.loads((self._hf_cache_path() / "config.json").read_text())
        # The shipped qwen3_components/codec_decoder checkpoint is also used
        # by the 50-frame raw decoder tail.  Its generic HF config does not
        # carry the tail-only input fields, so the composite Qwen3-TTS package
        # supplies this fixed runtime contract here.
        config.setdefault("frames", QWEN3TTS_CODEC_TAIL_FRAMES)
        config.setdefault(
            "input_shape", [1, 1024, 1, QWEN3TTS_CODEC_TAIL_FRAMES]
        )
        return config

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
