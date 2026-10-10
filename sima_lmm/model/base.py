import json
import logging
import multiprocessing
import numpy as np
import os
import shutil
from typing import ClassVar, NotRequired, TypedDict

from abc import ABC
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from ml_dtypes import bfloat16
from pathlib import Path

from transformers import GenerationConfig

from afe.apis.defines import (
    QuantizationParams, TensorTessellateParameters, bfloat16_scheme, default_quantization,
    quantization_scheme, SkipCalibration
)
from afe.backends.backends import Backend
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.model import Model as SDKModel
import afe.apis.loaded_net
import afe.ir.serializer
from sima_lmm.config.layer_id import LayerID
from sima_lmm.config.vlm_config import BaseConfig, VlmConfig, VlmArchType, LlmArchType
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.gguf.gguf_conversion import GgufModel
from sima_lmm.model.sima_analysis import get_tessellate_parameters
from sima_lmm.preproc.vlm_helper import VlmHelper
from sima_lmm.logging import (
    ScopedLogLevel, sima_log_exception, sima_log_dbg, sima_log_info
)


class FileGenMode(Enum):
    """
    File generation mode.
    """
    # Use this mode to generate floating-point Model SDK files.
    SOURCE_TO_FP = auto()

    # Use this mode to quantize Model SDK files.
    FP_TO_QUANT = auto()

    # Use this mode to generate quantized Model SDK files.
    SOURCE_TO_QUANT = auto()

    # Use this mode to generate MPK tar.gz file.  Input is quantized Model SDK files ("QUANT").
    MODEL_SDK_COMPILE = auto()

    # Use this mode to generate the script to run on the board.
    DEVKIT = auto()

    # Use this mode to generate all files.
    ALL = auto()


class FileGenPrecision(str, Enum):
    """
    Precision used when generating files.
    """
    # Both activations and weights are in BF16.
    BF16 = "BF16"

    # Activations are in BF16 and weights are in INT8.
    A_BF16_W_INT8 = "A_BF16_W_INT8"

    # Activations are in BF16 and weights are in INT4.
    A_BF16_W_INT4 = "A_BF16_W_INT4"


class LoraGenMode(str, Enum):
    """
    LoRA graph generation mode.
    """
    # LoRA adapter is in a separate branch.
    LORA_BRANCH = "LORA_BRANCH"

    # LoRA adapter is merged to the base model.
    LORA_MERGED = "LORA_MERGED"

    # LoRA adapter is not enabled.
    LORA_DISABLED = "LORA_DISABLED"


GenConfiguration = TypedDict(
    "GenConfiguration", {
        "precision": dict[LayerID, FileGenPrecision],
        "lora": NotRequired[dict[LayerID, LoraGenMode]],
    }
)


LayerConfiguration = TypedDict(
    "LayerConfiguration", {
        "precision": FileGenPrecision,
        "lora": NotRequired[LoraGenMode],
    }
)


def _quantization_params(precision: FileGenPrecision) -> QuantizationParams:
    """
    Get the quantization parameters to use for quantizing a model with
    the given precision.
    """
    match precision:
        case FileGenPrecision.BF16:
            wq = bfloat16_scheme()
        case FileGenPrecision.A_BF16_W_INT8:
            wq = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
        case FileGenPrecision.A_BF16_W_INT4:
            wq = quantization_scheme(asymmetric=False, per_channel=True, bits=4)
        case _:
            raise RuntimeError(
                f"Model SDK files generation for precision={precision} is not implemented"
            )

    return (
        default_quantization
            .with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(wq)
    )


class EvalMode(str, Enum):
    """
    Model evaluation mode.
    """
    # Evaluate using HuggingFace model and processor.
    HF = "hf"

    # Evaluate using quantized Model SDK files.
    SDK = "sdk"


@dataclass
class BaseModel(ABC):
    """
    Base implementation for visual-language model file generation.

    Attributes:
        cfg: Configuration of the model.
        model_name: Name of the model. This will be used to determine the generated files' names.
        sima_path: Path to store the SiMa-specific files.
        hf_model: LocalHuggingFaceModel or GgufModel object
            for obtaining source parameters.
        weight_prefix: The prefix of weight tensor names in the source model.
    """
    cfg: BaseConfig
    model_name: str
    sima_path: Path = field(default="sima_files", kw_only=True)
    hf_model: LocalHuggingFaceModel | GgufModel | None = field(default=None, kw_only=True)
    vlm_helper: VlmHelper | None = field(default=None, kw_only=True)
    enable_filter_sharing: ClassVar[bool] = True

    def gen_files(
        self, gen_mode: FileGenMode, *,
        layer_cfg: LayerConfiguration | None = None,
        log_level: int = logging.NOTSET, resume: bool = False
    ) -> bool:
        """
        Generates files based on the provided file generation mode.

        Args:
            gen_mode: File generation mode.
            layer_cfg: The configuration of precision and lora mode to be used for Model SDK
                graph generation and quantization.
            log_level: Logging level.
            resume: Set to generate only when the file cannot be found.

        Returns:
            True if the file is created and False if the file creation is skipped.
        """
        if resume and self.get_gen_file_name(gen_mode).is_file():
            return False

        if layer_cfg is None:
            layer_cfg = {
                "precision": FileGenPrecision.BF16,
                "lora": LoraGenMode.LORA_BRANCH
            }

        enable_verbose_error_messages()
        with ScopedLogLevel(log_level):
            match gen_mode:
                case FileGenMode.SOURCE_TO_FP:
                    self.sima_model_sdk_path.mkdir(parents=True, exist_ok=True)
                    self.generate_graph(layer_cfg, quantizable=True)
                case FileGenMode.FP_TO_QUANT:
                    self.sima_path.mkdir(parents=True, exist_ok=True)
                    self.quantize_model_sdk(layer_cfg, log_level=log_level)
                case FileGenMode.SOURCE_TO_QUANT:
                    self.sima_model_sdk_path.mkdir(parents=True, exist_ok=True)
                    self.generate_graph(layer_cfg, quantizable=False)
                case FileGenMode.MODEL_SDK_COMPILE:
                    self.sima_path.mkdir(parents=True, exist_ok=True)
                    self.gen_mpk_files(log_level=log_level)
                case _:
                    raise RuntimeError(
                        f"Files generation for gen_mode={gen_mode} is not implemented"
                    )
        return True

    def run_model(self, eval_mode: EvalMode, ifms: list[np.ndarray]) -> list[np.ndarray]:
        """
        Runs the model based on the evaluation mode.
        """
        print_inouts = os.getenv("SIMA_VLM_EVAL_PRINT_INOUTS", False)
        if print_inouts:
            for idx, ifm in enumerate(ifms):
                sima_log_dbg(f"{self.model_name} ifm{idx} {ifm.shape} {ifm}")
        match eval_mode:
            case EvalMode.SDK:
                ofms = self._run_model_sdk_model(ifms)
            case _:
                raise RuntimeError(f"Running model for eval mode={eval_mode} is not implemented")
        if print_inouts:
            for idx, ofm in enumerate(ofms):
                sima_log_dbg(f"{self.model_name} ofm{idx} {ofm.shape} {ofm}")
        return ofms

    @property
    def vision_model_name(self) -> str:
        return f"{self.model_name}_vision"

    @property
    def language_model_name(self) -> str:
        return f"{self.model_name}_language"

    @property
    def sima_model_sdk_path(self) -> Path:
        """Path to the generated quantized Model SDK files."""
        return self.sima_path / "sdk"

    @property
    def sima_mpk_path(self) -> Path:
        """Path to the generated MPK files."""
        return self.sima_path / "mpk"

    @property
    def sdk_fp_file_name(self) -> Path:
        """Path to the generated floating-point Model SDK file."""
        return self.sima_model_sdk_path / f"{self.model_name}.fp32.sima"

    @property
    def sdk_file_name(self) -> Path:
        """Path to the generated quantized Model SDK file."""
        return self.sima_model_sdk_path / f"{self.model_name}.sima"

    @property
    def mpk_file_name(self) -> Path:
        """Path to the generated quantized Model SDK file."""
        return self.sima_mpk_path / f"{self.model_name}_mpk.tar.gz"

    @property
    def sima_devkit_path(self) -> Path:
        """Path to the generated files for DEVKIT."""
        return self.sima_path / "devkit"

    def get_gen_file_name(self, gen_mode: FileGenMode) -> Path:
        match gen_mode:
            case FileGenMode.SOURCE_TO_FP:
                return self.sdk_fp_file_name
            case FileGenMode.FP_TO_QUANT | FileGenMode.SOURCE_TO_QUANT:
                return self.sdk_file_name
            case FileGenMode.MODEL_SDK_COMPILE:
                return self.mpk_file_name
            case _:
                raise RuntimeError(f"Files generation for gen_mode={gen_mode} is not implemented")

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool
    ):
        """
        Builds and saves this component graph from source weights and configuration.

        Args:
            layer_cfg: The configuration of precision and lora mode to be used for Model SDK
                graph generation and quantization.
            quantizable: Whether to create files for input to the quantizer.
                If True, the input model must have floating-point data.
                If False, the input model may have floating-point or quantized data.
        """
        raise NotImplementedError("generate_graph method is not implemented")

    def quantize_model_sdk(self, layer_cfg: LayerConfiguration, log_level: int):
        """Quantizes a floating-point Model SDK file, producing a quantized Model SDK file.

        Args:
            layer_cfg: The configuration including precision used for quantization.
            log_level: Logging level.
        """

        sima_model_sdk_path = self.sdk_fp_file_name
        net = afe.ir.serializer.load_awesomenet(
            str(sima_model_sdk_path.name), str(sima_model_sdk_path.parent)
        )

        quant_params = _quantization_params(layer_cfg["precision"])
        quant_params.calibration_method = SkipCalibration()
        optimization_configs = afe.apis.loaded_net._update_optimization_configs_with_quant_config(
            quant_params
        )
        calibrate_and_quantize_net = afe.driver.passes.calibration_quantization(
            optimization_configs, system_backend=Backend.EV
        )
        net = calibrate_and_quantize_net(net, None).run()
        afe.ir.serializer.save_awesomenet(
            net, self.model_name, str(self.sima_model_sdk_path)
        )

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get tessellate parameters to use for this model's inputs on the MLA.
        Empty overrides infer standard HWC16 layouts and automatic tile sizes.
        Override tensor indices for exceptional layouts such as strided caches.
        This function applies to components with one MLA subnet.
        """
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get tessellate parameters to use for this model's output on the MLA.
        Empty overrides infer standard HWC16 layouts and automatic tile sizes.
        Override tensor indices for exceptional layouts such as strided caches.
        This function applies to components with one MLA subnet.
        """
        return {}

    def gen_mpk_files(self, log_level: int) -> SDKModel:
        """Generates MPK files.

        Args:
            log_level: Logging level.
        """
        # Set tessellation parameters to perform tessellation and detessellation on MLA.
        model = SDKModel.load(
            self.model_name, self.sima_model_sdk_path, include_unquantized_net=False
        )
        input_params = self.get_mla_input_tessellate_params()
        output_params = self.get_mla_output_tessellate_params()
        tessellate_parameters = get_tessellate_parameters(model, input_params, output_params)

        retained_temporary_directory_name = os.getenv("SIMA_AUTO_LMM_RETAIN_DIR", None)
        if retained_temporary_directory_name is not None:
            # Append the model name to the retained directory name to prevent including extra
            # artifacts in the model's MPK tarball.
            retained_temporary_directory_name = (
                    Path(retained_temporary_directory_name) / self.model_name
            )
        # For speculative decoding, use higher effort for compilation for better results.
        is_speculative = (
            isinstance(self.cfg, VlmConfig)
            and self.cfg.lm_cfg.speculative_decoding_cfg is not None
        )
        layout_search_effort_level = 1 if is_speculative else 0
        model.compile(
            self.sima_mpk_path, compress=True, log_level=log_level, preserve=False,
            tessellate_parameters=tessellate_parameters,
            retained_temporary_directory_name=retained_temporary_directory_name,
            enable_filter_sharing=self.enable_filter_sharing,
            layout_search_effort_level=layout_search_effort_level, deployable=False
        )
        return model

    def gen_devkit_files(self, precision: dict[LayerID, FileGenPrecision], resume: bool = False):
        """Generates files for devkit."""
        assert isinstance(self.cfg, VlmConfig)
        self.sima_devkit_path.mkdir(parents=True, exist_ok=True)

        cfg_json_file_name = self.sima_devkit_path / "vlm_config.json"
        embeddings_file_name = self.sima_devkit_path / f"{self.language_model_name}_embeddings.bin"
        embeddings_scale_file_name = (
            self.sima_devkit_path / f"{self.language_model_name}_embedding_scales.bin"
        )
        write_embedding_scales = (
            self.cfg.pipeline_cfg.quantize_embeddings
            and not (resume and embeddings_scale_file_name.is_file())
        )
        write_embeddings = (
            not (resume and embeddings_file_name.is_file())
            or write_embedding_scales
        )
        write_cfg = (
            not (resume and cfg_json_file_name.is_file())
            or (write_embeddings and self.cfg.pipeline_cfg.quantize_embeddings)
        )
        if write_embeddings or write_embedding_scales:
            embeddings, embeddings_scale = self.get_language_embeddings_tensor()

            # Speculative drafts use the target embedding table and therefore have no local
            # embedding artifacts. Preserve the table dtype exactly for models that own one.
            if embeddings is None:
                assert embeddings_scale is None
            else:
                assert (
                    embeddings_scale is not None
                    or not self.cfg.pipeline_cfg.quantize_embeddings
                )
                if write_embeddings:
                    if not self.cfg.pipeline_cfg.quantize_embeddings:
                        embeddings = embeddings.astype(bfloat16)
                    embeddings.tofile(embeddings_file_name)
                if write_embedding_scales:
                    assert embeddings_scale is not None
                    embeddings_scale.astype(bfloat16).tofile(embeddings_scale_file_name)
            del embeddings
            del embeddings_scale

        # gpt_oss attention sinks, stacked (num_layers, num_attention_heads) for the devkit.
        if self.cfg.lm_cfg.arch == LlmArchType.GPT_OSS:
            sinks_file_name = self.sima_devkit_path / f"{self.language_model_name}_sinks.bin"
            if not (resume and sinks_file_name.is_file()):
                base_name = self.hf_model.language_model_param_base_name
                sinks = np.stack([
                    np.asarray(
                        self.get_hf_param(f"{base_name}.layers.{i}.self_attn.sinks"),
                        dtype=np.float32,
                    )
                    for i in range(self.cfg.lm_cfg.num_hidden_layers)
                ])
                sinks.astype(bfloat16).tofile(sinks_file_name)

        if write_cfg:
            cfg_dict = asdict(self.cfg)
            cfg_dict["language_model_name"] = self.language_model_name
            if self.cfg.vm_cfg is not None:
                cfg_dict["vision_model_name"] = self.cfg.get_vision_model_names(
                    self.vision_model_name
                )
            if isinstance(self.hf_model, GgufModel):
                cfg_dict["gguf_file_name"] = self.hf_model.file_path.name
            with open(self.sima_devkit_path / "vlm_config.json", "w") as f:
                json.dump(cfg_dict, f, indent=4)

        per_layer_embeddings_file_name = (
            self.sima_devkit_path / f"{self.language_model_name}_per_layer_embeddings.bin"
        )
        per_layer_embeddings_scale_file_name = (
            self.sima_devkit_path
            / f"{self.language_model_name}_per_layer_embedding_scales.bin"
        )
        write_per_layer_embedding_scales = (
            self.cfg.model_type == VlmArchType.VLM_GEMMA4
            and self.cfg.pipeline_cfg.quantize_embeddings
            and not (resume and per_layer_embeddings_scale_file_name.is_file())
        )
        write_per_layer_embeddings = (
            self.cfg.model_type == VlmArchType.VLM_GEMMA4
            and (
                not (resume and per_layer_embeddings_file_name.is_file())
                or write_per_layer_embedding_scales
            )
        )
        if write_per_layer_embeddings or write_per_layer_embedding_scales:
            per_layer_embeddings, per_layer_embeddings_scale = (
                self.get_language_per_layer_embeddings_tensor()
            )
            assert per_layer_embeddings is not None
            if write_per_layer_embeddings:
                if not self.cfg.pipeline_cfg.quantize_embeddings:
                    per_layer_embeddings = per_layer_embeddings.astype(bfloat16)
                per_layer_embeddings.tofile(per_layer_embeddings_file_name)
            if write_per_layer_embedding_scales:
                assert per_layer_embeddings_scale is not None
                per_layer_embeddings_scale.astype(bfloat16).tofile(
                    per_layer_embeddings_scale_file_name
                )
            del per_layer_embeddings
            del per_layer_embeddings_scale

        if isinstance(self.hf_model, LocalHuggingFaceModel):
            # Copy the HF files.
            src_file_names = list(self.hf_model.hf_cache.rglob("*.json"))
            src_file_names.extend(self.hf_model.hf_cache.rglob("*.txt"))
            src_file_names.extend(self.hf_model.hf_cache.rglob("*.jinja"))
            src_file_names.extend(self.hf_model.hf_cache.rglob("*.model"))
            for src_file_name in src_file_names:
                # Ignore the files not from HF and safetensor files.
                if (
                    src_file_name.is_relative_to(self.sima_path)
                    or "safetensors" in src_file_name.name
                ):
                    continue

                dst_file_name = self.sima_devkit_path / src_file_name.name
                if not (resume and dst_file_name.is_file()):
                    shutil.copy(src_file_name, dst_file_name)

            # Write the precision json file.
            precision_file_name = self.sima_devkit_path / "precision.json"
            if not (resume and precision_file_name.is_file()):
                precision_list = list()
                for layer_id, layer_p in precision.items():
                    precision_list.append(
                        {
                            "part": layer_id.part,
                            "idx": layer_id.part_idx,
                            "expert_idx": layer_id.expert_idx,
                            "precision": layer_p,
                        }
                    )
                with open(precision_file_name, "w") as f:
                    json.dump(precision_list, f, indent=4)

            # Save the EAGLE3 draft model's d2t/t2d mapping tensors if present
            for tensor_name in ("d2t", "t2d"):
                out_path = self.sima_devkit_path / f"{tensor_name}.npy"
                if resume and out_path.is_file():
                    continue
                if tensor_name in self.hf_model.weight_map:
                    tensor = self.hf_model.load_np_param(tensor_name)
                    np.save(out_path, tensor)
        else:
            assert isinstance(self.hf_model, GgufModel)
            # Copy the GGUF file to construct the VlmHelper.
            dst_gguf_file = self.sima_devkit_path / self.hf_model.file_path.name
            if not (resume and dst_gguf_file.is_file()):
                shutil.copy(self.hf_model.file_path, dst_gguf_file)

            # Extract processor_config.json and preprocessor_config.json.
            if self.cfg.is_multimodal:
                raise NotImplementedError(
                    "Extract config files from GGUF to create AutoProcessor is not supported"
                )

            # Extract generation_config.json.
            gguf_generation_config_file = self.hf_model.file_path.parent / "params"
            if gguf_generation_config_file.is_file():
                try:
                    with open(gguf_generation_config_file, "r") as f:
                        generation_cfg_dict = json.load(f)

                    if stop_tokens := generation_cfg_dict.get("stop"):
                        if isinstance(stop_tokens, int):
                            stop_tokens = [stop_tokens]
                        generation_cfg_dict["eos_token_id"] = self.vlm_helper.hf_tokenizer.tokenize(
                            "".join(stop_tokens).encode("utf-8"), add_bos=False, special=True
                        )
                        del generation_cfg_dict["stop"]
                    if (repetition_penalty := generation_cfg_dict.get("repeat_penalty")) is None:
                        generation_cfg_dict["repetition_penalty"] = repetition_penalty
                        del generation_cfg_dict["repeat_penalty"]
                    if "num_predict" in generation_cfg_dict:
                        del generation_cfg_dict["num_predict"]

                    generation_cfg_dict["do_sample"] = (
                        generation_cfg_dict.get("temperature", 1.0) != 1.0
                        or generation_cfg_dict.get("top_p", 1.0) != 1.0
                        or generation_cfg_dict.get("typical_p", 1.0) != 1.0
                        or "min_p" in generation_cfg_dict
                        or generation_cfg_dict.get("top_k", 50) != 50
                    )

                    generation_cfg = GenerationConfig.from_dict(generation_cfg_dict)
                    generation_cfg.save_pretrained(self.sima_devkit_path)
                except json.decoder.JSONDecodeError:
                    sima_log_info("Failed to create generation_config.json file GGUF's params file")

    def check_hf_param(self, name: str) -> bool:
        """Checks if a parameter tensor exists in the LocalHuggingFaceModel object
        or GgufModel object.

        Args:
            name: Full name of the parameter in HuggingFace convention.

        Returns:
            True if the parameter tensor exists.
        """
        return self.hf_model.param_exists(name)

    def get_hf_param(self, name: str) -> np.ndarray | tuple:
        """Gets the parameter tensor from the LocalHuggingFaceModel object
        or GgufModel object.

        Args:
            name: Full name of the parameter in HuggingFace convention.

        Returns:
            The parameter tensor in numpy array. For GGUF model or llm-compressor
            quantized weights return scales and quantized weight, followed by an
            explicit group size when the source uses grouped quantization.

        """
        assert isinstance(self.hf_model, (LocalHuggingFaceModel, GgufModel)), \
            f"Unsupported model type: {type(self.hf_model)}"

        return self.hf_model.load_np_param(name)

    def _run_model_sdk_model(self, ifms: list[np.ndarray]) -> list[np.ndarray]:
        model = SDKModel.load(
            self.model_name, self.sima_model_sdk_path, include_unquantized_net=False
        )
        ifm_dict = {}
        for ifm_name, ifm in zip(model._net.input_node_names, ifms):
            ifm_dict[ifm_name] = ifm

        # Keep the code below in comment. The code is used to evalutate the model using mla
        # instruction simulator instead of the ml_kernels. To enable the code, need to avoid the
        # AwesomeNet deepcopy in the generate_mpk_json_data() of afe/backends/mpk/interface.py.
        # There are other bugs in afe/backends/mla/afe_to_n2a_compiler/n2a_backend_runner.py that
        # need to be fixed before it can be run properly.
        # if "post" in self.model_name:
        #     from afe.backends.mla.afe_to_n2a_compiler.n2a_backend_runner import (
        #         create_n2a_backend_runner, RunMode
        #     )

        #     model.execute(ifm_dict, keep_layer_outputs="all", output_file_path="tmp/gold.pkl")
        #     model = self.gen_mpk_files(logging.DEBUG)

        #     run_mode = RunMode.SIMULATOR
        #     backend_runner = create_n2a_backend_runner(
        #         model._net, out_dir="tmp", run_mode=run_mode, platform_type=model._net.target
        #     )
        #     output = model._net.run(ifm_dict, node_callable=backend_runner.execute_node)
        #     print("output", output, flush=True)
        #     exit()
        #TODO Change to jax again after arm bug is fixed
        return model.execute(ifm_dict, use_jax=False)

    def gen_files_from_model_list(
        self,
        model_list: list[tuple["BaseModel", LayerConfiguration]],
        gen_mode: FileGenMode,
        num_processes: int,
        log_level: int,
        resume: bool
    ):
        stage = {
            FileGenMode.SOURCE_TO_FP: "graph",
            FileGenMode.FP_TO_QUANT: "quantize",
            FileGenMode.SOURCE_TO_QUANT: "quantized graph",
            FileGenMode.MODEL_SDK_COMPILE: "compile",
        }.get(gen_mode, "generate")
        total = len(model_list)
        if total:
            print(f"[{stage}] Processing {total} components", flush=True)
        if num_processes != 1 and len(model_list) > 1:
            os.environ["SIMA_MLA_SIM_PARALLEL"] = "1"
            with ProcessPoolExecutor(
                max_workers=num_processes, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                results = {}
                generated_file_name = None
                try:
                    for model, layer_cfg in model_list:
                        generated_file_name = model.get_gen_file_name(gen_mode)
                        future = executor.submit(
                            model.gen_files, gen_mode, layer_cfg=layer_cfg,
                            log_level=logging.NOTSET, resume=resume
                        )
                        results[future] = (model.model_name, generated_file_name)
                    # Include every spawned worker in the manager's exit watch set.
                    executor._executor_manager_thread_wakeup.wakeup()
                    with ScopedLogLevel(log_level):
                        for completed, future in enumerate(as_completed(results), start=1):
                            model_name, generated_file_name = results[future]
                            created = future.result()
                            if created:
                                sima_log_info("Created %s.", generated_file_name)
                            else:
                                sima_log_info("Skipped %s.", generated_file_name)
                            action = "Completed" if created else "Skipped"
                            print(f"[{stage} {completed}/{total}] {action} {model_name}", flush=True)
                except BaseException as exc:
                    # Python 3.11/3.12 have no public executor worker-termination API.
                    # Stop active work before the context manager waits for shutdown.
                    processes = list((executor._processes or {}).values())
                    for process in processes:
                        if process.is_alive():
                            process.kill()
                    executor.shutdown(wait=True, cancel_futures=True)
                    for process in processes:
                        process.close()
                    if isinstance(exc, KeyboardInterrupt):
                        msg = "Ctrl-C received. Stop generating files."
                    elif isinstance(exc, BrokenProcessPool):
                        msg = f"Process worker died while creating {generated_file_name}."
                    else:
                        msg = f"Error occurred while creating {generated_file_name}."
                    sima_log_exception(msg)
                    raise
        else:
            with ScopedLogLevel(log_level):
                for completed, (model, layer_cfg) in enumerate(model_list, start=1):
                    try:
                        created = model.gen_files(
                            gen_mode, layer_cfg=layer_cfg, log_level=log_level, resume=resume
                        )
                        if created:
                            sima_log_info("Created %s.", model.get_gen_file_name(gen_mode))
                        else:
                            sima_log_info("Skipped %s.", model.get_gen_file_name(gen_mode))
                        action = "Completed" if created else "Skipped"
                        print(f"[{stage} {completed}/{total}] {action} {model.model_name}", flush=True)
                    except Exception:
                        msg = f"Error occured when creating {model.get_gen_file_name(gen_mode)}."
                        sima_log_exception(msg)
                        raise
