import copy
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.config.vlm_config import (
    LlmArchType, ModelFormat, SPECULATIVE_BUDGET, VisionArchType, VlmConfig, model_file_type,
)
from sima_lmm.gguf.gguf_conversion import GgufModel
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.model.base import (
    BaseModel, EvalMode, FileGenMode, FileGenPrecision, LoraGenMode, GenConfiguration
)
from sima_lmm.model.language_model import LanguageModel
from sima_lmm.model.vision_model import VisionModel
from sima_lmm.preproc.vlm_helper import Chat, VlmHelper
from sima_lmm.utils import ceil_div_row, mla_max_num_rows
from sima_utils.logging.sima_logger import sima_log_info, sima_log_warning



class _TrivialContextManager:
    """
    Context manager that does nothing.
    """
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass


class _LoadParamsContextManager:
    """
    Load model parameters on entry, unload on exit.
    """
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.load_all_params()

    def __exit__(self, exc_type, exc_value, traceback):
        self._model.unload_all_params()


@dataclass
class VisionLanguageModel(BaseModel):
    """Vision-language model implementation."""
    language_model: LanguageModel = field(init=False)

    def __post_init__(self):
        self.language_model = LanguageModel(
            self.cfg, self.language_model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, vlm_helper=self.vlm_helper,
        )

    @staticmethod
    def from_hf_cache(
        model_name: str,
        hf_cache_path: Path | str,
        onnx_path: Path | str,
        sima_path: Path | str,
        max_num_tokens: int,
        system_prompt: str | None = None,
        chat_template: str | None = None,
        override_language_group_size: int | None = None,
        override_language_group_offsets: list[int] | None = None,
        override_language_future_token_mask_size: int = 1,
        return_logits: bool = False,
        enable_filter_sharing: bool = False,
        quantize_embeddings: bool = False,
        split_mlp: bool = False,
        image_resolution: list[int] | None = None,
        target_model: "VisionLanguageModel | None" = None,
    ) -> "VisionLanguageModel":
        """Creates a VisionLanguageModel object from cached Hugging Face model.

        Args:
            model_name: Model name. This is used as a file name prefix for the generated onnx
                and model sdk files.
            hf_cache_path: Path to the cached Hugging Face model.
            onnx_path: Path to the generated ONNX files.
            sima_path: Path to the generated SiMa files.
            max_num_tokens: Maximum number of tokens, including both input and output tokens.
            system_prompt: System prompt.
            return_logits: Return logits at the last layer output instead of argmax token IDs.
            enable_filter_sharing: True if sharing filters between group and single models is
                enabled.
            quantize_embeddings: True if embedding table is quantized.
            split_mlp: True if mlp is being split into multiple parts.
            target_model: Target VisionLanguageModel when constructing a draft model.
                Copies tokenizer and embeddings (if missing) from it. None for non-draft models.
        Returns:
            A VisionLanguageModel object for file generation or evaluation.
        """
        model_format = model_file_type(Path(hf_cache_path))
        if model_format == ModelFormat.FORMAT_HF:
            hf_model = LocalHuggingFaceModel.create_from_directory(directory=hf_cache_path)
            model_config = hf_model.config
        else:
            assert model_format == ModelFormat.FORMAT_GGUF
            hf_model = GgufModel(hf_cache_path)
            model_config = hf_model.model_config

        vlm_cfg = VlmConfig.from_hf_config(
            model_format, hf_cache_path, model_config, image_resolution=image_resolution
        )
        # Set the token size and offsets for group processing.
        vlm_cfg.config_pipeline(
            system_prompt, chat_template, max_num_tokens, override_language_group_size,
            override_language_group_offsets, override_language_future_token_mask_size
        )

        vlm_cfg.pipeline_cfg.set_enable_filter_sharing(enable_filter_sharing)
        vlm_cfg.pipeline_cfg.set_split_mlp(split_mlp)
        vlm_cfg.pipeline_cfg.set_return_logits(return_logits)

        # Embeddings quantization is only supported for LLMs.
        if quantize_embeddings and vlm_cfg.is_supported_multimodal:
            raise NotImplementedError(
                f"Embeddings quantization is only supported for LLMs."
            )
        vlm_cfg.pipeline_cfg.set_quantize_embeddings(quantize_embeddings)

        if target_model is not None:
            # Some draft models use target model's tokenization scheme.
            tokenizer_files = ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")
            has_tokenizer = any(
                list(Path(hf_cache_path).rglob(f)) for f in tokenizer_files
            )
            if has_tokenizer:
                vlm_helper = VlmHelper(vlm_cfg, hf_cache_path)
            else:
                vlm_helper = target_model.vlm_helper

            # Set speculative decoding configs for the draft model
            vlm_cfg.lm_cfg.set_speculative_decoding_config(
                dict(is_draft=True, speculative_budget=SPECULATIVE_BUDGET["draft"])
            )

        else:
            vlm_helper = VlmHelper(vlm_cfg, hf_cache_path)

        return VisionLanguageModel(
            cfg=vlm_cfg,
            hf_model=hf_model,
            model_name=model_name,
            onnx_path=Path(onnx_path),
            sima_path=Path(sima_path),
            vlm_helper=vlm_helper,
        )

    def set_lora_adapter(self, lora_path: Path):
        lora_config = LocalHuggingFaceModel.load_lora_adapter(lora_path)
        self.cfg.lm_cfg.set_lora_adapter(lora_config)

    def configure_speculative_decoding(self, is_draft: bool = False):
        speculative_budget = SPECULATIVE_BUDGET["draft"] if is_draft else SPECULATIVE_BUDGET["target"]
        self.cfg.lm_cfg.set_speculative_decoding_config(
            dict(is_draft=is_draft, speculative_budget=speculative_budget)
        )

    def gen_files(
        self,
        gen_mode: FileGenMode = FileGenMode.ALL,
        *,
        gen_config: GenConfiguration,
        log_level: int = logging.NOTSET,
        num_processes: int = 1,
        resume: bool = False,
    ):
        """
        Generates files based on the provided file generation mode.

        Args:
            gen_mode: File generation mode.
            gen_config: Generation configuration of precision and lora for each layer.
                Layer IDs can be obtained using VlmConfig.get_layer_ids.
                The precision dict is a map from layer ID to precision for each layer to be
                processed. Unrecognized layer IDs will be ignored.  Layers that are not in the map
                will not be processed.
                The lora dict is a map from layer ID to lora graph mode for each layer.
            log_level: Logging level.
            resume: Generate the files only if it does not exist.
        """
        sima_log_info("Generating %s files...", gen_mode)

        precision = gen_config["precision"]

        if gen_mode == FileGenMode.DEVKIT:
            return self.gen_devkit_files(precision=precision, resume=resume)
        elif not (self.sima_devkit_path / "vlm_config.json").is_file():
            self.gen_devkit_files(precision=precision, resume=False)

        if gen_mode == FileGenMode.SOURCE_TO_ONNX:
            num_processes = 1
            gen_context = _LoadParamsContextManager(self.hf_model)
        else:
            gen_context = _TrivialContextManager()

        with gen_context:
            if self.cfg.vm_cfg is not None and self.cfg.is_supported_multimodal:
                # This model includes a vision model
                elem_size = 2
                seq_len = self.cfg.vm_cfg.seq_len
                num_mla_rows_per_head = seq_len * ceil_div_row(seq_len) * elem_size
                is_single_vision_model = num_mla_rows_per_head <= mla_max_num_rows

                vision_model = VisionModel(
                    self.cfg, self.vision_model_name, onnx_path=self.onnx_path,
                    sima_path=self.sima_path, hf_model=self.hf_model,
                    is_single_vision_model=is_single_vision_model
                )
                vision_model.gen_files(
                    gen_mode, gen_config=gen_config, log_level=log_level,
                    num_processes=num_processes, resume=resume
                )

            self.language_model.gen_files(
                gen_mode, gen_config=gen_config, log_level=log_level,
                num_processes=num_processes, resume=resume
            )
        sima_log_info("%s files generation completed.", gen_mode)

    def evaluate(self, eval_mode: EvalMode, chat: Chat) -> str | np.ndarray:
        """Evaluates the model with the input query and the image in the specified mode.
        Args:
            eval_mode: Evaluation mode.
            query: User query.
            images: Paths to the images or preprocessed images in numpy array.
        Returns:
            Generated output text or output logits.
        """
        prompt, input_idxs, images, _ = self.vlm_helper.preprocess(chat)

        sima_log_info("prompt='%s'", prompt)
        sima_log_info("input_idxs=%s", input_idxs)

        # Run the model to generate output text.
        if eval_mode == EvalMode.HF:
            output_dict = self.hf_model.execute_hf(
                messages=chat.messages, images=chat.images,
                max_new_tokens=self.cfg.pipeline_cfg.max_num_tokens, device="cpu"
            )
            num_input_tokens = output_dict["input_ids"].shape[1]
            output_tokens = output_dict["output_tokens"]
            output_tokens = np.expand_dims(
                output_tokens[num_input_tokens:self.cfg.pipeline_cfg.max_num_tokens], axis=0
            )
            output_text = output_dict["generated_text"]
        else:
            ifms = [np.array([input_idxs])]
            if images is not None:
                ifms.extend(
                    [np.expand_dims(image, axis=0).astype(np.float32) for image in images]
                )
            if self.cfg.pipeline_cfg.return_logits:
                return self.run_model(eval_mode, ifms)
            output_tokens = self.run_model(eval_mode, ifms)
            assert output_tokens.ndim == 2 and output_tokens.shape[0] == 1
            output_text = self.vlm_helper.decode(output_tokens[0].tolist())
        sima_log_info("output text=%s", output_text)
        return output_text

    def run_model(self, eval_mode: EvalMode, ifms: list[np.ndarray]) -> list[np.ndarray]:
        embeddings_tensor, embeddings_scale = self.language_model.get_embeddings_tensor()
        assert embeddings_scale is None or np.isscalar(embeddings_scale), (
            "Per-channel embeddings quantization is not supported."
        )
        if len(ifms) == 1:
            text_token_idxs = ifms[0]
            num_queries = text_token_idxs.shape[0]
            assert num_queries == 1, f"Only one query is supported. Got {num_queries}."

            input_embeds = np.array(
                [[[embeddings_tensor[x] for x in text_token_idxs[0]]]],
                dtype=embeddings_tensor.dtype
            )
        else:
            assert self.cfg.vm_cfg is not None
            assert len(ifms) == 2
            text_token_idxs, image_tensor = ifms
            num_queries = text_token_idxs.shape[0]
            assert num_queries == 1, f"Only one query is supported. Got {num_queries}."

            if not self.cfg.is_supported_multimodal:
                raise NotImplementedError(
                    f"Evaluating {self.cfg.model_type} with image is not supported."
                )

            vision_model = VisionModel(
                self.cfg, self.vision_model_name, onnx_path=self.onnx_path, sima_path=self.sima_path
            )
            vision_proj, = vision_model.run_model(eval_mode, [image_tensor])
            input_embeds = self.vlm_helper.multimodal_concat(
                text_token_idxs[0], [vision_proj], embeddings_tensor
            )
            input_embeds = np.expand_dims(input_embeds, axis=(0, 1))

        return self.language_model.run_model(eval_mode, [input_embeds])

    def get_language_embeddings_tensor(self) -> np.ndarray:
        base_name = self.hf_model.language_model_param_base_name
        embeddings_tensor, _ = self.language_model.get_embeddings_tensor(
            weight_name=f"{base_name}.embed_tokens.weight",
            embed_scale=(
                self.cfg.lm_cfg.hidden_size ** 0.5
                if self.cfg.lm_cfg.arch == LlmArchType.GEMMA
                else 1.0
            )
        )
        return embeddings_tensor

    def get_language_per_layer_embeddings_tensor(self) -> np.ndarray:
        base_name = self.hf_model.language_model_param_base_name
        per_layer_embeddings, _ = self.language_model.get_embeddings_tensor(
            weight_name=f"{base_name}.embed_tokens_per_layer.weight",
            embed_scale=(
                self.cfg.lm_cfg.hidden_size_per_layer_input ** 0.5
                if self.cfg.lm_cfg.arch == LlmArchType.GEMMA
                else 1.0
            ),
        )
        return per_layer_embeddings
