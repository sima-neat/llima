import json
import logging
import numpy as np
import shutil
import time

from dataclasses import asdict, dataclass
from ml_dtypes import bfloat16
from pathlib import Path

from sima_utils.logging.sima_logger import ScopedLogLevel, sima_log_info
from transformers.audio_utils import mel_filter_bank

from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel, find_file
from sima_lmm.model.base import (
    BaseModel, EvalMode, FileGenMode, FileGenPrecision, LayerConfiguration
)
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_init_model import WhisperDecoderInitModel
from sima_lmm.model.whisper_decoder_language_detect_model import WhisperDecoderLanguageDetectModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel
from sima_lmm.model.whisper_encoder_model import WhisperEncoderModel
from sima_lmm.preproc.whisper_preproc import load_and_preprocess_numpy
from sima_lmm.tokenizer.whisper_tokenizer import get_tokenizer
from sima_lmm.config.whisper_config import WhisperConfig


@dataclass
class WhisperModel(BaseModel):
    """Whisper model implementation."""
    use_future_token_mask: bool

    @staticmethod
    def from_hf_cache(
        model_name: str,
        hf_cache_path: Path | str,
        onnx_path: Path | str,
        sima_path: Path | str,
        use_future_token_mask: bool,
        enable_filter_sharing: bool = False,
        enable_log_probe: bool = False,
    ) -> "WhisperModel":
        """Creates a WhisperModel object from cached Hugging Face model.

        Args:
            model_name: Model name. This is used as a file name prefix for the generated onnx
                and model sdk files.
            hf_cache_path: Path to the cached Hugging Face model.
            onnx_path: Path to the generated ONNX files.
            sima_path: Path to the generated SiMa files.
        Returns:
            A WhisperModel object for file generation or evaluation.
        """
        hf_model = LocalHuggingFaceModel.create_from_directory(directory=hf_cache_path)
        whisper_cfg = WhisperConfig.from_hf_config(hf_cache_path, hf_model.config)
        whisper_cfg.log_probe_enabled = enable_log_probe

        return WhisperModel(
            cfg=whisper_cfg,
            hf_model=hf_model,
            model_name=model_name,
            onnx_path=Path(onnx_path),
            sima_path=Path(sima_path),
            use_future_token_mask=use_future_token_mask,
            use_filter_sharing=enable_filter_sharing,
        )

    def gen_files(
        self,
        gen_mode: FileGenMode,
        *,
        precision: dict[str, LayerConfiguration] | None = None,
        log_level: int = logging.NOTSET,
        num_processes: int = 1,
        part: str | None = None,
        part_idx: int | None = None,
        resume: bool = False,
    ):
        """
        Generates files based on the provided file generation mode.

        Args:
            gen_mode: File generation mode.
            precision: Layer configuration to be used for Model SDK quantization mode.
            log_level: Logging level.
            part: Name of the part to be generated.
            part_idx: Specific index of the part to be generated. For pre/post model, the index is
                the layer index; for cache model, the index is the token index.
            resume: Generate the files if missing.
        """
        if gen_mode == FileGenMode.ALL:
            gen_modes = [
                FileGenMode.DEVKIT, FileGenMode.SOURCE_TO_ONNX, FileGenMode.ONNX_TO_QUANT,
                FileGenMode.MODEL_SDK_COMPILE
            ]
            for gen_mode in gen_modes:
                self.gen_files(
                    gen_mode, precision=precision, log_level=log_level, num_processes=num_processes,
                    part=part, part_idx=part_idx, resume=resume
                )
            return

        sima_log_info("Generating %s files...", gen_mode)
        if gen_mode == FileGenMode.DEVKIT:
            self.gen_devkit_files(resume=resume)
            return

        encoder = False
        language_detect = False
        init_layer_idx_list = list()
        single_pre_layer_idx_list = list()
        single_post_layer_idx_list = list()
        single_cache_token_idx_list = list()
        if part_idx is None:
            if part == "all":
                part = None
            if part in ("encoder", None):
                encoder = True
            if part in ("language_detect", None):
                language_detect = True
            if part in ("init", None):
                init_layer_idx_list = list(range(self.cfg.decoder_layers))
            if part in ("single_pre", None):
                single_pre_layer_idx_list = list(range(self.cfg.decoder_layers))
            if part in ("single_post", None):
                single_post_layer_idx_list = list(range(self.cfg.decoder_layers))
            if part in ("single_cache", None):
                if self.use_future_token_mask:
                    single_cache_token_idx_list = [self.cfg.max_target_positions - 1]
                else:
                    single_cache_token_idx_list = list(
                        range(WhisperDecoderInitModel.num_tokens, self.cfg.max_target_positions)
                    )
        else:
            match part:
                case "init":
                    assert 0 <= part_idx < self.cfg.decoder_layers
                    init_layer_idx_list.append(part_idx)
                case "single_pre":
                    assert 0 <= part_idx < self.cfg.decoder_layers
                    single_pre_layer_idx_list.append(part_idx)
                case "single_cache":
                    assert (
                        WhisperDecoderInitModel.num_tokens
                        <= part_idx
                        < self.cfg.max_target_positions
                    )
                    if self.use_future_token_mask:
                        assert part_idx == self.cfg.max_target_positions - 1
                    single_cache_token_idx_list.append(part_idx)
                case "single_post":
                    assert 0 <= part_idx < self.cfg.decoder_layers
                    single_post_layer_idx_list.append(part_idx)
                case _:
                    raise RuntimeError(
                        f"Generating files for part={part} and part_idx={part_idx} is not supported"
                    )

        # Determine the precision for each part.
        if precision is None:
            precision = dict()
        if not isinstance(precision, dict):
            raise TypeError(
                "precision must be a dict of layer configurations like "
                "{'all': {'precision': FileGenPrecision.A_BF16_W_INT8}}"
            )

        default_precision = {"precision": FileGenPrecision.A_BF16_W_INT8}
        all_precision = precision.get("all", default_precision)
        encoder_precision = precision.get("encoder", all_precision)
        decoder_precision = precision.get("decoder", all_precision)
        language_detect_precision = precision.get("language_detect", decoder_precision)
        init_precision = precision.get("init", decoder_precision)
        single_precision = precision.get("single", decoder_precision)
        single_pre_precision = precision.get("single_pre", single_precision)
        single_post_precision = precision.get("single_post", single_precision)
        single_cache_precision = precision.get("single_cache", single_precision)

        if gen_mode == FileGenMode.ONNX_TO_QUANT:
            with ScopedLogLevel(log_level):
                sima_log_info("Quantization precision:")
                sima_log_info("  Encoder      = %s", encoder_precision)
                sima_log_info("  Language det = %s", language_detect_precision)
                sima_log_info("  Init         = %s", init_precision)
                sima_log_info("  Single pre   = %s", single_pre_precision)
                sima_log_info("  Single cache = %s", single_cache_precision)
                sima_log_info("  Single post  = %s", single_post_precision)

        model_list = list()
        if encoder:
            model_list.append((self._get_part_model("encoder"), encoder_precision))
        if language_detect:
            model_list.append(
                (self._get_part_model("language_detect"), language_detect_precision)
            )

        for layer_idx in init_layer_idx_list:
            model_list.append((self._get_part_model("init", layer_idx=layer_idx), init_precision))

        for layer_idx in single_pre_layer_idx_list:
            model_list.append(
                (self._get_part_model("pre", layer_idx=layer_idx), single_pre_precision)
            )

        for layer_idx in single_post_layer_idx_list:
            model_list.append(
                (self._get_part_model("post", layer_idx=layer_idx), single_post_precision)
            )

        for token_idx in single_cache_token_idx_list:
            model_list.append(
                (self._get_part_model("cache", token_idx=token_idx), single_cache_precision)
            )

        if gen_mode == FileGenMode.SOURCE_TO_ONNX:
            num_processes = 1
            self.hf_model.load_all_params()

        self.gen_files_from_model_list(model_list, gen_mode, num_processes, log_level, resume)

        if gen_mode == FileGenMode.SOURCE_TO_ONNX:
            self.hf_model.unload_all_params()
        sima_log_info("%s files generation completed.", gen_mode)

    def evaluate(
        self, eval_mode: EvalMode, audio: Path | str | np.ndarray, language: str | None = None
    ) -> str:
        """Evaluates the model with the input audio in the specified mode.

        Args:
            eval_mode: Evaluation mode.
            audio: Path to the audio or preprocessed audio in numpy array.
        Returns:
            Generated output text.
        """
        hf_preprocessor_config_json_file = find_file(
            directory=self.hf_model.hf_cache, filename="preprocessor_config.json"
        )
        audio_tensor = load_and_preprocess_numpy(
            audio, hf_preprocessor_config_json_file=hf_preprocessor_config_json_file
        )
        detect_language = self._is_auto_language(language)
        tokenizer_language = "en" if detect_language else language
        self.tokenizer = self._get_tokenizer(language=tokenizer_language, task="transcribe")

        if eval_mode == EvalMode.HF:
            raise NotImplementedError("Evaluating Whisper using HuggingFace is not supported yet.")
        else:
            input_idxs = np.array(
                [self.tokenizer.sot_sequence_including_notimestamps], dtype=np.int32
            )
            ifms = [input_idxs, audio_tensor]
            output_tokens = self.run_model(eval_mode, ifms, detect_language=detect_language)
            assert output_tokens.ndim == 2 and output_tokens.shape[0] == 1
            output_text = self.tokenizer.decode_without_special_tokens(output_tokens[0].tolist())
        sima_log_info("output text=%s", output_text)
        return output_text

    def run_model(
        self, eval_mode: EvalMode, ifms: list[np.ndarray], detect_language: bool = False
    ) -> list[np.ndarray]:
        assert len(ifms) == 2
        input_token_ids = np.array(ifms[0], copy=True)
        assert input_token_ids.shape == (1, WhisperDecoderInitModel.num_tokens)
        assert ifms[1].shape == (1, 1, self.cfg.max_source_positions * 2, self.cfg.num_mel_bins)

        # Initialize caches.
        encoder_cache_key = [
            np.zeros(
                (
                    1,
                    self.cfg.decoder_attention_heads,
                    self.cfg.max_source_positions,
                    self.cfg.decoder_head_dim,
                ),
                dtype=np.float32
            )
            for _ in range(self.cfg.decoder_layers)
        ]
        encoder_cache_value = [
            np.zeros(
                (
                    1,
                    self.cfg.decoder_attention_heads,
                    self.cfg.max_source_positions,
                    self.cfg.decoder_head_dim,
                ),
                dtype=np.float32
            )
            for _ in range(self.cfg.decoder_layers)
        ]
        decoder_cache_key = [
            np.zeros((1, 1, self.cfg.max_target_positions, self.cfg.d_model), dtype=np.float32)
            for _ in range(self.cfg.decoder_layers)
        ]
        decoder_cache_value = [
            np.zeros((1, 1, self.cfg.max_target_positions, self.cfg.d_model), dtype=np.float32)
            for _ in range(self.cfg.decoder_layers)
        ]

        # Encoder.
        encoder_ifms = [ifms[1]]
        encoder_model = self._get_part_model("encoder")
        encoder_ofms = encoder_model.run_model(eval_mode, encoder_ifms)

        if detect_language:
            language_detect_model = self._get_part_model("language_detect")
            language_detect_ofms = language_detect_model.run_model(eval_mode, [encoder_ofms[0]])
            detected_language_index = int(language_detect_ofms[0].item())
            detected_language_token = self._language_token_from_index(detected_language_index)
            input_token_ids[0, 1] = detected_language_token
            sima_log_info(
                "Detected language: %s token=%d index=%d",
                self._language_code_from_token(detected_language_token),
                detected_language_token,
                detected_language_index
            )

        # Decoder init.
        perf_cnt_begin = time.perf_counter_ns()
        token_embeddings_tensor = self.get_token_embeddings_tensor()
        decoder_input_embeds = np.array(
            [[[token_embeddings_tensor[x] for x in input_token_ids[0]]]],
            dtype=token_embeddings_tensor.dtype
        )
        num_tokens = WhisperDecoderInitModel.num_tokens
        decoder_init_ofms = None
        for layer_idx in range(self.cfg.decoder_layers):
            if layer_idx == 0:
                decoder_init_ifms = [
                    decoder_input_embeds,
                    encoder_ofms[0]
                ]
            else:
                decoder_init_ifms = [decoder_init_ofms[0], encoder_ofms[0]]
            decoder_init_model = self._get_part_model("init", layer_idx=layer_idx)
            decoder_init_ofms = decoder_init_model.run_model(eval_mode, decoder_init_ifms)

            # Update cache.
            cache_output_idx = (
                2
                if self.cfg.log_probe_enabled and layer_idx == self.cfg.decoder_layers - 1
                else 1
            )
            decoder_cache_key[layer_idx][..., :num_tokens, :] = (
                decoder_init_ofms[cache_output_idx]
            )
            decoder_cache_value[layer_idx][..., :num_tokens, :] = (
                decoder_init_ofms[cache_output_idx + 1]
            )
            encoder_cache_key[layer_idx] = decoder_init_ofms[cache_output_idx + 2]
            encoder_cache_value[layer_idx] = decoder_init_ofms[cache_output_idx + 3]
        new_token = decoder_init_ofms[0].item()
        new_tokens = [new_token]
        perf_cnt_delta = (time.perf_counter_ns() - perf_cnt_begin) * 1e-9
        sima_log_info("Got token: %d in %fs", new_token, perf_cnt_delta)

        # Decoder generation.
        generation_position_embeddings = np.expand_dims(
            self.get_position_embeddings_tensor(), axis=(0, 1)
        )
        post_ofms = None
        for token_idx in range(WhisperDecoderInitModel.num_tokens, self.cfg.max_target_positions):
            next_token_idx = token_idx + 1
            sima_log_info("Processing token no. %d-%d", token_idx, next_token_idx)
            perf_cnt_begin = time.perf_counter_ns()

            for layer_idx in range(self.cfg.decoder_layers):
                if layer_idx == 0:
                    pre_ifms = [
                        np.expand_dims(token_embeddings_tensor[new_token], axis=(0, 1, 2)),
                        generation_position_embeddings[..., token_idx:next_token_idx, :]
                    ]
                else:
                    pre_ifms = [post_ofms[0]]
                pre_model = self._get_part_model("pre", layer_idx=layer_idx)
                pre_ofms = pre_model.run_model(eval_mode, pre_ifms)
                residual_ifm = (
                    pre_ofms[WhisperDecoderPreModel.positioned_residual_output_idx]
                    if layer_idx == 0
                    else pre_ifms[0]
                )

                # Update cache.
                decoder_cache_key[layer_idx][..., token_idx:next_token_idx, :] = pre_ofms[1]
                decoder_cache_value[layer_idx][..., token_idx:next_token_idx, :] = pre_ofms[2]

                if self.use_future_token_mask:
                    aligned_token_idx = self.cfg.max_target_positions - 1
                else:
                    aligned_token_idx = token_idx
                cache_ifms = [
                    pre_ofms[0],
                    decoder_cache_key[layer_idx][..., :aligned_token_idx + 1, :],
                    decoder_cache_value[layer_idx][..., :aligned_token_idx + 1, :],
                ]
                if self.use_future_token_mask:
                    mask = np.full(
                        (1, 1, 1, self.cfg.max_target_positions),
                        np.finfo(np.float32).min,
                        dtype=np.float32
                    )
                    mask[..., :next_token_idx] = 0
                    cache_ifms.append(mask)
                cache_model = self._get_part_model("cache", token_idx=aligned_token_idx)
                cache_ofms = cache_model.run_model(eval_mode, cache_ifms)

                post_ifms = [
                    residual_ifm,
                    cache_ofms[0],
                    encoder_cache_key[layer_idx],
                    encoder_cache_value[layer_idx],
                ]
                post_model = self._get_part_model("post", layer_idx=layer_idx)
                post_ofms = post_model.run_model(eval_mode, post_ifms)

            new_token = post_ofms[0].item()
            new_tokens.append(new_token)
            perf_cnt_delta = (time.perf_counter_ns() - perf_cnt_begin) * 1e-9
            sima_log_info("Got token: %d in %fs", new_token, perf_cnt_delta)

            if new_token == self.tokenizer.eot:
                break
        # Return the generated tokens.
        return np.array([new_tokens])

    def get_token_embeddings_tensor(self) -> np.ndarray:
        assert self.hf_model, "HF cache needs to be provided to obtain the embeddings tensor."
        embeddings_name = "model.decoder.embed_tokens.weight"
        embeddings_tensor = self.hf_model.load_np_param(embeddings_name)
        return embeddings_tensor.astype(bfloat16).astype(np.float32)

    def get_position_embeddings_tensor(self) -> np.ndarray:
        assert self.hf_model, "HF cache needs to be provided to obtain the embeddings tensor."
        embeddings_name = "model.decoder.embed_positions.weight"
        embeddings_tensor = self.hf_model.load_np_param(embeddings_name)
        return embeddings_tensor.astype(bfloat16).astype(np.float32)

    @staticmethod
    def _is_auto_language(language: str | None) -> bool:
        return language is None or language == "" or language == "auto"

    def _get_tokenizer(self, language: str | None = None, task: str | None = None):
        hf_tokenizer_json_file = find_file(
            directory=self.hf_model.hf_cache, filename="tokenizer.json"
        )
        return get_tokenizer(
            multilingual=True, num_languages=self.cfg.num_languages, language=language, task=task,
            hf_tokenizer_json_file=hf_tokenizer_json_file
        )

    def _language_token_from_index(self, language_idx: int) -> int:
        language_tokens = self.tokenizer.all_language_tokens
        if language_idx < 0 or language_idx >= len(language_tokens):
            raise RuntimeError(f"Invalid detected language index: {language_idx}")
        return language_tokens[language_idx]

    def _language_code_from_token(self, token_id: int) -> str:
        language_tokens = self.tokenizer.all_language_tokens
        language_codes = self.tokenizer.all_language_codes
        return language_codes[language_tokens.index(token_id)]

    def gen_devkit_files(self, resume: bool = False):
        """Generates files for devkit."""
        assert isinstance(self.cfg, WhisperConfig)
        self.sima_devkit_path.mkdir(parents=True, exist_ok=True)

        # Write the config json file.
        cfg_json_file_name = self.sima_devkit_path / "whisper_config.json"
        if not (resume and cfg_json_file_name.is_file()):
            cfg_dict = asdict(self.cfg)
            cfg_dict["model_name"] = self.model_name
            cfg_dict["decoder_use_future_token_mask"] = self.use_future_token_mask
            cfg_dict["log_probe_enabled"] = self.cfg.log_probe_enabled
            tokenizer = self._get_tokenizer()
            cfg_dict["language_token_ids"] = list(tokenizer.all_language_tokens)
            cfg_dict["language_codes"] = list(tokenizer.all_language_codes)
            with open(self.sima_devkit_path / "whisper_config.json", "w") as f:
                json.dump(cfg_dict, f, indent=4)

        # Obtain the embeddings tensor from the hf model.
        token_embeddings_file_name = (
            self.sima_devkit_path / f"{self.model_name}_token_embeddings.npy"
        )
        if not (resume and token_embeddings_file_name.is_file()):
            np.save(token_embeddings_file_name, self.get_token_embeddings_tensor().astype(bfloat16))

        position_embeddings_file_name = (
            self.sima_devkit_path / f"{self.model_name}_position_embeddings.npy"
        )
        if not (resume and position_embeddings_file_name.is_file()):
            np.save(
                position_embeddings_file_name,
                self.get_position_embeddings_tensor().astype(bfloat16)
            )

        # Copy the tokenizer model.
        tokenizer_dst_file_name = self.sima_devkit_path / "tokenizer.json"
        if not (resume and tokenizer_dst_file_name.is_file()):
            tokenizer_src_file_name = find_file(
                directory=self.hf_model.hf_cache, filename="tokenizer.json"
            )
            assert isinstance(tokenizer_src_file_name, Path) and tokenizer_src_file_name.is_file()
            shutil.copy(tokenizer_src_file_name, tokenizer_dst_file_name)

        # Copy the preprocessor config and embed the mel filters needed by the C++ runtime.
        preprocessor_dst_file_name = self.sima_devkit_path / "preprocessor_config.json"
        if not (resume and preprocessor_dst_file_name.is_file()):
            preprocessor_src_file_name = find_file(
                directory=self.hf_model.hf_cache, filename="preprocessor_config.json"
            )
            assert (
                isinstance(preprocessor_src_file_name, Path)
                and preprocessor_src_file_name.is_file()
            )
            with open(preprocessor_src_file_name, "r") as f:
                preprocessor_cfg = json.load(f)

            if "mel_filters" not in preprocessor_cfg:
                feature_size = preprocessor_cfg["feature_size"]
                n_fft = preprocessor_cfg.get("n_fft", 400)
                sampling_rate = preprocessor_cfg.get("sampling_rate", 16000)
                preprocessor_cfg["mel_filters"] = mel_filter_bank(
                    num_frequency_bins=1 + n_fft // 2,
                    num_mel_filters=feature_size,
                    min_frequency=0.0,
                    max_frequency=sampling_rate / 2,
                    sampling_rate=sampling_rate,
                    norm="slaney",
                    mel_scale="slaney",
                ).T.tolist()

            with open(preprocessor_dst_file_name, "w") as f:
                json.dump(preprocessor_cfg, f, indent=4)

    def _get_part_model(
        self, part: str, layer_idx: int | None = None, token_idx: int | None = None
    ) -> BaseModel:
        match part:
            case "encoder":
                model_name = f"{self.model_name}_encoder"
                return WhisperEncoderModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model
                )
            case "language_detect":
                model_name = f"{self.model_name}_decoder_language_detect"
                return WhisperDecoderLanguageDetectModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, use_filter_sharing=self.use_filter_sharing
                )
            case "init":
                model_name = f"{self.model_name}_decoder_init_layer{layer_idx}"
                return WhisperDecoderInitModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, layer_idx=layer_idx,
                    enable_log_probe=(
                        self.cfg.log_probe_enabled and layer_idx == self.cfg.decoder_layers - 1
                    ),
                    use_filter_sharing=self.use_filter_sharing
                )
            case "pre":
                model_name = f"{self.model_name}_decoder_n1_pre_layer{layer_idx}"
                return WhisperDecoderPreModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=1, layer_idx=layer_idx,
                    use_filter_sharing=self.use_filter_sharing
                )
            case "post":
                model_name = f"{self.model_name}_decoder_n1_post_layer{layer_idx}"
                return WhisperDecoderPostModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=1, layer_idx=layer_idx,
                    skip_encoder_kv_proj=True, output_encoder_kv_cache=False,
                    enable_log_probe=(
                        self.cfg.log_probe_enabled and layer_idx == self.cfg.decoder_layers - 1
                    ),
                    use_filter_sharing=self.use_filter_sharing
                )
            case "cache":
                model_name = f"{self.model_name}_decoder_n1_cache_token{token_idx}"
                return WhisperDecoderCacheModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=1, token_idx=token_idx,
                    use_future_token_mask=self.use_future_token_mask,
                    use_filter_sharing=self.use_filter_sharing
                )
