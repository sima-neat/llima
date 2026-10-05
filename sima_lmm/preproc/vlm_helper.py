from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from pprint import pprint
import re
import time
from typing import Any

import PIL
import llama_cpp
import numpy as np
import cv2

from transformers import (
    AutoProcessor, AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase, ProcessorMixin
)
from transformers.image_utils import load_images
from transformers.utils import GENERATION_CONFIG_NAME

from sima_lmm.config.vlm_config import VlmConfig,VisionArchType
from sima_utils.logging.sima_logger import sima_log_warning, sima_log_info

class VlmHelper:
    """VLM helper class with processors."""

    hf_tokenizer: PreTrainedTokenizerBase | None = None
    hf_processor: ProcessorMixin | None = None
    system_prompt: str | None = None
    stop_tokens: list[int]
    input_res: list[int] | int | None = None

    def __init__(
        self,
        vlm_cfg: VlmConfig,
        hf_path: Path,
    ):
        try:
            if hf_path.suffix == ".gguf":
                self.hf_tokenizer = LlamaCPPWrapper(
                    model_path=str(hf_path), vocab_only=True, verbose=False
                )
                self.hf_processor = None
            else:
                # Use tokenizer_config.json to find the actual directory with the HuggingFace files.
                # Assuming all the files are in the same directory.
                candidate_file_names = [
                    "tokenizer_config.json", "tokenizer.json", "tokenizer.model"
                ]
                for candidate_file_name in candidate_file_names:
                    candidates = list(hf_path.rglob(candidate_file_name))
                    if not candidates:
                        continue
                    assert len(candidates) == 1
                    hf_path = candidates[0].parent
                    break
                else:
                    raise RuntimeError(
                        f"Cannot find files from path: path={hf_path}, files={candidates}"
                    )

                # use custom chat_template if specified by user
                kwargs = {}
                if vlm_cfg.pipeline_cfg.chat_template is not None:
                    kwargs["chat_template"] = vlm_cfg.pipeline_cfg.chat_template

                # Slow image processor is actually faster than the fast image processor on the
                # devkit but the reason is unknown. Currently transformers uses slow image processor
                # and fast tokenizer if use_fast is not specified. Until transformers fixed the
                # default to use fast image processor, we can create the processor without
                # specifying the use_fast.
                if vlm_cfg.is_multimodal:
                    self.hf_processor = AutoProcessor.from_pretrained(hf_path, **kwargs)
                    self.hf_tokenizer = self.hf_processor.tokenizer
                    assert isinstance(self.hf_processor, ProcessorMixin)
                    if vlm_cfg.vm_cfg.arch in (
                        VisionArchType.SIGLIP2,
                        VisionArchType.QWEN2_VISION_ENCODER,
                        VisionArchType.QWEN3_VISION_ENCODER,
                        VisionArchType.QWEN3_5_VISION_ENCODER,
                    ):
                        self.input_res = vlm_cfg.vm_cfg.image_size # lfm2,qwen vl needs to be statically resized before hf preprocessor.
                else:
                    self.hf_processor = None
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_path, **kwargs)
                    assert isinstance(self.hf_tokenizer, PreTrainedTokenizerBase)

            # Determine the stop tokens.
            if (hf_path / GENERATION_CONFIG_NAME).is_file():
                hf_generation_config = GenerationConfig.from_pretrained(
                    hf_path, use_cache=False, cache_implementation=None
                )
                eos_token_id = hf_generation_config.eos_token_id
            else:
                eos_token_id = self.hf_tokenizer.eos_token_id

            self.set_system_prompt(vlm_cfg.pipeline_cfg.system_prompt)
            self.set_chat_template(vlm_cfg.pipeline_cfg.chat_template)

            if isinstance(eos_token_id, list):
                self.stop_tokens = eos_token_id
            else:
                assert isinstance(eos_token_id, int)
                self.stop_tokens = [eos_token_id]
            if self.is_multimodal != vlm_cfg.is_multimodal:
                sima_log_warning(
                    "Inconsistent settings for is_multimodal: vlm_cfg=%s, hf=%s",
                    vlm_cfg.is_multimodal,
                    self.is_multimodal,
                )

        except Exception as e:
            raise RuntimeError(f"Cannot create HuggingFace Processor from {hf_path}") from e

    def set_system_prompt(self, system_prompt: str | None):
        self.system_prompt = system_prompt

    def set_chat_template(self, chat_template: str | None):
        self.chat_template = chat_template

    def preprocess(self, chat: "Chat") -> tuple[str, list[int], list[np.ndarray] | None]:
        """Preprocess the input query and the images.

        Args:
            Chat: Input chat with messages and images.

        Returns:
            Tuple of formatted prompt, tokenized input query and preprocessed images.
        """
        # Use HF processor or Llama CPP wrapper to preprocess the messages and images.
        if self.hf_tokenizer.chat_template is not None:
            formatted_prompt = self.hf_tokenizer.apply_chat_template(
                chat.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            formatted_prompt = chat.get_last_query()

        if self.hf_processor is None:
            if chat.images:
                raise RuntimeError("Image(s) are not expected for this model")
            processed_dict = self.hf_tokenizer(text=formatted_prompt, add_special_tokens=False)
            input_ids = processed_dict["input_ids"]
            return formatted_prompt, input_ids, None, None
        else:
            assert isinstance(self.hf_processor, ProcessorMixin)
            if chat.images:
                images = [
                    str(image) if isinstance(image, Path) else image for image in chat.images
                ]
                images = load_images(images)
            else:
                images = None
            if self.input_res is not None and images is not None:
                if isinstance(self.input_res, list):
                    target_h, target_w = self.input_res
                else:
                    target_h = target_w = self.input_res
                target_size = (target_w, target_h)
                resized_images = []
                for img in images:
                    img_np = np.array(img)
                    resized_img_np = cv2.resize(img_np, target_size, interpolation=cv2.INTER_CUBIC)
                    resized_images.append(PIL.Image.fromarray(resized_img_np))
                images = resized_images
            if self.hf_tokenizer.bos_token is not None and formatted_prompt.startswith(
                self.hf_tokenizer.bos_token
            ):
                add_special_tokens = False
            else:
                add_special_tokens = True

            processed_dict = self.hf_processor(
                text=formatted_prompt, images=images, add_special_tokens=add_special_tokens
            )
            processed_images = processed_dict.get("pixel_values")
            image_grid_thw = processed_dict.get("image_grid_thw")
            if image_grid_thw is not None and not isinstance(image_grid_thw, list):
                    image_grid_thw = np.array(image_grid_thw).tolist()
            # Processed_images can be a numpy array or a list of numpy arrays
            # or a list of torch.Tensor or a single torch.Tensor.
            if processed_images is not None:
                if isinstance(processed_images, list):
                    if isinstance(processed_images[0], np.ndarray):
                        if processed_images[0].ndim > 2:
                            processed_images = [
                                img.transpose((1, 2, 0)) for img in processed_images
                            ]
                    else:
                        # Assuming torch.Tensor.
                        if processed_images[0].ndim > 2:
                            processed_images = [
                                img.permute(1, 2, 0).numpy() for img in processed_images
                            ]
                        else:
                            # (B,num_patches,patch_feature_size) for lfm2-vl.
                            processed_images = [img.numpy() for img in processed_images]
                # The processor returned a single NumPy array.
                elif isinstance(processed_images, np.ndarray):
                    # A 4D array of shape (B, C, H, W) that needs transposition.
                    if processed_images.ndim == 4:
                        processed_images = processed_images.transpose((0, 2, 3, 1))
                    elif processed_images.ndim == 2: 
                            # Qwen VL case with shape seq_len,patch_feature_size
                            num_images=len(images)
                            seq_len = processed_images.shape[0] // num_images
                            patch_dim = processed_images.shape[1]
                            processed_images = processed_images.reshape(num_images, seq_len, patch_dim)
                else:
                    # Assuming torch.Tensor.
                    if processed_images.ndim == 4:
                        processed_images = processed_images.permute(0, 2, 3, 1).numpy()
                    elif processed_images.ndim == 3:  # (B,num_patches,patch_feature_size) for lfm2-vl.
                        processed_images = processed_images.numpy()
                    elif processed_images.ndim == 2:
                        # Qwen VL case with shape seq_len,patch_feature_size
                        num_images=len(images)
                        seq_len = processed_images.shape[0] // num_images
                        patch_dim = processed_images.shape[1]
                        processed_images = processed_images.numpy().reshape(num_images, seq_len, patch_dim)

            return formatted_prompt, processed_dict["input_ids"][0], processed_images, image_grid_thw

    def encode(self, text: str) -> list[int]:
        return self.hf_tokenizer.encode(text)

    def decode(self, output_tokens: list[int]) -> str:
        return self.hf_tokenizer.decode(output_tokens)

    def multimodal_concat(
        self,
        text_token_ids: list[int],
        vision_proj_list: list[np.ndarray],
        embed_weight: np.ndarray,
    ) -> np.ndarray:
        """Combine text and vision embedding tensors.

        Args:
            text_token_ids: The text token ids with image placeholder.
            vision_proj_list: The vision projection tensors.
            embed_weight: The embedding weight matrix.

        Returns:
            A tensor with combined text and vision embeddings.
        """
        assert vision_proj_list[0].shape[-1] == embed_weight.shape[-1]

        num_image_token_ids_per_image = prod(vision_proj_list[0].shape[:-1])

        hidden_size = embed_weight.shape[-1]
        total_tokens = len(text_token_ids)
        result = np.empty((total_tokens, hidden_size), dtype=embed_weight.dtype)

        image_token_id = self.image_token_id
        text_positions: list[int] = []
        text_ids: list[int] = []
        image_spans: list[tuple[int, int]] = []
        idx = 0
        image_count = 0
        while idx < total_tokens:
            token_id = text_token_ids[idx]
            if image_token_id is not None and token_id == image_token_id:
                image_spans.append((idx, image_count))
                idx += num_image_token_ids_per_image
                image_count += 1
            else:
                text_positions.append(idx)
                text_ids.append(token_id)
                idx += 1
        if text_ids:
            result[np.asarray(text_positions, dtype=np.int32)] = embed_weight[np.asarray(text_ids)]
        for dest_idx, proj_idx in image_spans:
            vision_proj = vision_proj_list[proj_idx].reshape(-1, hidden_size)
            result[dest_idx : dest_idx + vision_proj.shape[0]] = vision_proj
        return result

    @property
    def is_multimodal(self) -> bool:
        return self.hf_processor is not None

    @property
    def image_token_id(self) -> int | None:
        
        image_token_id=getattr(
            self.hf_tokenizer,
            "image_token_id",
            self.hf_tokenizer.added_tokens_encoder.get("<image>"),
        )
        if image_token_id:
            return image_token_id
        else:  # Qwen case
            return getattr(
                self.hf_tokenizer,
                "image_token_id",
                self.hf_tokenizer.added_tokens_encoder.get("<|image_pad|>"),
            )


@dataclass
class Chat:
    vlm_helper: VlmHelper
    messages: list[dict[str, str | list[dict[str, str]]]] = field(default_factory=list)
    images: list[str | Path | PIL.Image.Image] | None = None

    def __post_init__(self):
        self.set_system_prompt(self.vlm_helper.system_prompt)

    def set_system_prompt(self, system_prompt: str | None):
        self.vlm_helper.set_system_prompt(system_prompt)

        self.messages.clear()
        if not system_prompt:
            return

        if self.vlm_helper.is_multimodal:
            content = [{"type": "text", "text": system_prompt}]
        else:
            content = system_prompt
        self.messages.append({"role": "system", "content": content})

    def get_system_prompt(self) -> str | None:
        if not self.messages or self.messages[0]["role"] != "system":
            return None

        if self.vlm_helper.is_multimodal:
            return self.messages[0]["content"][0]["text"]
        else:
            return self.messages[0]["content"]

    def add_query(self, query: str):
        if self.vlm_helper.is_multimodal:
            num_images_in_messages = 0
            for message in self.messages:
                if isinstance(message["content"], str):
                    continue
                for item in message["content"]:
                    if item["type"] == "image":
                        num_images_in_messages += 1
            content = []
            if self.images:
                for idx in range(num_images_in_messages, len(self.images)):
                    content.append({"type": "image", "image": self.images[idx]})
            content.append({"type": "text", "text": query})
        else:
            content = query
        self.messages.append({"role": "user", "content": content})

    def get_last_query(self) -> str:
        if not self.messages:
            return ""
        content = self.messages[-1]["content"]
        if self.vlm_helper.is_multimodal:
            query_items = list(filter(lambda x: x["type"] == "text", content))
            if len(query_items) != 1:
                raise RuntimeError(f"Only expect one text query but got {len(query_items)}")
            return query_items[0]["text"]
        else:
            return content

    def update_last_query(self, query: str):
        if self.messages:
            self.messages.pop()
        self.add_query(query)

    def add_response(self, response: str):
        if self.vlm_helper.is_multimodal:
            self.messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )
        else:
            self.messages.append({"role": "assistant", "content": response})

    def clear_messages(self):
        if not self.messages:
            return
        if self.messages[0]["role"] == "system":
            clear_start_idx = 1
        else:
            clear_start_idx = 0
        self.messages[clear_start_idx:] = []

    def add_image(self, image: str | Path | PIL.Image.Image):
        if self.images is None:
            self.images = [image]
        else:
            self.images.append(image)

    def extract_images_from_messages(self):
        if not self.vlm_helper.is_multimodal:
            # Not a multimodal model. Nothing to be done.
            return

        if self.images:
            # Images are already populated. Nothing to be done.
            return

        self.images = list()
        for message in self.messages:
            sima_log_info(f'message["content"]: {message["content"]}')
            if not isinstance(message["content"], list):
                continue 
            
            for item in message["content"]:
                if item["type"] != "image":
                    continue
                self.images.append(item["image"])
                del item["image"]

    def clear_images(self):
        if self.images is not None:
            self.images.clear()

    def clear_history(self):
        self.clear_messages()
        self.clear_images()

    def print_history(self):
        pprint(self.messages)


class LlamaCPPWrapper(llama_cpp.Llama):
    """Wrapper around the llama.cpp Llama class to provide the same API as HF tokenizer."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Create the chat template separately because llama_cpp.Llama does not provide API to apply
        # chat template without inference.
        # The following code is copied from https://github.com/abetlen/llama-cpp-python.
        self.eos_token = (
            self._model.token_get_text(self.eos_token_id) if self.eos_token_id != -1 else ""
        )
        self.bos_token = (
            self._model.token_get_text(self.bos_token_id) if self.bos_token_id != -1 else ""
        )

        if not (chat_template := self.metadata.get("tokenizer.chat_template")):
            raise NotImplementedError("GGUF file without chat template is not supported")
        self.chat_template = chat_template
        self.chat_formatter = llama_cpp.llama_chat_format.Jinja2ChatFormatter(
            template=self.chat_template,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            stop_token_ids=[self.eos_token_id],
        )

        # Find image token from the chat template.
        if image_token_search := re.search(r"<.*image.*>", self.chat_template):
            self.image_token = image_token_search[0]
            self.image_token_id = self.encode(self.image_token, False)[0]
        else:
            self.image_token = None
            self.image_token_id = None

        # This is added to keep the same interface of HF that is used to find the image token.
        self.added_tokens_encoder = dict()

    @property
    def eos_token_id(self) -> int:
        return self.token_eos()

    @property
    def bos_token_id(self) -> int:
        return self.token_bos()

    def apply_chat_template(
        self, messages: list, *, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool
    ) -> str:
        assert add_generation_prompt
        assert not tokenize
        result = self.chat_formatter(messages=messages)
        return result.prompt

    def __call__(self, text: str, *args: Any, **kwargs: Any) -> dict:
        input_ids = self.encode(text, False)
        return {"input_ids": input_ids}

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        return self.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)

    def decode(self, output_tokens: list[int]) -> str:
        return self.detokenize(output_tokens).decode("utf-8")
