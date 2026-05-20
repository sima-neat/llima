import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import numpy as np
import safetensors.numpy
import torch
import transformers
from PIL import Image
from ml_dtypes import bfloat16, int4
from transformers import AutoConfig, MistralConfig
from transformers.image_utils import load_images
from sima_utils.logging.sima_logger import sima_log_info
import os


HF_SINGLE_MODEL_FILENAME = 'model.safetensors'
HF_WEIGHT_INDEX_FILENAME = 'model.safetensors.index.json'


@dataclass
class CompressedTensorsConfig:
    """Configuration for llm-compressor compressed-tensors quantization (AWQ/GPTQ).

    Attributes:
        symmetric: Boolean indicating if all quantization groups are symmetric.
        group_sizes: Group sizes found in quantization groups.
        ignore: List of modules ignored during quantization.
        targets: Target module names found in quantization groups.
    """
    symmetric: bool = True
    group_sizes: list[int] = field(default_factory=list)
    ignore: list[str] = field(default_factory=lambda: ["lm_head"])
    targets: list[str] = field(default_factory=list)

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> Self | None:
        """Parse quantization_config from HuggingFace config.json."""
        quant_config = config.get("quantization_config", {})

        if quant_config.get("quant_method") != "compressed-tensors":
            return None

        config_groups = quant_config.get("config_groups", {})
        if not config_groups:
            raise ValueError("compressed-tensors model is missing config_groups.")

        symmetric = True
        group_sizes: list[int] = []
        targets: list[str] = []
        for group_cfg in config_groups.values():
            weights_config = group_cfg.get("weights", {})
            group_symmetric = weights_config.get("symmetric", True)
            if not group_symmetric:
                symmetric = False
            group_size = weights_config.get("group_size")
            if group_size is not None:
                group_sizes.append(group_size)
            targets.extend(group_cfg.get("targets", []))

        if not symmetric:
            raise ValueError("Only symmetric quantization is supported.")

        return cls(
            symmetric=symmetric,
            group_sizes=group_sizes,
            ignore=quant_config.get("ignore", ["lm_head"]),
            targets=targets or ["Linear"],
        )


def infer_num_bits_from_packed_shape(
    packed_shape: tuple[int, ...], original_shape: tuple[int, ...]
) -> int:
    """Infer quantized weight bit-width from packed and original shapes.

    We expect int32 packing where the pack factor is (32 / num_bits).
    """
    packed_numel = math.prod(packed_shape)
    original_numel = math.prod(original_shape)
    if packed_numel <= 0 or original_numel <= 0:
        raise ValueError(
            f"Invalid tensor sizes for bit inference: packed={packed_shape}, original={original_shape}"
        )
    if original_numel % packed_numel != 0:
        raise ValueError(
            "Cannot infer num_bits because original_numel is not divisible by packed_numel: "
            f"packed={packed_shape}, original={original_shape}"
        )

    pack_factor = original_numel // packed_numel
    # Compressed Tensor format packs int4/int8 weights in int32.
    if 32 % pack_factor != 0:
        raise ValueError(f"Invalid pack factor {pack_factor} for int32 packed weights.")

    num_bits = 32 // pack_factor
    if num_bits not in (4, 8):
        raise ValueError(f"Unsupported inferred num_bits={num_bits}.")
    return num_bits

def unpack_compressed_tensor(
    packed: np.ndarray, original_shape: tuple[int, int], num_bits: int = 4
) -> np.ndarray:
    """
    Unpack sub-byte values from each INT32 element using llm-compressor's
    INTERLEAVED format.

    Args:
        packed: Array of shape (out_features, packed_dim) with int32 dtype
        num_bits: Number of bits per value (e.g. 4 or 8)

    Returns:
        Unpacked array of shape (out_features, packed_dim * (32/num_bits))
        - int4 dtype if num_bits=4
        - int8 dtype if num_bits=8
    """
    if num_bits not in (4, 8):
         raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    packed = packed.astype(np.int32)    
    packed_tensor = torch.from_numpy(packed)
    
    unpacked_tensor = unpack_from_int32(
        value=packed_tensor,
        num_bits=num_bits,
        shape=torch.Size(original_shape),
        packed_dim=1
    )

    # Convert back to numpy and cast to appropriate dtype
    output = unpacked_tensor.numpy()
    if num_bits == 4:
        return output.astype(int4)
    else:  # num_bits == 8
        return output.astype(np.int8)


def unpack_from_int32(
    value: torch.Tensor,
    num_bits: int,
    shape: torch.Size,
    packed_dim: int = 1,
) -> torch.Tensor:
    """
    Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
    original bit range.

    Mirrors compressed_tensors' unpack_from_int32 implementation.
    """
    if value.dtype is not torch.int32:
        raise ValueError(
            f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
        )

    if num_bits not in (4, 8):
        raise ValueError("Unpacking is only supported for num_bits in {4, 8}")

    pack_factor = 32 // num_bits

    # unpack
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked = torch.zeros(
            (value.shape[0], value.shape[1] * pack_factor),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[1])
        unpacked = unpacked[:, :original_row_size]
    elif packed_dim == 0:
        unpacked = torch.zeros(
            (value.shape[0] * pack_factor, value.shape[1]),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[0])
        unpacked = unpacked[:original_row_size, :]
    else:
        raise ValueError("packed_dim must be 0 or 1")

    # bits are packed in unsigned format, reformat to signed
    offset = pow(2, num_bits) // 2
    unpacked = (unpacked - offset).to(torch.int8)

    return unpacked



DEFAULT_LAYER_NAMES = {
    'model': {'key': 'model', 'prefix': 'layers'},
    'language_model': {'key': 'language_model.model', 'prefix': 'layers'},
    'vision_model': {'key': 'vision_tower', 'prefix': 'layers'},
    'lm_head': {'key': 'lm_head'},
    'projector': {'key': 'projector'},
    'norm': {'key': 'model.norm'},
}

HF_LORA_CONFIG_FILE = 'adapter_config.json'
HF_DRAFT_MODEL_CONFIG_FILE = 'config.json'


@dataclass
class LocalHuggingFaceModel:
    """
    LocalHuggingFaceModel:
        A representation of a Local HF Cache with all relevant parts.

    Args:
        hf_cache: Path - The base directory for the HF model cache.
        weights: dict - A map of the weight file name and the path to the cached weights.
        config: dict - The HF Model config file as a dictionary.
        weight_map: dict - A map of weight names to weight files.
        metadata: dict - Any available/relevant metadata.
        layer_names: dict[str, dict] - A dictionary containing layer name and prefix data as described below:
            Layer name prefixes and suffixes
                - Language Models
                    - Block prefix and suffix
                    - Language Model Head prefix and suffix
                - Vision Language Model
                    - Vision Model prefix and suffix
                    - Projector prefix and suffix
                    - Block prefix and suffix
                    - Language Model Head prefix and suffix
    """

    hf_cache: Path
    weights: dict
    config: dict
    weight_map: dict
    metadata: dict
    layer_names: dict
    params: dict | None = None
    compressed_tensors_config: CompressedTensorsConfig | None = None

    @staticmethod
    def create_from_directory(
        directory: str | Path, layer_names: dict | None = None
    ) -> 'LocalHuggingFaceModel':
        """
        create_from_directory - Validate and build a local HF Cache from a user-defined path.
        1. Verify that the model has a config file, and all the weight files are present.
        2. Build a LocalHuggingFaceModel object with resolved paths to the various weight/config
            files, the model configuration, the layername map, and the weight-map, which tells
            us where to find a specific weight.
        Args:
            directory: User provided path to a local HF Cache.
            layer_names: A mapping of layer prefixes, suffixes for each major component in the
                model.
        Returns:
            A LocalHuggingFaceModel object with all relevant attributes.
        """
        directory = Path(directory) if isinstance(directory, str) else directory
        layer_names = layer_names if layer_names else DEFAULT_LAYER_NAMES
        assert set(layer_names.keys()).issubset(DEFAULT_LAYER_NAMES.keys()), (
            f'Ensure that the suffix/prefix mapping uses the following keys: {DEFAULT_LAYER_NAMES}'
        )
        assert directory.exists(), f'{directory} does not exist!'
        assert directory.is_dir(), f'{directory} is not a directory!'
        # Load model config
        config_file = find_file(directory=directory, filename="config.json", resolve=False)
        assert config_file
        hf_config = AutoConfig.from_pretrained(config_file.parent)
        config = hf_config.to_dict()

        # HF has head_dim to be None for Mistral model
        if isinstance(hf_config, MistralConfig) and config["head_dim"] is None:
            config["head_dim"] =  config["hidden_size"] // config["num_attention_heads"]

        # Load and verify HF Index File
        # Some small models do not have an index file, so we need to build one here.
        weight_file = find_file(directory=directory, filename=HF_SINGLE_MODEL_FILENAME)
        if weight_file:
            # Handles the case where only a single .safetensors file is found.
            # This means there is no .index.json, so the weight-map needs to be build manually.
            # Weight-map structure {'model_param_name': 'safetensors_file_name'}
            weight_files = {HF_SINGLE_MODEL_FILENAME: weight_file}
            index_data = {
                'weight_map': {
                    k: HF_SINGLE_MODEL_FILENAME
                    for k in safetensors.numpy.load_file(weight_file).keys()
                }
            }
        else:
            # There is no `model.safetensors`, so we use the `model.safetensors.index.json`.
            index_file = find_file(
                directory=directory, filename=HF_WEIGHT_INDEX_FILENAME
            )
            assert index_file
            with index_file.open('r') as fp:
                index_data = json.load(fp=fp)
            # Check Weights
            safetensors_filenames = set(index_data['weight_map'].values())
            weight_files = {
                filename: find_file(directory=directory, filename=filename)
                for filename in safetensors_filenames
            }
        assert all(isinstance(w, Path) and w.exists() for w in weight_files.values()), (
            f'Could not find all weight files: {weight_files}'
        )
        # Parse llm-compressor quantization config if present
        compressed_tensors_config = CompressedTensorsConfig.from_hf_config(config)
        if compressed_tensors_config:
            sima_log_info(
                "[llm-compressor] Detected compressed-tensors model: "
                "symmetric=%s, group_sizes=%s, targets=%s, ignore=%s",
                compressed_tensors_config.symmetric,
                compressed_tensors_config.group_sizes,
                compressed_tensors_config.targets,
                compressed_tensors_config.ignore
            )

        return LocalHuggingFaceModel(
            hf_cache=directory,
            weights=weight_files,
            config=config,
            weight_map=index_data['weight_map'],
            metadata=index_data.get('metadata', 'N/A'),
            layer_names=layer_names,
            compressed_tensors_config=compressed_tensors_config,
        )

    @property
    def is_gguf(self) -> bool:
        return False

    def is_compressed_tensors_model(self) -> bool:
        """Check if this is an llm-compressor quantized model."""
        return self.compressed_tensors_config is not None

    @staticmethod
    def load_lora_adapter(lora_path: Path | str) -> dict:
        directory = Path(lora_path) if isinstance(lora_path, str) else lora_path
        adapter_path = find_file(directory, HF_LORA_CONFIG_FILE)
        assert adapter_path, f"No LoRA adapter found in the lora path: {lora_path}"
        with adapter_path.open('r') as f:
            lora_config = json.load(f)
        return lora_config

    @staticmethod
    def load_draft_model(draft_model_path: Path | str) -> dict:
        directory = Path(draft_model_path) if isinstance(draft_model_path, str) else draft_model_path
        draft_path = find_file(directory, HF_DRAFT_MODEL_CONFIG_FILE)
        assert draft_path, f"No draft model found in draft_path: {draft_path}"
        with draft_path.open('r') as f:
            draft_model_config = json.load(f)

        # check if draft model is valid
        is_single_layer = draft_model_config.get("num_hidden_layers") == 1
        has_draft_vocab = "draft_vocab_size" in draft_model_config
        if not(is_single_layer and has_draft_vocab):
            raise ValueError(
                f"Model at {draft_model_path} does not appear to be a valid draft model. "
                f"Expected num_hidden_layers=1 and a draft_vocab_size field in config.json."
            )
        return draft_model_config

    def param_exists(self, param_name: str) -> bool:
        """Check if a parameter exists in the model.
        
        For llm-compressor models, also checks for packed weight versions.
        """
        if self.params and param_name in self.params:
            return True
        if self.weight_map.get(param_name) is not None:
            return True
        # For llm-compressor models, check for packed version
        if self.is_compressed_tensors_model() and param_name.endswith('.weight'):
            packed_name = param_name + "_packed"
            if self.weight_map.get(packed_name) is not None:
                return True
        return False

    def load_np_param(self, param_name: str) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Load a parameter into a numpy array.

        Args:
            param_name: Parameter name, as it appears in the HF Cache

        Returns:
            The parameter as an numpy array, or a tuple of (scales, weights) if quantized.
        """
        if self.params and param_name in self.params:
            return self.params[param_name]

        # For llm-compressor models, check for packed version
        packed_name = param_name + "_packed"
        if packed_name in self.weight_map:
            # Load the three associated tensors for quantized weights
            scale_name = param_name + "_scale"
            shape_name = param_name + "_shape"

            packed = self._load_raw_tensor(packed_name)
            scale = self._load_raw_tensor(scale_name)
            shape_tensor = self._load_raw_tensor(shape_name)
            zero_point_name = param_name + "_zero_point"
            if zero_point_name in self.weight_map:
                raise ValueError(
                    f"Asymmetric quantization is not supported for {param_name} "
                    f"(found tensor {zero_point_name})."
                )

            original_shape = tuple(shape_tensor.astype(int))
            num_bits = infer_num_bits_from_packed_shape(packed.shape, original_shape)

            unpacked = unpack_compressed_tensor(
                packed, original_shape=original_shape, num_bits=num_bits
            )
            if unpacked.size != math.prod(original_shape):
                raise ValueError(
                    f"Unpacked weight size mismatch for {param_name}: "
                    f"got {unpacked.size}, expected {math.prod(original_shape)} "
                    f"from shape {original_shape}."
                )
            scale = scale.astype(np.float32)

            if np.any(scale < 0):
                raise ValueError(
                f"Negative scales found in {param_name}. "
                "The compiler requires all quantization scales to be positive. "
                "Please re-quantize the model with positive scales."
                )

            return (scale, unpacked)

        # shard -> the safetensors symlink file, usually called 'model-00001-of-00002.safetensors'
        model_shard = self.weight_map.get(param_name)
        assert model_shard, f'Could not find {param_name} in the model cache.'

        safetensors_file = self.weights[model_shard]
        return safetensors.numpy.load_file(filename=safetensors_file)[param_name]

    def _load_raw_tensor(self, tensor_name: str) -> np.ndarray:
        """Load a raw tensor from safetensors files by exact name."""
        if self.params and tensor_name in self.params:
            return self.params[tensor_name]

        model_shard = self.weight_map.get(tensor_name)
        if not model_shard:
            raise KeyError(f'Could not find {tensor_name} in the model cache.')

        safetensors_file = self.weights[model_shard]
        return safetensors.numpy.load_file(filename=safetensors_file)[tensor_name]


    def load_all_params(self):
        self.params = dict()
        for safetensors_filename in self.weights.values():
            self.params.update(safetensors.numpy.load_file(filename=safetensors_filename))

    def unload_all_params(self):
        self.params = None

    @cached_property
    def vision_model_param_base_name(self) -> str:
        # Known vision model base name segments (extensible for future models).
        vision_base_names = ("vision_tower", "visual")
        
        # Filter for keys that have a known vision base name as a path segment.
        vision_keys = [
            name for name in self.weight_map.keys()
            if any(seg in name.split(".") for seg in vision_base_names)
        ]

        if not vision_keys:
            raise RuntimeError(
                f"Cannot determine param base name: No vision keys found in weight map."
                f" Looked for segments: {vision_base_names}. Keys available: {self.weight_map.keys()}"
            )

        # Find the longest common prefix of all vision keys.
        common_prefix = os.path.commonprefix(vision_keys)

        if '.' in common_prefix:
            common_prefix = common_prefix.rsplit('.', 1)[0]

        if not common_prefix:
            raise RuntimeError(f"Could not find a common prefix for vision keys: {vision_keys}")

        return common_prefix

    @cached_property
    def language_model_param_base_name(self) -> str:
        for name in self.weight_map.keys():
            parts = name.split(".")
            if "language_model" in parts and "layers" in parts:
                return ".".join(parts[:parts.index("layers")])

        base_names = list()
        for name in self.weight_map.keys():
            if "lm_head" in name:
                continue
            for part_name in name.split("."):
                if part_name in ("language_model", "model"):
                    base_names.append(part_name)
                elif base_names:
                    return ".".join(base_names)
            if base_names:
                break
        raise RuntimeError(
            "Cannot determine the param base name for language model from the weight names:"
            f" {self.weight_map.keys()}."
        )

    def load_param(
        self, component: str, layer_idx: int, parameter_name: str
    ) -> np.ndarray:
        """
        load_param Load a huggingface parameter from a layer index, component and the paramater name.

        Args:
            component: LLM Component, from DEFAULT_LAYER_NAMES
            layer_idx: Layer index (int). If the integer is positive, the parameter from that block
                index is used. If the number is negative, then there is no block_index associated with
                that parameter.
            parameter_name: paramater name

        Returns:
            The parameter as a numpy array.
        """
        assert component in DEFAULT_LAYER_NAMES.keys(), (
            f'Component {component} is not in {DEFAULT_LAYER_NAMES.keys()}'
        )
        layer_idx = layer_idx if layer_idx >= 0 else ''
        component_key = self.layer_names.get(component, {}).get('key', '')
        weight_suffix = self.layer_names.get(component, {}).get('suffix', '')
        weight_prefix = self.layer_names.get(component, {}).get('prefix', '')
        block_key = '.'.join(
            (str(b) for b in (weight_prefix, layer_idx, weight_suffix) if b != '')
        )
        weight_key = '.'.join(
            (str(w) for w in (component_key, block_key, parameter_name) if w != '')
        )
        return self.load_np_param(param_name=weight_key)

    def execute_hf(
        self,
        messages: list[dict[str, str | list[dict[str, str]]]],
        images: list[str | Path | Image.Image] | None = None,
        device: str = 'cpu',
        do_sample: bool = False,
        top_p: bool = None,
        max_new_tokens: int = 20,
        num_return_sequences: int = 1,
    ) -> dict:
        """
        execute_hf Execute an LLM using the transformers library.

        Resolving Inference Device:
            Verify torch device - You can input an invalid device ordinal (ex. 'cuda:100' on a machine with 1 GPU)
            This will only throw an error, once you try to do an operation on that device, so adding code for that.

        Loading HF LLM:
            The base cache path + the relative path from the base path to the model save files.
            <Model Root>
            ├── blobs
            ├── refs
            └── snapshots
                └── <hash>
                    ├── config.json -> <symlink>
                    ├── ... LLM files
                    └── model.safetensors.index.json -> <symlink>
            Loading a transformer from a pretrained folder requires the unresolved snapshots/hash/
            Every model must have a config saved in a config.json

        Args:
            query: User query in string.
            input_ids: User query in a list of token ids.
            images: List of images.
            device: Inference device. Defaults to 'cpu'.
            do_sample: Whether or not to use sampling ; use greedy decoding otherwise. Defaults to False (greedy decoding).
            top_p: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
            max_new_tokens: The maximum length the generated tokens can have. Defaults to 20.
            num_return_sequences: The number of independently computed returned sequences for each element in the batch.. Defaults to 1.

        Raises:
            ValueError: When the provided inference device is invalid.

        Returns:
            A dict contaning prompt, input_ids, images, output tokens and text.
        """
        try:
            device = torch.device(device=device)
            torch.ones((1, 1, 1, 1), device=device)

        except RuntimeError as re:
            raise ValueError(
                f'Device: {device} is invalid. Refer to https://pytorch.org/docs/stable/tensor_attributes.html#torch-device for valid device types. {re}'
            ) from re

        # HF Model Loading
        config_dir = find_file(
            directory=self.hf_cache, filename='config.json', resolve=False
        )
        subfolder = config_dir.relative_to(self.hf_cache).parent

        arch = self.config["architectures"][0]
        if "ForConditionalGeneration" in arch:
            automodel = getattr(transformers, arch)
        else:
            automodel = transformers.AutoModelForCausalLM

        with torch.no_grad() and device:
            model = automodel.from_pretrained(
                self.hf_cache, subfolder=subfolder, local_files_only=True
            ).to(device)

            # Note: the output is different between use_fast=True and False.
            # use_fast=True: pixle_values is list of torch tensor.
            # use_fast=False: pixle_values is list of numpy array.
            processor = transformers.AutoProcessor.from_pretrained(
                self.hf_cache, subfolder=subfolder, local_files_only=True, use_fast=True
            )

            is_multimodal = hasattr(processor, "image_processor")
            if is_multimodal:
                for message in messages:
                    if isinstance(message["content"], str):
                        message["content"] = [{"type": "text", "text": message["content"]}]

            if processor.chat_template is not None:
                formatted_prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                assert len(messages) == 1 and messages[0]["role"] == "user"
                content = messages[0]["content"]
                assert isinstance(content, str)
                formatted_prompt = content

            if is_multimodal:
                if images:
                    images = [str(image) if isinstance(image, Path) else image for image in images]
                    images = load_images(images)
                else:
                    images = None

                if (
                    processor.tokenizer.bos_token is not None
                    and formatted_prompt.startswith(processor.tokenizer.bos_token)
                ):
                    add_special_tokens = False
                else:
                    add_special_tokens = True
                processed_dict = processor(
                    text=formatted_prompt, images=images, return_tensors="pt",
                    add_special_tokens=add_special_tokens
                )
            else:
                assert not images
                processed_dict = processor(
                    text=formatted_prompt, add_special_tokens=False, return_tensors="pt"
                )

            input_ids = processed_dict["input_ids"]
            pixel_values = processed_dict.get("pixel_values")

            output = model.generate(
                input_ids=input_ids, pixel_values=pixel_values,
                do_sample=do_sample,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
            )
        output = output[0, input_ids.shape[1]:]
        output_tokens = output.cpu().numpy()
        generated_text = processor.decode(output, skip_special_tokens=True)

        res = {}
        res["prompt"] = formatted_prompt
        res["input_ids"] = input_ids
        res["pixel_values"] = pixel_values
        res["output_tokens"] = output_tokens.tolist()
        res["generated_text"] = generated_text
        return res


def find_file(directory: Path, filename: str, resolve: bool = True) -> Path | None:
    """
    find_file Utility function to recursively find a file within a directory.
        - HF hashes/model names will be unique, but they might be nested differently, so this
        function is a workaround.

    Args:
        directory: Directory with model files/weights.
        filename: filename (stem).
        resolve: Whether or not the path needs to be resolved.

    Returns:
        A resolved path to the file. This will account for the symlinks in HF Caches.
        If the file is not found, it returns None.
    """

    # Resolve the symlinks.
    # Some HF Caches have a .no_exist folder with erroneous files, these need to be excluded.
    matches = []
    for x in directory.rglob(filename):
        if '.no_exist' not in str(x) and "sima_files" not in str(x):
            matches.append(x.resolve() if resolve else x)
            
    # These need to be unique, they will be stored under a unique hash.
    if len(matches) == 0:
        return None
    assert len(matches) == 1, f'Found multiple instances of {filename} in {directory}!'
    return matches[0]

if __name__ == '__main__':
    import sys

    hf_path = Path(sys.argv[1]).absolute()
    image_path = Path(sys.argv[2]).absolute() if len(sys.argv) > 2 else None

    if image_path is not None:
        images = [image_path]
        query = "What is shown in this image? "
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": query}
        ]
    else:
        images = None
        if "paligemma" in str(hf_path):
            query = "Describe the capital of Italy "
        else:
            query = "What is capital of Italy? "
        content = query
    messages = [{"role": "user", "content": content}]

    hf_cache = LocalHuggingFaceModel.create_from_directory(
        directory=hf_path,
        layer_names=None,  # Not supplying a layername here, the default mapping works for LLaMa/Gemma
    )

    max_output_token_length = 30
    results = hf_cache.execute_hf(
        messages=messages, images=images, max_new_tokens=max_output_token_length, device='cpu'
    )

    print("\n")
    print(f"Model: {hf_path.name}")
    print(f"Image: {image_path}")
    print(f'Prompt: {results["prompt"]}')
    print(f'Output tokens: {results["output_tokens"]}')
    print(f'Response: {results["generated_text"]}')
    print("\n")
