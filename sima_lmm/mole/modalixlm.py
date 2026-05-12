#########################################################
# Copyright (C) 2026 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################

"""
Defines wrappers around `lm_eval.models.huggingface.HFLM` to support the Modalix-ZMQ interface.
"""

import json
import time
from pathlib import Path
from typing import Any, Literal

import msgpack
import numpy as np
import torch
import transformers
import zmq
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import MultiTokenEOSCriteria, stop_sequences_criteria
from loguru import logger

from sima_lmm.mole import performance_bench
from sima_lmm.mole.modalixboard import ModalixBoard


class HuggingFaceLM(HFLM):
    """
    HuggingFaceLM Wrapper around `lm_eval.models.huggingface.HFLM` to warn folks when they run CPU
    inferences.
    """

    def __init__(self, *args, **kwargs) -> None:
        logger.info(f"Instantiating {self.__class__.__name__}")
        kwargs["batch_size"] = kwargs.get("batch_size", "auto")  # No memory contraints on GPU.
        super().__init__(*args, **kwargs)

    def _warn_device(self) -> None:
        if "cuda" not in self.device.type:
            logger.warning(f"Running inference on a {self.device}! This may take a very long time!")

    def loglikelihood(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        self._warn_device()
        return super().loglikelihood(requests, disable_tqdm)

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        self._warn_device()
        return super().loglikelihood_rolling(requests, disable_tqdm)

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        self._warn_device()
        return super().generate_until(requests, disable_tqdm)


class ModalixLM(HFLM):
    _supported_hf_lms = [transformers.AutoModelForCausalLM]

    def __init__(self, board: ModalixBoard, *args, **kwargs) -> None:
        logger.info(f"Instantiating {self.__class__.__name__}")
        assert int(kwargs.get("batch_size", -1)) == 1, "batch_size must be set to 1!"
        super().__init__(*args, **kwargs)

        assert self.AUTO_MODEL_CLASS in self._supported_hf_lms, (
            f"Model <{self.AUTO_MODEL_CLASS}> is not supported!"
            f" This evaluation suite only supports {self._supported_hf_lms}"
        )

        self.profile_data = []
        self.last_profile_data: dict = {}

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.curve_publickey, self.zmq_socket.curve_secretkey = zmq.curve_keypair()
        self.zmq_socket.curve_serverkey = b"xA?#1AE663fk][M)Dd9}x?#nV2Iy!p3^&>kerN.>"
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 10 * 60 * 1000)  # 10mins.
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, 10 * 60 * 1000)  # 10mins.
        self.zmq_socket.connect(addr=board.tcp_uri)
        logger.info(f"ZMQ Client connected to: {board.tcp_uri}")

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,  # noqa: ARG002 - Not supporting Seq2Seq right now.
        labels: torch.Tensor | None = None,  # noqa: ARG002 - Not supporting Seq2Seq right now.
    ) -> torch.Tensor:
        """
        Args:
            inps: A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
                [batch, sequence_ctx]. The size of sequence may vary from call to call.
            attn_mask: A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
                (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM.
            labels: A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
                (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM.
        Returns:
            A torch tensor of shape [batch, sequence, vocab] with the logits returned from the
            model's decoder
        """
        zmq_res = self._zmq_infer(tensor=inps, request_type="model_call")
        logger.debug(f"Inp: {inps.size()} ZMQ: {zmq_res.size()}")
        return zmq_res

    def _zmq_infer(
        self,
        tensor: torch.Tensor,
        request_type: Literal["model_call", "generate"],
        **kwargs,
    ) -> torch.Tensor:
        """
        _zmq_infer Run LLM Inference over ZMQ. This does not support batched inference.

            Input shape     : [1, num_tokens] # Batch frozen to 1
            Ouptut Shape    : [1, num_tokens, vocab_size] # Batch frozen to 1
        Args:
            tensor: Model inputs.

        Returns:
            Model outputs.
        """
        profile_data = {}
        assert tensor.size(0) == 1, f"ZMQ does not support batched inference, got {tensor.size()}"

        device = tensor.device
        np_tensor = tensor.detach().cpu().numpy()  # [1, N]
        np_tensor = np_tensor.astype(np.uint32)

        request_metadata = {
            "type": request_type,
            "tensor_dtype": np_tensor.dtype.name,
            "tensor_shape": np_tensor.shape,
            **kwargs
        }
        request_messages = [
            msgpack.packb(request_metadata, use_bin_type=True),
            np_tensor.tobytes()
        ]

        # ZMQ Step
        zmq_t0 = time.perf_counter_ns()
        self.zmq_socket.send_multipart(request_messages)
        response_messages = self.zmq_socket.recv_multipart(track=True)
        zmq_t1 = time.perf_counter_ns()
        assert len(response_messages) == 2

        # ZMQ Post-Proc
        response_metadata = msgpack.unpackb(response_messages[0], raw=False)
        assert "tensor_dtype" in response_metadata and "tensor_shape" in response_metadata
        match response_metadata["tensor_dtype"]:
            case "uint32":
                out_dtype = torch.uint32
            case "bfloat16":
                out_dtype = torch.bfloat16
            case "float64":
                out_dtype = torch.float64
            case _:
                raise RuntimeError(f"Unexpected dtype: {response_metadata['tensor_dtype']}")
        out_shape = response_metadata["tensor_shape"]
        out_tensor = torch.frombuffer(response_messages[1], dtype=out_dtype).view(*out_shape)
        out_tensor = out_tensor.to(device=device)

        profile_data["in_tokens"] = tensor.size(1)
        profile_data["out_tokens"] = out_tensor.size(1)
        profile_data["zmq_infer_time"] = zmq_t1 - zmq_t0
        self.profile_data.append({**profile_data, **response_metadata, "tensor_data": out_tensor})
        return out_tensor

    def _calc_dump_profile(self, filename: str = "profile-dump"):
        dumpfile = Path("profiling") / (f"profile-{filename}.json")
        logger.info(f"Dumping profile data -> {dumpfile}")
        if len(self.profile_data) == 0:
            logger.debug("Nothing to profile!")
            return
        with dumpfile.open("w") as fp:
            json.dump(self.profile_data, fp, sort_keys=True)

    def perf_bench(self, n_samples: int = 128, max_new_tokens: int = 256):
        """
        perf_bench Run performance benchmark using LAMBADA (cimec/lambada) dataset.

        Args:
            n_samples: The number of unique text samples to generate for each context length bucket.
            max_new_tokens: The maximum number of new tokens to generate.
        """
        logger.info("Starting performance benchmark using sample data from cimec/lambada")
        sample_data = performance_bench.create_sample_data(
            tokenizer=self.tokenizer,
            max_toks=self.max_length,
            n_samples=n_samples,
        )

        def inference_fn(tokens: list[int]) -> list[float]:
            context = torch.tensor([tokens], device=self.device)
            generate_time_tensor = self._zmq_infer(
                tensor=context,
                request_type="generate_for_perf",
                max_num_tokens=min(self.max_length, len(tokens) + max_new_tokens)
            )
            return generate_time_tensor.cpu().tolist()[0]

        logger.info('Running Performance Benchmark')
        perf_stats = performance_bench.perf_bench(
            data=sample_data,
            inference_fn=inference_fn
        )

        logger.info("Summarizing Results")
        summary = performance_bench.summarize(
            data=perf_stats, model_name=self._model.config.name_or_path
        )
        return summary

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        _model_generate Generates text from a context, prepares the arguments for ZMQ inference, and
            then runs that ZMQ inference.

        Args:
            context: The input tensor of token IDs.
            max_length: The maximum length of the generated sequence.
            stop: A list of stop-tokens (as strings).
            **generation_kwargs: Additional keyword arguments for the generation process, like
                'temperature' or 'do_sample'.

        Returns:
            The generated sequence of token IDs.
        """
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)  # type: ignore
        do_sample = generation_kwargs.get("do_sample")

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding
        # strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False  # type: ignore

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(  # type: ignore
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        stop_tokens = [tok for tok_list in stopping_criteria for tok in tok_list.sequence_ids]

        generation = self._zmq_infer(
            tensor=context, request_type="generate", max_num_tokens=max_length,
            stop_token_ids=stop_tokens
        )
        return generation

    def _generate_torch(
        self,
        context: torch.Tensor,
        max_length: int,
        stopping_criteria: list[MultiTokenEOSCriteria],
        pad_token_id: int,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        _generate_torch Generates text from a context using the HF Model and torch mixed precision.
        This method is a wrapper around HF's `model.generate()` call. It passes the context,
        stopping criteria, and other generation arguments directly to the model.

        Args:
            context: The input tensor of token IDs.
            max_length: The maximum length of the generated sequence.
            stopping_criteria: A list of criteria objects that determine when to stop generation.
            pad_token_id: The ID of the padding token.
            **generation_kwargs: Additional keyword arguments passed directly to the
                `model.generate()` method.

        Returns:
            The generated sequence of token IDs, including the input context.
        """
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )

    def _model_call_torch(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,  # noqa: ARG002 - Not supporting Seq2Seq right now.
        labels: torch.Tensor | None = None,  # noqa: ARG002 - Not supporting Seq2Seq right now.
    ) -> torch.Tensor:
        """
        Args:
            inps: A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
                [batch, sequence_ctx]. The size of sequence may vary from call to call.
            attn_mask: A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
                (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM.
            labels: A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
                (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM.
        Returns:
            A torch tensor of shape [batch, sequence, vocab] with the logits returned from the
            model's decoder
        """
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            return self.model(inps).logits
