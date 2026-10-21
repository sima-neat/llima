"""Image/text wire format and reproducibility helpers, independent of task scoring."""

import base64
import hashlib
import io
import json
import re
from pathlib import Path

import msgpack
import zmq
from PIL import Image


class UnsupportedInput(ValueError):
    """An input cannot be evaluated faithfully by this backend."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_identity(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size}


def tree_identity(root: Path) -> dict:
    return {str(path.relative_to(root)): file_identity(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def ordered_content(context: str, visuals) -> tuple[list[dict], list[dict]]:
    """Preserve upstream placeholders; without placeholders use upstream image-first ordering."""
    images = []

    def collect(value):
        if isinstance(value, Image.Image):
            if getattr(value, "n_frames", 1) != 1:
                raise UnsupportedInput("Animated/multiframe images require a separate protocol")
            images.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        else:
            raise UnsupportedInput("Only PIL images are supported; audio, video and paths are unsupported")

    collect(visuals)
    if not isinstance(context, str) or not images:
        raise UnsupportedInput("Image/text tasks require a text context and at least one image")
    encoded, identities = [], []
    for image in images:
        output = io.BytesIO()
        # Lossless RGB transport: all resizing/normalization stays in the installed runtime.
        image.convert("RGB").save(output, format="PNG")
        data = output.getvalue()
        encoded.append({"type": "image", "image": "data:image/png;base64," + base64.b64encode(data).decode()})
        identities.append({"sha256": digest(data), "width": image.width, "height": image.height,
                           "transport": "RGB PNG", "bytes": len(data)})
    matches = list(re.finditer(r"<image(?:\s+(\d+))?>", context))
    if re.search(r"<image[^>]*>", re.sub(r"<image(?:\s+\d+)?>", "", context)):
        raise UnsupportedInput("Unsupported image placeholder syntax")
    if not matches:
        return encoded + [{"type": "text", "text": context}], identities
    if len(matches) != len(images):
        raise UnsupportedInput("Image placeholder count does not match image inputs")
    numbered = [match.group(1) is not None for match in matches]
    if any(numbered) and not all(numbered):
        raise UnsupportedInput("Mixed numbered and unnumbered image placeholders")
    indices = [int(match.group(1)) - 1 if match.group(1) else i for i, match in enumerate(matches)]
    if sorted(indices) != list(range(len(images))):
        raise UnsupportedInput("Every image must occur exactly once in the prompt")
    content, cursor = [], 0
    for match, index in zip(matches, indices):
        if match.start() > cursor:
            content.append({"type": "text", "text": context[cursor:match.start()]})
        content.append(encoded[index])
        cursor = match.end()
    if cursor < len(context):
        content.append({"type": "text", "text": context[cursor:]})
    return content, [identities[index] for index in indices]


def generation_settings(kwargs: dict, override: int | None = None) -> dict:
    allowed = {"max_new_tokens", "temperature", "do_sample", "top_p", "num_beams", "until"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise UnsupportedInput(f"Unsupported generation settings: {sorted(unknown)}")
    if (kwargs.get("temperature", 0) != 0 or kwargs.get("do_sample", False)
        or kwargs.get("top_p", 1) != 1 or kwargs.get("num_beams", 1) != 1):
        raise UnsupportedInput("Modalix image accuracy currently supports greedy decoding only")
    budget = override if override is not None else kwargs.get("max_new_tokens", 256)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 65535:
        raise UnsupportedInput("Output token budget must be an integer in [1, 65535]")
    stops = kwargs.get("until") or []
    stops = [stops] if isinstance(stops, str) else stops
    if not isinstance(stops, (list, tuple)) or any(not isinstance(stop, str) or not stop for stop in stops):
        raise UnsupportedInput("Stop sequences must be nonempty strings")
    return {"max_new_tokens": budget, "temperature": 0, "do_sample": False,
            "top_p": 1, "num_beams": 1, "until": list(stops)}


class MultimodalClient:
    """The same transport is used by remote hosts and DevKit-local evaluators."""

    def __init__(self, board, max_context: int, enable_thinking: bool = False):
        if not 1 <= max_context <= 65535:
            raise UnsupportedInput("Context limit must fit the runtime's uint16 token budget")
        self.max_context = max_context
        self.enable_thinking = enable_thinking
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.rcvtimeo = self.socket.sndtimeo = 10 * 60 * 1000
        self.socket.curve_publickey, self.socket.curve_secretkey = zmq.curve_keypair()
        self.socket.curve_serverkey = b"xA?#1AE663fk][M)Dd9}x?#nV2Iy!p3^&>kerN.>"
        self.socket.connect(board.tcp_uri)

    def request(self, request_type: str, payload: bytes = b"", **settings):
        self.socket.send_multipart([
            msgpack.packb({"type": request_type, **settings}, use_bin_type=True), payload,
        ])
        frames = self.socket.recv_multipart()
        if len(frames) != 2:
            raise RuntimeError("Malformed benchmark response: expected two frames")
        metadata = msgpack.unpackb(frames[0], raw=False)
        if metadata.get("error"):
            if metadata.get("error_type") == "unsupported":
                raise UnsupportedInput(metadata["error"])
            raise RuntimeError(metadata["error"])
        if (metadata.get("tensor_dtype") != "utf8" or metadata.get("result_type") != request_type
            or metadata.get("tensor_shape") != [len(frames[1])]):
            raise RuntimeError("Benchmark server does not support the image/text accuracy protocol")
        return frames[1].decode("utf-8"), metadata

    def info(self) -> dict:
        result, _ = self.request("multimodal_info")
        info = json.loads(result)
        if info.get("protocol_version") != 1 or info.get("decoding") != "greedy":
            raise UnsupportedInput("Unsupported multimodal benchmark server version or decoding mode")
        if self.max_context > info["max_num_tokens"]:
            raise UnsupportedInput("Requested context exceeds compiled model context")
        return info

    def generate(self, messages: list[dict], settings: dict) -> tuple[str, dict]:
        answer, metadata = self.request(
            "generate_multimodal", json.dumps({"messages": messages}, ensure_ascii=False).encode(),
            max_num_tokens=self.max_context, max_new_tokens=settings["max_new_tokens"],
            enable_thinking=self.enable_thinking,
        )
        if metadata.get("finish_reason") not in ("stop", "length"):
            raise RuntimeError("Runtime did not report a supported generation finish reason")
        inputs, outputs = metadata.get("input_tokens"), metadata.get("generated_tokens")
        if (type(inputs) is not int or type(outputs) is not int or inputs < 1 or outputs < 1
            or outputs > settings["max_new_tokens"] or inputs + settings["max_new_tokens"] > self.max_context
            or not re.fullmatch(r"[0-9a-f]{64}", metadata.get("prompt_sha256") or "")):
            raise RuntimeError("Runtime returned invalid token counts, prompt identity or output budget")
        metadata.setdefault("raw_answer", answer)
        # Match harness generate_until semantics without flattening multi-token stops to token IDs.
        positions = [answer.find(stop) for stop in settings["until"] if stop in answer]
        if positions:
            answer = answer[:min(positions)]
            metadata["answer_stop_sequence_matched"] = True
        return answer, metadata

    def close(self):
        self.socket.close(linger=0)
        self.context.term()
