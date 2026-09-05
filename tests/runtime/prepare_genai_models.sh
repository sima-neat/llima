#!/usr/bin/env bash
set -euo pipefail

# Keep the shared text/VLM defaults aligned with Core's GenAI runtime tests.
# The layered Whisper encoder and reasoning fixtures provide LLiMa-specific coverage.
DEFAULT_LLIMA_MODELS_PATH="/media/nvme/llima/models"
DEFAULT_TEXT_MODEL="Qwen2.5-0.5B-Instruct-Autoround-a16w4"
DEFAULT_TEXT_REVISION="0b47c65d41d5f457458746796966ca031fd7d608"
DEFAULT_VLM_MODEL="LFM2.5-VL-450M-Autoround-a16w4"
DEFAULT_VLM_REVISION="228bfc6d0a4b3f354367a012f6fe7bec00c4625e"
DEFAULT_ASR_MODEL="whisper-small-a16w8-layered-encoder"
DEFAULT_ASR_REPO="florianvoss/whisper-small-a16w8-layered-encoder"
DEFAULT_ASR_REVISION="c0a34f15eaeee13fc7d80cd545c3fb828dc5010f"
DEFAULT_REASONING_QWEN_MODEL="Qwen3-0.6B-Autoround-a16w4"
DEFAULT_REASONING_QWEN_REPO="simaai/Qwen3-0.6B-Autoround-a16w4"
DEFAULT_REASONING_QWEN_REVISION="bfe4a547a56d94fb4474cbc7e8fcbc1e4ee276ac"
DEFAULT_REASONING_GEMMA_MODEL="Gemma-4-E2B-it-TextOnly-GPTQ-a16w4"
DEFAULT_REASONING_GEMMA_REPO="simaai/Gemma-4-E2B-it-TextOnly-GPTQ-a16w4"
DEFAULT_REASONING_GEMMA_REVISION="1c72307f46ad055a641420310b36a061715b0508"

validate_model_name() {
  local env_name="$1"
  local model_name="$2"

  if [[ -z "${model_name}" ||
        "${model_name}" = /* ||
        "${model_name}" == *"/"* ||
        "${model_name}" == *".."* ]]; then
    echo "ERROR: ${env_name} must be a model directory name under LLIMA_MODELS_PATH: ${model_name}" >&2
    exit 1
  fi
}

download_model() {
  local label="$1"
  local repo_id="$2"
  local model_name="$3"
  local expected_config="$4"
  local revision="${5:-}"
  local target_dir="${LLIMA_MODELS_PATH}/${model_name}"

  mkdir -p "${target_dir}"
  echo "[runtime-models] ${label}: synchronizing ${repo_id} to ${target_dir}"
  if [[ -n "${revision}" ]]; then
    hf download "${repo_id}" --revision "${revision}" --local-dir "${target_dir}"
  else
    hf download "${repo_id}" --local-dir "${target_dir}"
  fi

  if [[ ! -f "${target_dir}/${expected_config}" ]]; then
    echo "ERROR: downloaded ${label} model is missing ${expected_config}: ${target_dir}" >&2
    exit 1
  fi
}

if [[ "$#" -ne 0 ]]; then
  echo "Usage: $(basename "$0")" >&2
  exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI is required to prepare runtime model fixtures." >&2
  exit 1
fi

LLIMA_MODELS_PATH="${LLIMA_MODELS_PATH:-${DEFAULT_LLIMA_MODELS_PATH}}"
SIMA_TEST_LLIMA_TEXT_MODEL="${SIMA_TEST_LLIMA_TEXT_MODEL:-${DEFAULT_TEXT_MODEL}}"
SIMA_TEST_LLIMA_VLM_MODEL="${SIMA_TEST_LLIMA_VLM_MODEL:-${DEFAULT_VLM_MODEL}}"
SIMA_TEST_LLIMA_ASR_MODEL="${SIMA_TEST_LLIMA_ASR_MODEL:-${DEFAULT_ASR_MODEL}}"
SIMA_TEST_LLIMA_ASR_REPO="${SIMA_TEST_LLIMA_ASR_REPO:-${DEFAULT_ASR_REPO}}"
SIMA_TEST_LLIMA_REASONING_QWEN_MODEL="${SIMA_TEST_LLIMA_REASONING_QWEN_MODEL:-${DEFAULT_REASONING_QWEN_MODEL}}"
SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL="${SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL:-${DEFAULT_REASONING_GEMMA_MODEL}}"

validate_model_name "SIMA_TEST_LLIMA_TEXT_MODEL" "${SIMA_TEST_LLIMA_TEXT_MODEL}"
validate_model_name "SIMA_TEST_LLIMA_VLM_MODEL" "${SIMA_TEST_LLIMA_VLM_MODEL}"
validate_model_name "SIMA_TEST_LLIMA_ASR_MODEL" "${SIMA_TEST_LLIMA_ASR_MODEL}"
validate_model_name \
  "SIMA_TEST_LLIMA_REASONING_QWEN_MODEL" "${SIMA_TEST_LLIMA_REASONING_QWEN_MODEL}"
validate_model_name \
  "SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL" "${SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL}"

mkdir -p "${LLIMA_MODELS_PATH}"
download_model \
  "text" "simaai/${SIMA_TEST_LLIMA_TEXT_MODEL}" "${SIMA_TEST_LLIMA_TEXT_MODEL}" \
  "devkit/vlm_config.json" "${DEFAULT_TEXT_REVISION}"
download_model \
  "vlm" "simaai/${SIMA_TEST_LLIMA_VLM_MODEL}" "${SIMA_TEST_LLIMA_VLM_MODEL}" \
  "devkit/vlm_config.json" "${DEFAULT_VLM_REVISION}"
download_model \
  "asr" "${SIMA_TEST_LLIMA_ASR_REPO}" "${SIMA_TEST_LLIMA_ASR_MODEL}" \
  "devkit/whisper_config.json" "${DEFAULT_ASR_REVISION}"
download_model \
  "Qwen reasoning" "${DEFAULT_REASONING_QWEN_REPO}" \
  "${SIMA_TEST_LLIMA_REASONING_QWEN_MODEL}" "devkit/vlm_config.json" \
  "${DEFAULT_REASONING_QWEN_REVISION}"
download_model \
  "Gemma reasoning" "${DEFAULT_REASONING_GEMMA_REPO}" \
  "${SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL}" "devkit/vlm_config.json" \
  "${DEFAULT_REASONING_GEMMA_REVISION}"

echo "[runtime-models] ready under ${LLIMA_MODELS_PATH}"
