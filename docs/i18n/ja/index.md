# GenAIとLLiMa

LLiMaは、Model CompilerにおけるGenAIツールキットであり、LLM、VLM、およびASRモデルのコンパイル、テスト、ベンチマーク、デプロイ、および実行をModalix上で行うために使用されます。

LLiMaは、次の3つの入力形式をサポートしています。

- **Hugging Face safetensors** — 標準のLLMおよびVLMモデルのディレクトリ
- **GGUFファイル** — GGUF形式でパッケージ化されたLLMモデル
- **圧縮テンソルモデル** — 事前量子化された、GPTQ / AWQ スタイルのセーフテンソルモデル

SiMa.aiは、GenAIの事前コンパイルされたモデルを[のHugging Face、つまり](https://huggingface.co/simaai)に公開しています。適切なモデルがすでに存在する場合は、そこから始めてください。

具体的なGenAIのデモについては、[の](https://developer.sima.ai/examples)を参照してください。

## LLiMa 入手可能性

LLiMaのコンパイルツールは、デフォルトでModel Compilerにインストールされます。
LLiMaのランタイムは、Neatランタイムの一部として、Modalixにネイティブにインストールされます。
ランタイムのインストール手順については、[Neat Frameworkのインストール](/getting-started/neat-library/)を参照してください。

## 貢献

LLiMa 自体の変更を行うコントリビューターは、リポジトリの構造、開発環境、テスト段階、モデルへの入力に関するポリシー、およびプルリクエストの要件について、[ LLiMa コントリビューターガイド ](contributing.md)に従う必要があります。

## 対応モデル

以下の表に、サポートされているモデルのアーキテクチャとその機能を示します。

| モデルのアーキテクチャ | 種類 | 対応サイズ |
|----|----|----|
| [Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family) | LLM | [7b](https://huggingface.co/simaai/Llama-2-7b-chat-hf-a16w4) |
| [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31) | LLM | [8b](https://huggingface.co/simaai/Llama-3.1-8B-Instruct-a16w4) |
| [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32) | LLM | [1b](https://huggingface.co/simaai/Llama-3.2-1B-Instruct-GPTQ-a16w4), [3b](https://huggingface.co/simaai/Llama-3.2-3B-Instruct-a16w4) |
| [Gemma 1](https://huggingface.co/collections/google/gemma-release) | LLM | 2B、7B |
| [Gemma 2](https://huggingface.co/collections/google/gemma-2-release) | LLM | 2B、9B |
| [Gemma 3](https://huggingface.co/collections/google/gemma-3-release) | LLM | [1b](https://huggingface.co/simaai/gemma-3-1b-it-a16w4), [4b](https://huggingface.co/simaai/gemma-3-4b-it-a16w4) |
| [ Phi 3.5 mini ](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | LLM | [3.8b](https://huggingface.co/simaai/Phi-3.5-mini-instruct-a16w4) |
| [Qwen 2.5](https://huggingface.co/collections/Qwen/qwen25) | LLM | [0.5b](https://huggingface.co/simaai/Qwen2.5-0.5B-Instruct-GPTQ-a16w4)、[1.5b](https://huggingface.co/simaai/Qwen2.5-1.5B-Instruct-GPTQ-a16w4)、[3b](https://huggingface.co/simaai/Qwen2.5-3B-Instruct-GPTQ-a16w4)、[7b](https://huggingface.co/simaai/Qwen2.5-7B-Instruct-GPTQ-a16w4) |
| [Qwen 3](https://huggingface.co/collections/Qwen/qwen3) | LLM | [0.6b](https://huggingface.co/simaai/Qwen3-0.6B-GPTQ-a16w4)、[1.7b](https://huggingface.co/simaai/Qwen3-1.7B-GPTQ-a16w4)、[4b](https://huggingface.co/simaai/Qwen3-4B-Instruct-2507-GPTQ-a16w4)、[8b](https://huggingface.co/simaai/Qwen3-8B-GPTQ-a16w4) |
| [Mistral 1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | LLM | [7b](https://huggingface.co/simaai/Mistral-7B-Instruct-v0.3-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2) | LLM | [350m](https://huggingface.co/simaai/LFM2-350M-a16w4)、[1.2b](https://huggingface.co/simaai/LFM2-1.2B-a16w4)、[2.6b](https://huggingface.co/simaai/LFM2-2.6B-a16w4) |
| [Llava 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | VLM | [7b](https://huggingface.co/simaai/llava-1.5-7b-hf-a16w4) |
| [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) | VLM | [3b](https://huggingface.co/simaai/paligemma-3b-pt-224-a16w8) |
| [Gemma 3](https://huggingface.co/simaai/gemma3-siglip448-a16w4) | VLM | [4b](https://huggingface.co/simaai/gemma3-siglip448-a16w4) |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4) | VLM | [E2B](https://huggingface.co/simaai/gemma4-E2B-it), [E4B](https://huggingface.co/simaai/gemma4-E4B-it) |
| [ Qwen 2.5 VL ](https://huggingface.co/collections/Qwen/qwen25-vl) | VLM | [3b](https://huggingface.co/simaai/Qwen2.5-VL-3B-Instruct-GPTQ-a16w4), [7b](https://huggingface.co/simaai/Qwen2.5-VL-7B-Instruct-GPTQ-a16w4) |
| [ Qwen 3 VL ](https://huggingface.co/collections/Qwen/qwen3-vl) | VLM | [2b](https://huggingface.co/simaai/Qwen3-VL-2B-Instruct-GPTQ-a16w4), [4b](https://huggingface.co/simaai/Qwen3-VL-4B-Instruct-GPTQ-a16w4), [8b](https://huggingface.co/simaai/Qwen3-VL-8B-Instruct-GPTQ-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2-vl) | VLM | [450m](https://huggingface.co/simaai/LFM2-VL-450M-a16w4)、[1.6b](https://huggingface.co/simaai/LFM2-VL-1.6B-a16w4)、[3b](https://huggingface.co/simaai/LFM2-VL-3B-a16w4) |
| [Whisper](https://huggingface.co/openai/whisper-small) | ASR | [small](https://huggingface.co/simaai/whisper-small-a16w8) |

## 制限事項

| 制限の種類 | 説明 |
|----|----|
| モデルのアーキテクチャ | 上記に記載されているアーキテクチャに基づくモデルのみがサポートされます。 |
| モデルのパラメータ | パラメータ数が100億未満のモデルのみがサポートされます。 |
| HFモデル | モデルは、`config.json`、safetensor形式の重み、およびアーキテクチャに必要なトークナイザーとプロセッサーのファイルを含むローカルのHugging Faceディレクトリとして利用可能でなければなりません。`generation_config.json`はオプションです。 |
| GGUF モデル | GGUF形式は、LLMでのみサポートされます。VLMは、Hugging Faceのsafetensors形式からコンパイルする必要があります。Hugging Faceのsafetensorコンパイルと比較して、パフォーマンスが低下する可能性があることに注意してください。 |
| 圧縮テンソルモデル | サポートされているLLMおよびVLMは、llm-compressorを使用して作成された、事前に量子化されたsafetensorモデル（GPTQ/AWQ）を使用できます。モデルは、対称量子化とサポートされているcompressed-tensorsレイアウトを使用する必要があります。これらのモデルは、高いパフォーマンスを維持しながら、標準のINT4量子化よりも優れた精度を達成できます。 |
| Gemma3 VLM | 改良されたSigLip 448ビジョンエンコーダーを搭載。 |
| LLAMA 3.2 ビジョン | 画像認識モデルはサポートされていません。 |
