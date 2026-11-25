# LLiMa コマンドラインインターフェース

`llima` CLIをModalixで使用して、事前にコンパイルされたモデルを管理し、簡単なランタイムテストを実行します。モデルがロードされ、プロンプトを受け入れ、Neat Frameworkの直接APIまたはNeat GenAIサーバーエンドポイントに統合する前に、出力が生成されるかどうかを確認するのに役立ちます。

## モデルマネージャー

LLiMa は、`llima` CLI を通じてモデルマネージャーを提供します。これにより、コマンドラインから直接、事前にコンパイルされたモデルを検索、ダウンロード、一覧表示、削除、実行できます。モデルはデフォルトでは `/media/nvme/llima/models` に保存されます。別のモデルディレクトリを使用する場合は、`LLIMA_MODELS_PATH` を設定してください。

利用可能なモデルを参照してください。

``` console
modalix:~$ llima search
modalix:~$ llima search qwen
```

名前でモデルをダウンロードします。この際、`simaai/` の組織プレフィックスは使用しません。

``` console
modalix:~$ llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

モデルのアーティファクトを並行してダウンロードし、最もサイズの大きいアーティファクトから順にダウンロードします。一時的なHTTPエラーは自動的に再試行されます。同じモデルのダウンロードは逐次的に行われ、ダウンロードをキャンセルしても、すでにダウンロードおよび検証済みのすべてのアーティファクトは保持されます。

ローカルにインストールされているモデルのリストを表示し、削除します。

``` console
modalix:~$ llima list
modalix:~$ llima rm Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## LLiMa を実行中

`llima run` を、Modalix 上で初期モデルの検証を行うためのシンプルなランタイムとして使用してください。

CLIモードでは、チャット履歴がデフォルトで有効になっています。各プロンプトと応答は、`clear history` を使用して履歴をクリアするまで、次のターンで使用するためのコンテキストとして保持されます。プロンプトとともに送信された画像も、その履歴の一部として保持されます。`clear history` は送信済みのプロンプト、応答、およびすべての画像を削除します。設定済みのシステムプロンプトは、`clear system` で削除するか `set system` で置き換えるまで有効なままです。

``` console
modalix:~$ llima run <model> [options]
```

| 議論 | 説明 |
|----|----|
| `model` | モデルIDまたはパス（例：`Qwen3-VL-8B-Instruct-a16w4`）。 |
| `--stt_model_path` | 音声認識モデルで使用するELFファイルのパス（オプション）。 |

利用可能なすべてのオプションについては、`llima run -h` を実行してください。

デフォルト設定（`auto`）では、未加工の `.bin` ファイルがモデルの `devkit/` ディレクトリにある場合に
埋め込みテーブルがオフロードされます。ファイルがなければテーブルは DRAM に保持されます。
NVMe 以外のストレージでは、読み取り速度の低下が性能に影響する可能性があるため、警告が表示されます。
テーブルを DRAM に保持するには `SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=off llima run <model>` を実行します。
オフロードを必須にするには `SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=on llima run <model>` を実行します。
この場合、未加工のファイルが存在しないか無効であれば、コマンドは失敗します。

**例**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## インタラクティブなコマンド

CLIモードで`llima run`が起動したら、プロンプトで次のコマンドを使用してください。

| コマンド | 説明 |
|----|----|
| `add image <file>` | 現在のプロンプトのコンテキストに画像を追加します。 |
| `set system <prompt>` | システムプロンプトを設定します。 |
| `clear system` | システムプロンプト、チャット履歴、画像をすべて削除します。 |
| `clear history` | システムプロンプトを保持したまま、送信済みのプロンプト、応答、およびすべての画像を削除します。 |
| `print history` | チャットの履歴を印刷します。 |
| `set audio <file>` | 文字起こしする音声ファイルをクエリとして設定します。 |
| `set language <lang>` | 文字起こしに使用する言語を設定します。 |
| `set lora <name>` | `npy_files` フォルダにある LoRA の重みを使用してください。 |
| `unset lora` | LoRAモデルをベースラインモデルに戻します。 |
| `enable-thinking` | 思考モードを有効にし、チャット履歴をクリアします。 |
| `disable-thinking` | 思考モードをオフにし、チャット履歴をクリアします。 |
| `quit` | やめる。 |
| `help` | 利用可能なコマンドを表示します。 |


## Neat を使ってアプリケーションを構築しましょう。

`llima run` を使用してモデルを検証した後、[、GenAI モデル ](/develop-apps/development-workflow/genai-model/) を、一般的な API エンドポイントを通じて提供するか、C++ または Python アプリケーションから直接使用してください。

## 独立したプロンプトコンテキストの再利用

C++ アプリケーションは `ModelOptions.max_kv_cache_slots`（既定値は 1）で上限付きプールを読み込み、`GenerationOptions.cache_id` でコンテキストを選択できます。

```cpp
#include <sima_lmm/vision_language_model.hpp>

simaai::llima::VisionLanguageModel model(model_dir, simaai::llima::ModelOptions{4});
auto chat = model.create_chat();
chat.add_query("Hello");
simaai::llima::GenerationOptions options;
options.cache_id = "assistant";
auto result = model.generate(chat, options);
model.remove_kv_cache("assistant");
```

毎回、完全なプロンプトまたは履歴を送信してください。ID は再利用するプレフィックス状態を選択し、アプリケーションのメッセージは保存しません。ID を省略した従来のコンテキストも 1 スロットを使用します。デバイスのキャッシュ領域は構築時に確保され、重みと実行ワークスペースは共有されます。`kv_cache_bytes_per_slot()` はスロットごとのデバイスバイト数、`kv_cache_count()` は従来のコンテキストを含む割り当て数を返します。`remove_kv_cache()` は指定した関連付けを、`clear_kv_caches()` はすべての関連付けを解除しますが、物理領域は保持します。上限で新しい ID を要求すると `KVCacheCapacityError` が発生し、自動削除はありません。ライフサイクル操作はモデルがアイドル状態のときだけ使用でき、生成中の呼び出しは拒否されます。キャンセルは選択した状態を、LoRA の変更はすべての状態を無効化します。投機的デコードのターゲットとドラフトには同じスロット上限が必要です。オプションとメトリクスの追加により値型のレイアウトが変わるため、対応する LLiMa ヘッダーとライブラリで利用側を再ビルドしてください。
