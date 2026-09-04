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
