# MoLE - Modalix 言語モデル評価ツール

## 概要

MoLE（Modalix Language Model Evaluator）は、Modalix プラットフォームで動作する LLM と VLM の精度と性能を評価するベンチマークツールです。

テキストの精度評価には [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) を使用します。Modalix での画像とテキストの精度評価には [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) を使用し、上流のタスク、回答抽出、採点を再利用します。どちらも `llima-benchmark accuracy` を使用します。MoLE は次の 2 つのバックエンドをサポートします。

- **hf** — ホスト上で HuggingFace transformers を使用してテキストの精度を評価します（参照ベースライン）。
- **modalix** — `llima benchmark-server` を介して Modalix ボード上で評価します。

## インストール

SDK Docker コンテナの外部で、ホストまたは Modalix DevKit 上の独立した仮想環境に MoLE をインストールします。DevKit で実行する場合は、LLiMa ランタイムもインストールする必要があります。ランタイムのインストール手順は [Neat Framework のインストール](/getting-started/neat-library/)を参照してください。

`sima-cli` を使用して MoLE をインストールします。

``` console
host:~$ sima-cli neat install llima/mole
```

このコマンドは `~/sima-mole-venv` の仮想環境に MoLE をインストールします。

### 手動インストール

`sima-cli` を使用しない場合は、ダウンロードした MoLE パッケージのディレクトリで付属のインストーラーを実行します。

```bash
bash ./install_mole.sh --venv ~/sima-mole-venv
```

wheel を直接インストールする場合は、Python 3.11–3.13 の仮想環境を有効にし、wheel のファイル名をダウンロードしたバージョンに置き換えて次を実行します。

```bash
python -m pip install './sima_lmm-<version>-py3-none-any.whl[mole]'
python -m pip install --no-deps "lmms-eval==0.7.3" "tenacity>=8.3.0"
```

2 番目のコマンドは、MoLE の共通依存関係を再利用して画像ベンチマークを有効にします。

## 使用方法

まず、MoLE の仮想環境を有効にします。

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

ホストから Modalix の評価を実行する場合は `--board_ip` が必要です。Modalix DevKit では省略するとローカルランタイムを使用します。サーバーを起動する場合は `--board_model` を指定します。ローカルサーバーのログは `<output_dir>/server.log` に保存されます。既存のサーバーを起動・停止せずに接続するには `--no-board_start_server` を使用します。

MoLE は `llima-benchmark accuracy` と `llima-benchmark perf` のサブコマンドを提供します。テキストワークロードには、デプロイ済みモデルに一致する HuggingFace モデル ID またはローカルのトークナイザー/設定ディレクトリを使用します。画像ワークロードにはデプロイ済み VLM のアセットを使用し、`<model_id>` は実行のラベルになります。

### 精度ベンチマーク

#### テキストの精度

標準タスクでモデルの品質を評価します。

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b modalix \
    -t <task> \
    -o <output_dir> \
    --max_num_tokens <max_num_tokens> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board>
```

| 引数 | 説明 |
|----|----|
| `model_id` | テキスト精度評価用の HuggingFace モデル ID またはローカルのトークナイザー/設定ディレクトリ。画像精度評価では実行内容を示すラベルです。 |
| `-b` | 使用するバックエンド：`modalix`（ボード上で実行）または `hf`（参照ベースラインとしてホスト上で実行）。 |
| `-t` | **必須。** 1 つ以上の評価タスク。例：`hellaswag`、`triviaqa`、`piqa`、`winogrande`、`wikitext`。利用可能なすべてのタスクは[タスク一覧](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md)を参照してください。 |
| `-o` | ベンチマーク結果の出力ディレクトリ。 |
| `--board_ip` | リモートボードの IP またはホスト名。ホストから実行する場合は必須です。Modalix 上でローカル実行する場合は省略します。 |
| `--board_model` | DevKit 上のコンパイル済みモデルのディレクトリ。サーバー起動時に必要です。 |
| `--max_num_tokens` | 最大コンテキスト長。コンパイル時に使用した値以下である必要があります。 |
| `-n, --num_samples` | 評価するサンプル数。指定しない場合はタスクセット全体を実行します。 |
| `--board_ssh_user` | Modalix ボードの SSH ユーザー名。省略可能で、デフォルトは `sima` です。\# |
| `--board_ssh_pass` | Modalix ボードの SSH パスワード。省略可能です。非対話型の自動ベンチマークを有効にする場合に設定します。 |

:::important
ロジットまたは対数尤度を使用するタスクでは、コンパイル時に `--return_logits` が必要です。画像とテキストの評価を含む生成ベースのタスクでは不要です。[モデルのコンパイル](compilation_genai.md)を参照してください。
:::

`-b modalix` モードでは、結果の表に Modalix バックエンドの結果であることと対象ボードが表示されます。MoLE がトークン化とタスクのメタデータに使用するため、HuggingFace の `model_id` も表示されます。

HuggingFace バックエンドを参照ベースラインとして使用するには、次を実行します。

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b hf -t <task> -o <output_dir>
```

利用可能なすべてのオプションは `llima-benchmark accuracy -h` で確認できます。

#### 画像とテキストの精度

評価ツールは MoLE とともにインストールされ、デプロイ済み VLM のネイティブなチャットテンプレート、画像前処理、上流タスクの採点を使用します。次のタスクは Modalix 上で Qwen3.5-2B を使用してテスト済みです。

| タスク | 評価対象 | 検証 |
|----|----|----|
| `realworldqa` | 実世界のシーンと空間推論 | データセット全体 |
| `docvqa_val` | 文書の読み取りと OCR | スモークテスト |
| `chartqa` | グラフの読み取りと数値推論 | スモークテスト |
| `mmstar` | 視覚認識とマルチモーダル推論 | 10 サンプルのスモークテスト |
| `ai2d` | 科学的な図の理解 | 10 サンプルのスモークテスト |
| `ocrbench` | テキストの認識と理解 | 10 サンプルのスモークテスト |
| `textvqa_val` | 自然画像内のテキストに関する質問 | 10 サンプルのスモークテスト |

以下の制約を満たす他の個別の `lmms-eval` 画像入力の生成タスクも `-t` で選択できますが、すべてがテスト済みではありません。

```bash
llima-benchmark accuracy qwen3.5-2b -b modalix \
    -t realworldqa docvqa_val chartqa \
    --board_ip <devkit-address> --board_model <compiled-model-on-devkit> \
    --max_num_tokens 4096 -o results/vlm
```

DevKit では `--board_ip` を省略します。ホスト側のモデルアセットは不要です。対応するランタイムと MoLE のリリースを使用してください。

| 画像精度評価のオプション | 説明 |
|----|----|
| `--max_new_tokens` | タスクの出力トークン数の上限を上書きします。指定しない場合は上流のデフォルト値を維持します。 |
| `--enable_thinking` | モデルのネイティブな思考テンプレートを有効にします。デフォルトでは無効です。 |
| `--accuracy_manifest` | 以下のように固定したデータセットリビジョンまたは準備済みのローカルデータセットを使用します。 |

テキストと画像のタスクは、個別のタスク名を指定して別々に実行します。画像の精度評価は、貪欲デコーディングを使用する通常の VLM に対応します。音声/動画、マルチターン、視覚的 few-shot、繰り返しリクエストのタスクには対応しません。入力と出力トークン数の上限の合計が、指定したコンテキストとコンパイル済みコンテキストに収まる必要があります。入力は切り詰められません。複数の画像はタスクで定義された順序を維持します。ネイティブの EOS で生成を停止し、タスクの停止文字列で返される回答を切り詰めます。

#### 準備済みの画像データセット

データセットは自動的にダウンロードされます。リビジョンを固定した評価またはオフライン評価には `--accuracy_manifest manifest.json` を指定します。たとえば `-t chartqa` の場合：

```json
{
  "schema_version": 1,
  "tasks": {
    "chartqa": {
      "revision": "<40-character-dataset-commit>",
      "split": "test",
      "local_path": "datasets/chartqa"
    }
  }
}
```

HuggingFace の `DatasetDict.save_to_disk()` でローカルデータセットを準備し、上流の列、画像、採点用の分割を維持します。パスはマニフェストからの相対パスで、タスク名は `-t` と一致する必要があります。固定したリビジョンをダウンロードする場合は `local_path` を省略します。オフライン実行では `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` を設定します。

#### 画像精度評価の結果

出力ディレクトリには次のファイルが含まれます。

- `accuracy.json`：集計指標、サンプル数、データセットリビジョン、デプロイ先パス、ランタイムバージョン、設定。
- `accuracy-answers.jsonl`：生成した回答、画像/プロンプトの識別情報、トークン数、終了理由、エラー。
- `accuracy-samples.jsonl`：上流のサンプル単位の採点結果。

`-n` を省略するとデータセット全体を実行します。失敗すると評価を停止し、途中までの回答を保存しますが、集計指標は生成しません。出力トークン数の上限による切り詰めは集計され、対象の回答も採点されます。

### 性能ベンチマーク

Modalix ボード上で、入力長ごとの最初のトークンまでの時間（TTFT）と 1 秒あたりのトークン数（TPS）を測定します。

``` console
(sima-mole-venv) host:~$ llima-benchmark perf <model_id> \
    -o <output_dir> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board> \
    --max_num_tokens <max_num_tokens> --max_new_tokens <max_new_tokens> \
    --input_lengths 1024 2048 3072 4096
```

| 引数 | 説明 |
|----|----|
| `model_id` | テキストワークロードでは、デプロイ済みモデルに一致する HuggingFace モデル ID またはローカルのトークナイザー/設定ディレクトリ。画像ワークロードでは実行内容を示すラベルです。 |
| `-o` | ベンチマーク結果の出力ディレクトリ。 |
| `--board_ip` | リモートボードの IP またはホスト名。ホストから実行する場合は必須です。Modalix 上でローカル実行する場合は省略します。 |
| `--board_model` | DevKit 上のコンパイル済みモデルのディレクトリ。サーバー起動時に必要です。 |
| `--max_num_tokens` | 最大コンテキスト長。コンパイル時に使用した値以下である必要があります。 |
| `--max_new_tokens` | 出力で生成するトークン数の上限。 |
| `--input_lengths` | ベンチマークする正確な入力トークン長を指定するオプションです。値は重複できず、各値と `--max_new_tokens` の合計が `--max_num_tokens` 以内である必要があります。省略すると、MoLE は 2 の累乗の長さのバケットを自動生成します。 |
| `--board_ssh_user` | Modalix ボードの SSH ユーザー名。省略可能で、デフォルトは `sima` です。 |
| `--board_ssh_pass` | Modalix ボードの SSH パスワード。省略可能です。非対話型の自動ベンチマークを有効にする場合に設定します。 |

利用可能なすべてのオプションは `llima-benchmark perf -h` で確認できます。

トークン長のバケットの代わりにネイティブなチャット/ツールプロンプトを使用するには、`--dataset bfcl-v3|ifeval|alpacaeval|gsm8k|dialogsum` を指定します。`-n` を省略すると選択したデータセット全体を実行し、`-n 20` では最初の 20 サンプルを実行します。データセットは自動的にダウンロードされ、キャッシュされます。BFCL は単一/複数の呼び出しケースを対象とします。ツールは実行されません。

`--board_model` によって通常のデコーディングか投機的デコーディングかが決まります。同じデータセットと設定で、`--board_model` と `-o` だけを変更して別々に実行し、モデルを比較します。各実行ではウォームアップ 1 回を測定から除外し、リクエストごとに KV 状態をリセットします。モデルのネイティブなチャットテンプレートと EOS 停止が適用されます。データセットの出力トークン数の上限には最初のトークンが含まれます。既存のバケットでは、最初のトークンと `--max_new_tokens` 回のデコードステップを測定します。

#### 画像とテキストの性能

`--dataset realworldqa|docvqa_val|chartqa` を使用すると、精度評価と同じ上流の画像/テキストプロンプトで、通常または投機的デコーディングの VLM を測定できます。

```bash
llima-benchmark perf qwen3.5-vlm --dataset chartqa \
    --board_ip <devkit-address> --board_model <compiled-model-or-speculative-bundle> \
    --max_num_tokens 4096 --max_new_tokens 256 -o results/vlm-perf
```

`-n` を省略すると、ウォームアップ 1 回を測定から除外した後、データセット全体を 1 回実行します。`--dataset_manifest` は `--accuracy_manifest` と同じ固定リビジョン/ローカルデータセット形式を受け付けます。タスク名は `--dataset` と一致する必要があります。画像はデータセットの解像度を維持し、ネイティブランタイムの前処理を使用します。ホスト側のモデルアセットは不要です。精度評価は引き続き通常のデコーディングに限定されます。

`-o` は、リクエストごとの計測時間、トークン数、集計指標、ドラフト寄与率、失敗、ベンチマーク設定、データセットの出典を `performance.json` に保存します。デコード TPS は TTFT を除外します。画像の TTFT にはネイティブの画像前処理とビジョンエンコーディングが含まれます。テキストの TTFT は言語モデルのプリフィルから始まります。どちらもネットワーク転送とホスト側のデータセット準備を除外します。ドラフト寄与率は、採用されたドラフトトークン数を生成された全トークン数で割った値です。これらのワークロードは精度ではなく性能を測定します。
