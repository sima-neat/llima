# MoLE - Modalix 言語モデル評価ツール

## 概要

MoLE (Modalix 言語モデル評価ツールは、モデルの精度とパフォーマンスを評価するためのベンチマークツールです。 LLMs が実行されています Modalix プラットフォーム。

拡張します。 [EleutherAIlm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) また、以下の2つのバックエンドをサポートしています。

- **hf** — ホスト上で HuggingFace transformers を使用して評価を実行します（ベースライン参照）。
- **modalix** — `llima benchmark-server` を使用して、Modalix ボード上で評価を実行します。

## インストール

SDK Docker コンテナの外で、ホストまたは Modalix DevKit の独立した仮想環境に MoLE をインストールしてください。DevKit での実行には、LLiMa ランタイムのインストールも必要です。ランタイムのインストール手順は [Neat Framework のインストール](/getting-started/neat-library/)を参照してください。

`sima-cli` を使用して MoLE をインストールします。

``` console
host:~$ sima-cli neat install llima/mole
```

これにより、MoLE が `~/sima-mole-venv` の仮想環境にインストールされます。

## 使用方法

まず、MoLE 仮想環境を有効にします。

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

ホストから Modalix 評価を実行する場合、`--board_ip` は必須です。Modalix DevKit では省略するとローカルのランタイムを使用します。サーバーを起動する場合は `--board_model` を指定してください。ローカルサーバーのログは `<output_dir>/server.log` に保存されます。既存のサーバーに接続し、起動や停止を行わない場合は `--no-board_start_server` を使用してください。

MoLE は `llima-benchmark accuracy` と `llima-benchmark perf` を提供します。Modalix では `<model_id>` に HuggingFace モデル ID、またはデプロイしたモデルと一致するローカルのトークナイザー／設定ディレクトリを指定できます。MoLE はこのトークナイザーでリクエストを準備します。

### 精度ベンチマーク

標準的なタスクにおけるモデルの品質を評価します。

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b modalix \
    -t <task> \
    -o <output_dir> \
    --max_num_tokens <max_num_tokens> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board>
```

| 議論 | 説明 |
|----|----|
| `model_id` | HuggingFace モデル ID（例：`meta-llama/Llama-3.2-3B-Instruct`）。`-b modalix` の場合、これはデプロイされたモデルのトークナイザー/設定と一致する必要があります。 |
| `-b` | 使用するバックエンド：`modalix`（オンボードで実行）または`hf`（ホスト上で参照ベースラインとして実行）。 |
| `-t` | **必須**。1つ以上の評価タスクが必要です。例：`hellaswag`、`triviaqa`、`piqa`、`winogrande`、`wikitext`。利用可能なすべてのタスクについては、[タスクリスト](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md)を参照してください。 |
| `-o` | ベンチマーク結果の出力ディレクトリ。 |
| `--board_ip` | リモートボードの IP アドレスまたはホスト名。ホストからの実行時は必須です。Modalix でのローカル実行時は省略します。 |
| `--board_model` | DevKit 上のコンパイル済みモデルディレクトリ。サーバーの起動時は必須です。 |
| `--max_num_tokens` | 最大コンテキスト長。コンパイル時に使用された値と等しいか、それよりも小さい値でなければなりません。 |
| `-n, --num_samples` | 評価するサンプルの数を指定します。指定しない場合は、すべてのタスクセットを実行します。 |
| `--board_ssh_user` | Modalix ボードの SSH ユーザー名。オプション。デフォルト値：`sima`。 |
| `--board_ssh_pass` | Modalix ボードの SSH パスワード。オプションです。設定することで、インタラクティブな操作なしの自動ベンチマークを有効にできます。 |

:::important
`-b modalix` を使用した精度と対数尤度によるベンチマークを行うには、デプロイされたモデルを `--return_logits` を使用してコンパイルする必要があります。このフラグはデフォルトでオフになっています。[モデルのコンパイル](compilation_genai.md) を参照してください。モデルがこのフラグなしでコンパイルされた場合、ベンチマークは次のエラーで失敗します: `model not compiled with --return_logits; accuracy/loglikelihood tasks are unsupported`.
:::

`-b modalix` モードでは、結果テーブルは Modalix バックエンドの結果としてラベル付けされ、ターゲットとなるボードの情報が含まれます。HuggingFace `model_id` は、MoLE がトークン化とタスクのメタデータに使用するため、引き続き表示されます。

参照ベースラインとして HuggingFace バックエンドを使用するには：

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b hf -t <task> -o <output_dir>
```

利用可能なすべてのオプションについて、`llima-benchmark accuracy -h` を実行してください。

### パフォーマンスのベンチマーク

さまざまな入力長に対して、Modalix ボード上で、最初のトークンが表示されるまでの時間（TTFT）と、1秒あたりのトークン数（TPS）を測定します。

``` console
(sima-mole-venv) host:~$ llima-benchmark perf <model_id> \
    -o <output_dir> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board> \
    --max_num_tokens <max_num_tokens> --max_new_tokens <max_new_tokens> \
    --input_lengths 1024 2048 3072 4096
```

| 議論 | 説明 |
|----|----|
| `model_id` | HuggingFace モデル ID（例：`meta-llama/Llama-3.2-3B-Instruct`）。Modalix のパフォーマンス評価を行う場合、これはデプロイされたモデルのトークナイザー/設定と一致する必要があります。 |
| `-o` | ベンチマーク結果の出力ディレクトリ。 |
| `--board_ip` | リモートボードの IP アドレスまたはホスト名。ホストからの実行時は必須です。Modalix でのローカル実行時は省略します。 |
| `--board_model` | DevKit 上のコンパイル済みモデルディレクトリ。サーバーの起動時は必須です。 |
| `--max_num_tokens` | 最大コンテキスト長。コンパイル時に使用された値と等しいか、それよりも小さい値でなければなりません。 |
| `--max_new_tokens` | 出力で生成するトークンの最大数。 |
| `--input_lengths` | ベンチマークに使用する、入力トークンの正確な長さをオプションで指定できます。値は一意である必要があり、各値に`--max_new_tokens`を加えた合計が、`--max_num_tokens`の範囲内に収まる必要があります。指定しない場合、MoLEは、自動的に2の累乗の範囲を生成します。 |
| `--board_ssh_user` | Modalix ボードの SSH ユーザー名。オプション項目で、デフォルト値は `sima` です。 |
| `--board_ssh_pass` | Modalix ボードの SSH パスワード。オプションです。設定することで、インタラクティブな操作なしの自動ベンチマークを有効にできます。 |

利用可能なすべてのオプションについて、`llima-benchmark perf -h` を実行してください。

入力長バケットの代わりに `--dataset bfcl-v3|ifeval|alpacaeval|gsm8k|dialogsum` でネイティブのチャット／ツールプロンプトを使用できます。`-n` を省略すると選択したデータセット全体を実行し、`-n 20` は先頭の 20 サンプルを実行します。データセットは自動的にダウンロードしてキャッシュします。BFCL は simple/multiple のケースを対象とし、ツール自体は実行しません。

`--board_model` により通常デコードか投機的デコードかが決まります。比較時は同じデータセットと設定で別々に実行し、`--board_model` と `-o` だけを変更してください。ウォームアップ 1 回を計測から除外し、毎回 KV 状態をリセットします。モデルのネイティブチャットテンプレートと EOS 停止を使用します。データセットの出力上限には最初のトークンを含み、既存のバケットでは最初のトークンと `--max_new_tokens` 回のデコードを計測します。

`-o` はタイミング、トークン数、集計指標、ドラフト寄与率、失敗、ベンチマーク設定、データセットの出典を `performance.json` に保存します。デコード TPS は TTFT を除外し、生成時間には入力準備と通信時間を含みません。これらのワークロードは精度ではなく性能を計測します。
