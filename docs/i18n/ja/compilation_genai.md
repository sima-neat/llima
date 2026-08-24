# モデルのコンパイル

## 概要

**Model Compiler** は、LLiMa コマンドラインツール `llima-compile` を提供し、
Hugging Face の safetensors ファイル、GGUF ファイル、または事前に量子化された
compressed-tensors モデル（GPTQ / AutoRound）からモデルをコンパイルします。

``` console
llima-compile [options] <model_path>
```

### モデルへの入力形式

LLiMa は、3つのモデル入力パスを受け入れます。チェックポイントの利用可能性、必要な精度、およびモデルが LLM であるか VLM であるかに基づいて、いずれかを選択してください。

| 入力 | 説明 | 使用するタイミング |
| --- | --- | --- |
| オリジナル Hugging Face safetensors | FP/BF16形式のチェックポイントで LLiMa コンパイル時に量子化されます。 | 完全に量子化された状態で一致するものが存在しないか、元の重みが必要になります。 |
| 事前に量子化された Hugging Face safetensors (GPTQ/AutoRound) | 量子化された重みを再利用するチェックポイントであり、LLiMaによって使用されます。 | コレクションに完全に一致するデータが存在する場合に、この方法が推奨されます。 |
| GGUF | 既存の量子化されたLLMチェックポイント。 | 便利な LLM フォールバック；サポートされていません VLMs. |

入力形式だけでは互換性が確立されるわけではありません。モデルのアーキテクチャ、サイズ、トークナイザー、およびマルチモーダルコンポーネントもサポートされている必要があります。

### まず、事前に量子化されたSiMa.aiモデルから始めます。

:::tip 推奨される入力
オリジナルのHugging FaceまたはGGUFのモデルをダウンロードする前に、[SiMa.ai Pre-Quantized Models collection](https://huggingface.co/collections/simaai/pre-quantized-models)を確認してください。
利用可能な場合は、要求されたアーキテクチャ、パラメータサイズ、バリアント、およびモダリティに対して、完全に一致するものを使用してください。
:::

コレクションチェックポイントはモデル固有のものであり、LLiMa compressed-tensors アーティファクトであり、これらは直接 `llima-compile` に渡すことができます。これらを使用することで、追加の浮動小数点から量子化へのコンパイラ段階を回避し、その精度とレイアウトを理解するために必要な量子化の情報を保持できます。これらはコンパイルされた Modalix モデルではなく、コンパイラの入力です。

既存のサポートされているモデルに対してカスタムのファインチューニングを行う場合、正確に一致するコレクションリポジトリは、そのモデル固有の `quantize.py`、`recipe.yaml`、および `versions.txt` を提供する場合があります。そのリポジトリのモデルカードを読み、ドキュメント化されたスクリプトを使用してください。単に類似したモデルからレシピを再利用しないでください。

``` console
hf download simaai/<model-repository> \
    --revision <immutable-revision> \
    --local-dir <prequantized-model-directory>
llima-compile <prequantized-model-directory> -o <output-directory>
```

### モデルについて説明してください。LLiMaのスキルを持つエージェントがそれをまとめることになります。

SiMa.aiとLLiMaは、Neat開発環境（Neat SDK）に含まれるスキルを通じて、エージェントによるモデルのコンパイルをすぐに実行できます。これらのスキルにより、コーディングエージェントは、LLMとVLMの互換性を評価し、利用可能な場合は正確な事前量子化された入力を選択し、インストールされたLLiMa CLIを使用し、Modalixのデプロイと検証のワークフローに従うことができます。

推奨されるエージェントによる方法は、モデルをコンパイルし、アクセス可能なModalix DevKitにデプロイし、結果と診断を検査し、コンパイルを改良することができます。従来のCLIコンパイルは、同じツールを通じて直接制御するための並行パスとして残ります。どちらの方法でも、標準的で検査可能なLLiMaアーティファクトが生成されるため、選択されたモデル、コマンド、オプション、および出力を確認したり、要件の変化に応じて2つのワークフローを切り替えたりすることができます。エージェントによるコンパイルを有効にするには、[Set up the Neat SDK](https://developer.sima.ai/software/getting-started/dev-environment/)を参照してください。

自然言語で完全なワークフローを要求してください。例：

``` text
Compile <model ID or local path> with LLiMa, deploy it to my Modalix at
<user@host>, and smoke-test it. Prefer an exact SiMa.ai pre-quantized
checkpoint when available.
```

エージェントは、モデルとレシピの出所を記録し、インストールされたバージョンのCLI（コマンドラインインターフェース）の仕様に従い、サポートされていないモデルの範囲や利用できないハードウェア検証があった場合には、別のモデルや形式に静かに置き換えるのではなく、その旨を報告します。

### コンパイルの出力

デフォルトの完全なパイプラインでは、以下のディレクトリ構造が生成されます。

``` text
output_directory/
└── sima_files/                # Compiled model files
    ├── devkit/                # Runtime configuration and model data
    │   ├── tokenizer.json
    │   ├── vlm_config.json
    │   └── ...
    ├── mpk/                   # MPK archives with compiled binaries
    │   ├── layer_0.tar.gz
    │   └── ...
    ├── npy_files/             # LoRA adapter weights (only when compiled with LoRA)
    │   ├── <adapter_name>/
    │   │   └── *.npy
    │   └── ...
    └── ...
```

## コマンドライン引数

`llima-compile`ツールは、コンパイルプロセスをカスタマイズするためのさまざまな引数を受け付けます。以下の表に、利用可能なオプションについて説明します。

| 議論 | 説明 |
|----|----|
| `model_path` | モデルのパスを入力してください（HuggingFaceディレクトリ、GGUFファイル、または事前に量子化された圧縮テンソルディレクトリ）。 |
| `-o, --output` | コンパイルされたファイルの出力先ディレクトリ。デフォルトでは、モデル名が使用されます。 |
| `-c, --configuration_file` | Python スクリプトを使用して、レイヤーごとに精度を設定します（例：混合精度の場合）。 |
| `--max_num_tokens` | 最大コンテキスト長。1024の倍数である必要があります。デフォルト値：4096。 |
| `--resume` | 既存のファイルをスキップすることで、中断されたビルドを再開します。 |
| `-j, --jobs` | 並列コンパイルするジョブの数。デフォルト：物理CPUコアの数。 |
| `--log_level` | ログレベル（DEBUG、INFO、WARNING、ERROR）。デフォルト：WARNING。 |
| `--input_height` | 入力画像の高さ（ピクセル単位）。`--input_width` とともに指定する必要があります。Qwen 2 VL、Qwen 3 VL、および Gemma 4 の場合は必須ですが、SigLIP2 モデルの構成済みのサイズを上書きする場合はオプションです。 |
| `--input_width` | 入力画像の幅をピクセル単位で指定します。`--input_height` とともに必ず指定してください。Qwen 2 VL、Qwen 3 VL、および Gemma 4 のモデルでは必須です。SigLIP2 モデルのデフォルトサイズを上書きする場合はオプションです。 |
| `--system_prompt` | CLIモードおよびモデルのウォームアップのために保存するシステムプロンプト。 |
| `--system_prompt_file` | システムプロンプトが記述されたテキストファイルへのパス。 |
| `--chat_template` | コンパイルされたモデルに保存するチャットテンプレート文字列です。システムプロンプトとチャットテンプレートファイルのオプションとは、どちらか一方のみを指定できます。 |
| `--chat_template_file` | チャットテンプレートが格納されているファイルへのパス。システムプロンプトオプションとは排他的であり、`--chat_template` とはどちらか一方のみを指定できます。 |

:::note
ほとんどのモデルは、最大8192トークンまでのコンテキスト長をサポートしています。8Kのコンテキスト長を有効にするには、`--max_num_tokens 8192`を使用してください。
:::

| 高度な議論 | 説明 |
|----|----|
| `--language_group_size` | プリフィル中に並列でトークンを処理する際のバッチサイズ。より大きな値（例：256）を使用すると、大規模な入力プロンプトに対するTTFT（トークン生成までの時間）を改善できますが、小規模な入力プロンプトに対するTTFTを低下させる可能性があります。デフォルト値：128。 |
| `--future_token_mask_size` | トークン位置を跨いでコンパイル済みのモデルを再利用するためのマスクサイズ。値を大きくすると、コンパイルされたバイナリファイルの数が減りますが、1秒あたりのトークン数（TPS）が低下する可能性があります。デフォルト値：128。 |
| `--enable_filter_sharing` | グループモデルと単一モデル間でフィルターの共有を有効にすることで、DRAMの使用量を削減できます。ただし、その代償として、TTFT（最初のトークンが生成されるまでの時間）が増加し、TPS（1秒あたりのトークン数）が低下します。これは、両方のモデルタイプが同じ精度を使用する場合にのみ有効であり、LoRA を使用してコンパイルする場合には必須です。 |
| `--no-quantize_embeddings` | サポートされているLLMおよびVLMでは、デフォルトで有効になっている埋め込みテーブルの量子化を無効にします。 |
| `--no-quantize_kv_cache` | デフォルトで有効になっている KV キャッシュの量子化を無効にします。 |
| `--return_logits` | 最終層の出力におけるロジットを返します（モデル評価器に必要なもの）。 |
| `--draft_model_path` | 推測によるデコードに使用する EAGLE3 のドラフト モデルへのパス。 |
| `--lora_name` | ベースモデルと同時にコンパイルされている LoRA アダプターの名前。 |
| `--lora_path` | ベースモデルと組み合わせてコンパイルするために、LoRAアダプターディレクトリへのパスを指定してください。 |
| `--compile_lora`, `--no-compile_lora` | LoRA のパスが指定された場合に、アダプターの重みをコンパイルするかどうかを設定します。デフォルトでは有効になっています。 |

## システムプロンプト

コンパイルされたモデル構成にシステムプロンプトを保存するには、`--system_prompt`または`--system_prompt_file`を使用してください。これらの引数は、互いに排他的です。

``` console
sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
    --system_prompt "You are a concise technical assistant." \
    -o Llama-3.2-3B-Instruct_out
```

CLIモードでは、これがデフォルトのシステムプロンプトになります。セッション中に、`set system <prompt>` で置き換えたり、`clear system` で削除したりできます。

Webモード/GenAIServerでは、保存されたプロンプトがモデルのウォームアップ時に使用され、最初のリクエストに対してキャッシュできます。これはAPIリクエストに自動的に追加されることはありません。クライアントは、最初のリクエストとそれに続くすべてのリクエストにおいて、システムプロンプトを`messages`配列に含める必要があります。

## 設定ファイル

設定ファイルは、各コンパイラユニットに対してコンパイルをカスタマイズし、混合精度および選択的コンパイルを可能にします。

LLM 推論は、2つの明確なフェーズで構成され、コンパイラはそれぞれのフェーズに対して最適化されたモデルを生成します。

- **事前処理（グループモデル）**: 入力プロンプトを、`language_group_size`（例：一度に128トークン）のバッチ単位で処理します。この段階で、TTFT（最初のトークンが生成されるまでの時間）が決定され、スループットが最適化されます。
- **デコード（単一トークンモデル）**: 出力トークンを1つずつ、自己回帰的に生成します。この段階で、1秒あたりのトークン数（TPS）が決まり、低遅延での生成に最適化されます。

これらのフェーズはそれぞれ異なるパフォーマンス特性を持つため、それぞれのフェーズに対して異なる量子化戦略を適用できます。 `is_group` 設定関数内のフラグ。

**入力パラメータ** `get_layer_configuration` この関数は、各コンパイラユニットに対して呼び出され、
以下の引数を受け取ります。

- `model_properties`：`{"num_hidden_layers": int}` を含む辞書。

- `layer`：次の要素を含む辞書：
  - `"part"`：`"PRE"`、`"CACHE"`、`"POST"`などの論理コンポーネント。
    `"VISION"`、`"DRAFT_FC"`、または`"PER_LAYER"`
  - `"is_group"`: 複数のトークン/グループのバリアントの場合は`True`、それ以外の場合は`False`
  - `"index"`：そのコンパイラユニットのインデックス。`"PRE"`および`"POST"`の場合
    通常は、トランスフォーマー層に対応します。`"CACHE"` の場合は、トランスフォーマー層ではなく、キャッシュまたはトークン位置のバリアントを識別します。

**戻り値**

この関数は、以下のキーを持つ辞書を返します。

- `"precision"`: 量子化レベル（オプション、デフォルト：`"BF16"`）
  - `"BF16"`: 最高精度 - 最高品質、最大サイズ、最も遅い
  - `"A_BF16_W_INT8"`：中程度の量子化 - 良好な品質、適度なサイズ。
  - `"A_BF16_W_INT4"`: 量子化レベルが高い - 許容できる品質、最も小さいサイズ、最も高速。

- `"compile"`: このレイヤーのコンパイルをスキップするように設定します（オプション、デフォルト：`True`）。設定を`False`にすると、コンパイルがスキップされます。

- `"lora"`：このレイヤーのLoRAモード（オプション、デフォルト：`"LORA_DISABLED"`）
  - `"LORA_DISABLED"`: このレイヤーではLoRAはサポートされていません。これは、設定ファイルが提供されていない場合のデフォルト設定であり、その結果、アダプターのオーバーヘッドがない標準的なモデルになります。
  - `"LORA_BRANCH"`：ベースモデルとともに、重みをゼロにした複数のLoRAブランチを並行してコンパイルします。アダプターの重みは、ランタイム時に`.npy`ファイルから読み込まれるため、モデルを再起動することなく、アダプターを動的に切り替えることができます。アダプターを必要に応じて切り替えたい場合は、このモードを使用してください。
  - `"LORA_MERGED"`: LoRAの重みを、ランタイム時にベースモデルの重みにマージします。アダプターはセッション中に常に有効になり、切り替えたり削除したりすることはできません。アダプターを常に適用し、動的な切り替えが必要ない場合に、このモードを使用してください。

:::note
**推奨される方法:** プリフィル中に品質を維持するために、グループレイヤーには INT8 (`A_BF16_W_INT8`) を、高速な生成のために、単一トークンレイヤーには INT4 (`A_BF16_W_INT4`) を、そして、画像認識の品質を維持するために、ビジョンエンコーダーには BF16 を使用します。 ほとんどのモデルにおいて、この構成は、モデルの精度、処理能力、およびメモリ使用量の最適なバランスを提供します。
:::

## 例

**例1：シンプルなLLMのコンパイル**

Hugging FaceからダウンロードしたLlamaモデルを、デフォルト設定でコンパイルします。

``` console
sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B-Instruct
sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct -o Llama-3.2-3B-Instruct_out
```

これにより、次のことが可能になります。

- すべてのレイヤーで、デフォルトの BF16 精度を使用してください。
- コンテキストの長さを4096トークンに設定します。
- `Llama-3.2-3B-Instruct_out` ディレクトリに出力します。

**例2：カスタムのコンテキスト長でコンパイルする**

``` console
sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B-Instruct
sima-user@docker-image-id:/home/docker$ llima-compile --max_num_tokens 4096 Llama-3.2-3B-Instruct -o Llama-3.2-3B-Instruct_out
```

これにより、次のことが可能になります。

- すべてのレイヤーで、デフォルトの BF16 精度を使用してください。
- コンテキストの長さを4096トークンに設定します。
- `Llama-3.2-3B-Instruct_out` ディレクトリに出力します。

**例 3：混合精度で Gemma 3 VLM をコンパイルする**

Gemma 3 VLM のように複雑なモデルの場合、異なるレイヤーに対して異なる精度を指定する必要がある場合があります（例：ビジョンエンコーダーを BF16 に設定）。

1.  **モデルをダウンロードしてください**:

    ``` console
    sima-user@docker-image-id:/home/docker$ hf download simaai/gemma3-siglip448 --local-dir gemma-3-model
    ```

2.  **設定ファイルを作成します**（例：`config.py`）。

    ``` python
    def get_layer_configuration(model_properties, layer):
        # Keep vision encoder in full precision
        if layer["part"] == "VISION":
            precision = "BF16"
        # Use INT8 for batch processing layers (better quality)
        elif layer["is_group"]:
            precision = "A_BF16_W_INT8"
        # Use INT4 for single-token layers (smaller size)
        else:
            precision = "A_BF16_W_INT4"
        return {"precision": precision}
    ```

3.  **コンパイラを実行します**：

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile -c config.py --max_num_tokens 2048 gemma-3-model -o gemma-3-model_out
    ```

**例4：高度な設定**

トランスフォーマー層ごとに制御可能な混合精度：

``` python
def get_layer_configuration(model_properties, layer):
    # PRE and POST indices normally identify transformer layers.
    if layer["part"] in {"PRE", "POST"} and layer["index"] < 4:
        return {"precision": "BF16"}

    # Keep every required compiler unit and use INT8 elsewhere.
    return {"precision": "A_BF16_W_INT8"}
```

`"CACHE"` のインデックスを、トランスフォーマー層のインデックスとして解釈しないでください。キャッシュのバリエーションを省略すると、コンパイルされた出力が不完全になり、ランタイムで利用できなくなる可能性があります。

**例 5: LLM を LoRA でコンパイルする**

LoRA (低ランク適応) を使用すると、ベースモデルを微調整し、アダプターを動的に適用または削除できます。これにより、ベースモデルを再コンパイルすることなく、ランタイムで変更できます。ベースモデルは、並列 LoRA ブランチ（ゼロで初期化）を使用してコンパイルされ、アダプターの重みは個別に `.npy` ファイルとしてコンパイルされ、必要に応じてロードされます。

:::note
LoRA を使用してコンパイルする場合は、フィルター共有を有効にする必要があります。`--enable_filter_sharing` を使用して有効にしてください。より高い精度を実現するために、LoRA ブランチは、INT4 が指定されていても、常に INT8 でコンパイルされます。
:::

1.  **ベースモデルとLoRAアダプターをダウンロードしてください**:

    ``` console
    sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B-Instruct --local-dir Llama-3.2-3B-Instruct
    sima-user@docker-image-id:/home/docker$ hf download <org>/<lora-adapter> --local-dir my-lora
    ```

2.  **設定ファイルを作成します**（例：`lora_config.py`）。

    `lora`キーは、各レイヤーごとのLoRAモードを制御します。`"LORA_BRANCH"`を使用して、ランタイム時に動的に切り替えを有効にします。

    ``` python
    def get_layer_configuration(model_properties, layer):
        if layer["is_group"]:
            return {"precision": "A_BF16_W_INT8", "compile": True, "lora": "LORA_BRANCH"}
        else:
            return {"precision": "A_BF16_W_INT4", "compile": True, "lora": "LORA_BRANCH"}
    ```

3.  **ベースモデルを、LoRAアダプターを使用してコンパイルします**:

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --enable_filter_sharing \
        --lora_name my_adapter \
        --lora_path my-lora \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

    これは、ベースモデルを1つのLoRAブランチでコンパイルし、アダプターの重みを`Llama-3.2-3B-lora-out/sima_files/npy_files/my_adapter/`に自動的にコンパイルします。

`--lora_name`と`--lora_path`を繰り返すことで、複数のアダプターを1回のステップでコンパイルできます。

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --enable_filter_sharing \
        --lora_name my_adapter_A --lora_path my-lora_A \
        --lora_name my_adapter_B --lora_path my-lora_B \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

4.  ベースモデルを再コンパイルせずに、さらに多くのアダプターを追加するには、各追加アダプターに対して `llima-compile-lora` を使用してください。

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile-lora Llama-3.2-3B-Instruct ./lora-c \
        -w Llama-3.2-3B-lora-out/sima_files/mpk \
        -o Llama-3.2-3B-lora-out/sima_files/npy_files/adapter_c
    ```

    **llima-compile-lora の引数**

    | 議論 | 説明 |
    |----|----|
    | `base_path` | 元のベースモデルディレクトリへのパス（HuggingFace形式）。 |
    | `lora_path` | コンパイルする際に使用する、LoRAアダプターディレクトリへのパス。 |
    | `-w, --weight_map_path` | **必須**。ベースモデルのコンパイルからの、`mpk/` フォルダへのパス。アダプターをコンパイルするために必要な重みマップが含まれています。 |
    | `-o, --output` | コンパイルされたアダプターファイル（`.npy`）の出力先ディレクトリ。デフォルトでは、アダプターのディレクトリ名が使用されます。 |
