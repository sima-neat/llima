# LLiMa への貢献

LLiMa には、ホスト側の GenAI コンパイラと、Modalix 用の C++ ランタイムが含まれています。このランタイムは、パッケージ化された CLI/HTTP/ZMQ エントリーポイントを通じて動作します。Python は CLI オーケストレーションであり、独立した公開ランタイム API ではありません。コンパイラとランタイムの環境および依存関係は分離してください。リポジトリのチェックアウトでは、`CONTRIBUTING.md` がクイックスタートを提供し、`AGENTS.md` がエージェント固有のルールを定義します。このガイドは、詳細なコントリビューターポリシーです。

## コーディングエージェントのスキル

標準のコントリビューター設定の一部として、両方のLLiMaコントリビュータースキルをインストールしてください。
これらは、デフォルトのNeat SDKプレイブックインデックスによって意図的にインストールされません。

```bash
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-contribute-to-llima
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-add-llima-model-support
```

一般的なコントリビューターのスキルは、リポジトリ全体にわたるコンパイラ、ランタイム、パッケージング、テスト、ドキュメント、およびスキルの変更を対象とします。モデルサポートスキルは、LLMおよびVLMアーキテクチャ、チェックポイント、テンソルレイアウト、トークナイザー、およびプロンプトコントラクトの互換性と実装ワークフローを追加します。両方をインストールした状態にして、コントリビューションがこれらの境界を越える場合に適切なガイダンスが利用できるようにします。


## リポジトリマップ

| 面積 | パス | 責任 |
| --- | --- | --- |
| 設定 | `sima_lmm/config/` | LLM、VLM、およびASR の設定契約 |
| 摂取 | `sima_lmm/hf/`, `sima_lmm/gguf/` | Hugging Face および GGUF の読み込みと変換 |
| コンパイル | `sima_lmm/model/`, `sima_lmm/preproc/` | モデルの構成要素、量子化、グラフ、および前処理 |
| ホストツール | `sima_lmm/host/` | コンパイル、デプロイ、LoRA、およびエントリポイントのベンチマークを実施します。 |
| 評価 | `sima_lmm/mole/` | MoLE ワークフロー |
| ランタイム CLI | `sima_lmm/devkit/` | Python CLIによるオーケストレーションとモデル管理 |
| C++ ランタイム | `sima_lmm/devkit/cpp/` | モデル、トークナイザー、MLA、CLI/HTTP/ZMQの実装、および内部CLIバインディング。 |
| テスト | `tests/` | コンパイラとModalixのランタイムテスト |
| パッケージ | `CMakeLists.txt`, `cmake/`, `build*.sh`, `tools/install_*.sh` | Debian、wheel、およびアーティファクトの組み立て |
| CI/キャッシュ | `.github/workflows/`, `tools/ci/`, `tools/hf-safetensors/` | ビルド、テスト、およびモデルキャッシュの作成 |
| ドキュメント/スキル | `README.md`, `docs/`, `skills/` | ユーザー、コントリビューター、およびプレイブックに関するガイダンス |

コンパイラのみに依存するパッケージは、`sima_lmm/devkit/` または Modalix のランタイムパッケージに含めてはなりません。

## 開発環境

### ランタイムとパッケージング

サポート対象のビルド環境として、Neat SDK を使用してください。すべてのランタイムパッケージとパッケージ化されたテストを、次のコマンドでビルドします。

```bash
./build.sh --all --clean
```

通常のビルドでは、サブモジュールを含む必要なセットアップがすべて行われます。標準のワークフローでは、別途依存関係のブートストラップ手順を実行する必要はありません。

便利な、より限定的なビルド：

```bash
./build.sh --clean --core
./build.sh --clean --core --dev
./build.sh --clean --cli
./build.sh --no-dist
```

出力は`build-deb/`の下に生成され、`dist/`の下に配置されます。MLAの実行と、実際の`llima run`の検証には、Modalixが必要です。

### コンパイラ開発

Model CompilerによってインストールされたPython 3.12環境を使用してください。以下の順序で検索します。

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

```bash
source <model-compiler-venv>/bin/activate
python -m pip install -e '.[sdk_ext,tests]'
llima-compile --help
```

インストール済みのコンパイラパッケージと競合するような、別の環境を作成しないでください。以下の方法で、公開用のビルドプロファイルを作成してください。

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

彼らは、`build/` の下でホイールツールを使用し、`dist/compiler/` と `dist/mole/` の下で出力結果をステージングします。

### WhisperとASRの開発

公開されている`llima-compile`ワークフローは、LLMとVLMを対象としています。既存のWhisperのコンパイルには、代わりにコントリビューターが提供するユーティリティである`scripts/gen_models--openai--whisper.py`が使用されます。

```bash
python scripts/gen_models--openai--whisper.py \
  --model_path /path/to/openai/whisper-small \
  --output /path/to/whisper-output \
  --part all
```

明示的なモデルパスを指定して、Model Compiler 環境で実行してください。

`--part` は、`all`、`encoder`、`language_detect`、`init`、`single_pre`、`single_post`、および `single_cache` を受け入れます。ログプローブを有効にしたデコーダーの出力をコンパイルするには、`--enable_log_probe` を追加します。完全なログプローブビルドを行うには、`--part all --enable_log_probe` を使用します。

Whisper モデルリポジトリには、エンコーダーのレイヤーごとに 1 つの ELF が含まれます。ランタイムは、エンコーダー ELF が単一にまとめられた従来のリポジトリをサポートしていません。レイヤー化されたモデルをダウンロードするか、現在の LLiMa バージョンでチェックポイントを再コンパイルしてください。

コンパイラの変更は通常、`sima_lmm/config/whisper_config.py`、`sima_lmm/model/whisper_*.py`、およびスクリプトに影響します。ランタイムの変更は、`sima_lmm/devkit/cpp/whisper_*` に影響します。`tests/README.md` に記載されているパッケージ化された C++ ASR ランタイムテストと、Modalix の代表的なオーディオを使用して検証します。これは、一般的な ASR アーキテクチャフレームワークではなく、Whisper に固有のパスです。

### ネイティブグラフコンポーネント

`sima_lmm/model/model_graph.py` の `ModelGraph` を使用します。コンポーネントのソース重み、精度、出力パスを一度設定します。グラフノードの構築はクラス内で行い、独立した配列レイアウトや命名のユーティリティはモジュール関数として残します。モデルコンポーネントはグラフのメソッドを使用するため、実装ではトポロジーに集中できます：

```python
from sima_lmm.model.model_graph import ModelGraph

def generate_graph(self, layer_cfg, quantizable):
    graph = ModelGraph(self, {"hidden": (1, 1, self.num_tokens, self.cfg.d_model)}, quantizable)
    hidden = graph.layer_norm("model.norm", graph.inputs["hidden"])
    output = graph.mlp("model.mlp", hidden, "gelu", residual=graph.inputs["hidden"])
    graph.save([output])
```

独立したコンポーネントの入力、トップレベルのトポロジー、`graph.save()` は `generate_graph()` 内で定義します。Whisper の統合デコーダーグラフで使用する pre/cache/post 部分など、共有するグラフ構築には `_build_nodes(graph, inputs)` を残します。

ログの範囲は `BaseModel.gen_files()` で設定されるため、グラフ構築に別のログ引数は不要です。

形状から推論される入力型は、`quantizable=True` の場合は FP32（後で量子化する浮動小数点グラフ）、`False` の場合は BF16（ソース重みの精度を使用する直接グラフ）です。キャッシュなどの整数入力には `input_dtypes={"cache": np.int8}` を使用し、名前を入力仕様に一致させます。低レベルの呼び出しでは、既存の明示的な AFE テンソル仕様も使用できます。入力と出力の順序は指定した仕様に従い、形状と出力型は AFE が推論します。`constant()` は浮動小数点データをアクティベーションの精度に変換します。特定のビット幅の整数定数が必要な場合は、例えば `dtype=np.int32` を指定します。ノード名は AFE の決定的な作成カウンターに従います。

グラフノードの型注釈には `model_graph.py` から `Node` をインポートします。ヘルパーには、別の `quantizable` フラグではなくグラフを渡します。`graph.constant()` は浮動小数点の精度を自動選択します。ホスト側の配列計算で同じ精度が必要な場合は、NumPy の `graph.dtype` を使用します。段階を示すフラグは `generate_graph()` とグラフの構築時にだけ使用します。

`save()` は MLA サブネットを終了し、外側のグラフの出力タプルを作成します。整数出力を保持し、EV 上で BF16 出力を FP32 に変換し、標準のアーティファクト名で保存します（浮動小数点グラフには `.fp32`）。完成したネットワークを取得するには `finish()` を使用します。どちらも `transform_subnet` を受け取り、外側の出力を取り出す前にモデル固有の書き換えを行えます。コンポーネントのヘルパーにはグラフ自体を渡し、入力、精度、ソース重みを同じオブジェクトに保持します。

名前付きの NumPy 入力で、完成したグラフを実行します：

```python
graph.finish([output])
outputs = graph.run(hidden=x)
outputs_jax = graph.run(hidden=x, use_jax=True)
graph.save()
```

入力の名前、形状、データ型は宣言した入力に一致する必要があり、暗黙的な型変換は行いません。出力は `finish()` に渡した順序に従います。NumPy 実行は AFE の fast mode を使用し、MLA の参照演算と結果が異なる場合があります。`use_jax=True` は JAX の参照実行を選択し、fast mode は効果を持ちません。JAX の演算は設定されたバックエンドを使用し、対応する JAX 環境では GPU を使用できます。この API は量子化、コンパイル、Modalix 上での実行を行いません。一度だけ終了処理を行い、その後は `run()` と `save()` を繰り返し使用します。`save(outputs)` で終了と保存を一度に行うこともできます。

一般的な操作には `add`, `sub`, `mul`, `matmul`, `concat`, `slice`, `transpose`, `reshape`, `softmax`, `topk`, `sum_channels`, `argmax`, `linear`, `conv`, `layer_norm`, `rms_norm`, `activation`, `softcap`, `mlp`, `rope`, `rope2d`, `split_heads`, `merge_heads`, `split_concat`, `clip`, `avgpool2d`, `space_to_depth`, `quant`, `dequant` があります。ゲート付き MLP には `projections=("gate_proj", "up_proj", "down_proj")` を使用し、既定値は `("fc1", "fc2")` です。`rms_norm("model.norm", input)` は言語モデル設定の epsilon と重みオフセットを使用します。vision や GDN の正規化で `epsilon` を明示すると、`weight_offset` も指定しない限り重みオフセットはゼロです。`rms_norm(None, input, epsilon=...)` は重みのないチャネルを推論します。RoPE は全体、部分、比率指定の split-half 回転をサポートします。`rope2d(input, cos_x, sin_x, cos_y, sin_y)` はチャネルの各四分の一を `[x-real, x-imag, y-real, y-imag]` の順で回転します。`split_heads(input, heads, repeat=...)` は grouped-query パターン向けに各ヘッドを繰り返します。`split_concat()` は、reshape だけでは表せない空間やトークンのレイアウトに AFE の再グループ化操作を提供します。`space_to_depth(input, blocksize)` は完全な空間ブロックをチャネルに統合します。`quant(input)` は `(int8_values, scale)` を返し、`dequant(int8_values, scale)` はグラフのアクティベーション精度を復元します。`argmax(input)` は INT32 のチャネルインデックスを返します。`slice(input, begin, end, stride, axis)` は、ランク 4 の FP32/BF16 テンソルに対する非整列の連続した単一軸チャネルスライスで、セレクター畳み込みを自動使用します。その他のスライスは AFE のネイティブ動作を維持します。

`linear("model.proj", input)` は `model.proj.weight` と任意のバイアスを解決し、OI のソース重みを SiMa のレイアウトに変換します。パックされた重み、スケール、整列していないグループサイズ、リロケーションのメタデータを保持します。`conv()` は OIW または OIHW のソースレイアウトを推論します。明示的な変換で、Qwen の 5 次元パッチ重みなど他のレイアウトも変換できます。`WeightOptions` はソース名、レイアウト、重み・スケール・バイアスの変換、リロケーションの上書き設定を説明します。グループ化された重みをスライスする場合は、対応する `scale_process_func` を指定し、チェックポイントの実際のグループサイズを保持します。LoRA のランクと統合アダプターの動作は、`linear()` と `mlp()` の明示的な引数です。

射影とヘッドのレイアウト操作で attention を構築します。既存の射影の丸めを保つため、クエリのスケーリングを `linear()` に組み込みます：

```python
queries = graph.split_heads(graph.linear("attn.q_proj", hidden, scale=head_dim ** -0.5), heads)
keys = graph.split_heads(graph.linear("attn.k_proj", hidden), heads)
values = graph.split_heads(graph.linear("attn.v_proj", hidden), heads)
context = graph.attention(queries, keys, values)
output = graph.linear("attn.out_proj", graph.merge_heads(context))
```

`split_heads()` は整列していないヘッドのチャネルをセレクター畳み込みで処理します。標準の vision attention は、これらの畳み込みを避けるため非整列ヘッドの射影重みをパディングします。グループ化された出力重みは元のレイアウトを保持します。`attention()` はクエリとキーの長さから、cross-attention を含む大きな attention テンソルでヘッド別の分岐を選択します。加算用の `mask` を受け取り、`score_scale` を指定しない限りクエリはスケーリング済みとみなします。このスケールは Q×K の後、マスクの前に適用し、Qwen vision などの BF16 演算順序を保持します。マスクはランク 4、ベクトル、スカラーのいずれかで、`[N,H,T_query,T_key]` にブロードキャストできる必要があります。未対応の形状では `ValueError` が発生します。

`graph.matmul(lhs, rhs)` は MLA のバッチ行列積を作成します。転置フラグの既定値はどちらも `False` です。Kᵀ×V には `transpose_a=True`、Q×Kᵀ には `transpose_b=True` を使用します。入力はランク 4 の FP32/BF16 テンソルで、バッチと縮約次元が一致する必要があります。ヘッド数が割り切れる場合、grouped-query attention では暗黙的な繰り返しを使用します。不正な形状や型では `ValueError` が発生します。

`graph.softmax(x)` の既定の軸は最後の軸です。`axis` の明示的な指定も可能です。連続した単一軸のスライスには `graph.slice(x, start=0, stop=128, axis=-1)` を使用し、複数軸には既存の `begin`/`end`/`stride`/`axis` リストを使用します。単一軸形式の開始位置は既定でゼロです。軸の明示と、範囲内の空でない境界が必要で、非整列チャネルの処理も維持します。二つの形式を混在させないでください。

`ModelGraph` は AFE の `SimaBuilder` を拡張するため、共通ヘルパーとネイティブ操作を同じオブジェクトで使用できます：

```python
projected = graph.linear("model.proj", graph.inputs["hidden"])
output = graph.add(projected, graph.inputs["hidden"])
graph.save([output])
```

簡潔なプリミティブ操作は AFE メソッドの直接の別名であり、シグネチャ、型推論、決定的な命名を保持します。一般的でない操作には、継承された `create_*` メソッドを同じグラフで使用できます。独自の複数サブネットのライフサイクルが必要なグラフでは、AFE の `SimaBuilder` と低レベルの操作ヘルパーを直接使用できます。

テセレーションは `sima_analysis.get_tessellate_parameters()` で一元的に推論します。HWC16 レイアウト、自動タイルサイズ、決定的な永続バッファ名を使用します。通常のコンポーネントは `BaseModel` から空の上書き設定を継承します。ストライド付き KV キャッシュなどの特殊なレイアウトでは、既存の `get_mla_input_tessellate_params()` / `get_mla_output_tessellate_params()` メソッドを上書きします。キーはテンソルのインデックスで、負のインデックスは末尾から数えます。

Whisper コンポーネント、すべてのネイティブ言語グラフ部分、標準の vision tower でこの API を使用しています。attention、MLP、回転のヘルパーは同じ実装を共有します。

## テスト

エラーが発生した箇所に基づいてテストを選択します。ビルドは、動作検証の代わりにはなりません。また、スキップされた必須テストケースは、合格とはみなされません。

### 気密試験

純粋な設定、マッピング、シリアライズ、検証、および数値演算のロジックを、モデルのダウンロードとは独立して維持します。

```bash
pytest -q <targeted-test-path>
```

### モデルを活用したコンパイラテスト

コンパイラテストは、`tests/compilation/` にあります。影響を受けるグループと、`tests/README.md` に記載されているマーカーを選択してください。例：

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
python -P -m pytest \
  -c pytest.ini \
  tests/compilation/configuration \
  -m compiler_config \
  --strict-markers \
  -vv -ra
```

`--model-inputs-path`と`LLIMA_HF_MODELS_PATH`は、準備されたHugging FaceのGGUF入力ルートを選択します。CIは、`tools/hf-safetensors/`の下にあるマニフェストを使用します。
フィクスチャのスキップを受け入れる代わりに、必要な入力を設定します。

テストマトリックス、期待される件数、およびベースラインポリシーは、`tests/README.md` に記載されています。CI の呼び出しは `.github/workflows/model-compiler-tests.yml` にあります。バイナリのベースラインをコミットする代わりに、実行中にネイティブ SDK グラフと数値比較アーティファクトを生成してください。

### ランタイムでの検証

候補となるパッケージと、ランタイムでのテストに使用する追加コンポーネントをビルドします。

```bash
./build.sh --all --clean
```

これはビルドのみを行い、テストは実行しません。Modalix に対応する候補の LLiMa パッケージをインストールし、追加のアーカイブを抽出して、`tests/README.md` に記載されている DevKit ランタイムテストの手順に従って、パッケージ化された CTest と pytest を実行します。

モデルのロード、推論、トークン化、マルチモーダル前処理、推測デコーディング、CLI/HTTP/ZMQ、またはリソースのライフサイクルに変更があった場合に、関連するハードウェアテストを実行します。必要に応じて、代表的なスモークテストを追加してください。

```bash
llima run <model_dir> --mode cli
```

VLM の変更については、画像に基づいたプロンプトを含めてください。手動による簡易テストは、影響を受けるパッケージのテスト範囲を補完しますが、完全に置き換えるものではありません。

Neat Core は、インストールされた LLiMa の C++ API およびランタイムパッケージを使用します。これらのいずれか、または Core の GenAI API を介して公開される動作が変更された場合は、公開またはキャッシュされた LLiMa のビルドではなく、候補となる `sima-lmm-core` および `sima-lmm-dev` パッケージに対して Core をビルドしてください。影響を受ける Core の GenAI C++ テストを Modalix 上で実行してください。この下流の検証は、独立したコンパイラ、ドキュメント、またはテストのみの変更には必要ありません。

### パッケージの妥当性確認

変更された各プロファイルをビルドします。

```bash
./build.sh --all --clean
./build_compiler_wheel.sh
./build_mole_package.sh
```

パッケージ名、ファイルの所有者、インストールマニフェスト、依存関係、チェックサム、およびメタデータを検証します。

## コーディング規約

### 互換性と範囲

インストールされたC++ヘッダー、CLIコマンド、シリアライズされた構成、パッケージメタデータ、および生成されたアーティファクトのレイアウトを、互換性のためのインターフェースとして扱います。可能な限り、変更を段階的に追加していくことを推奨します。互換性を損なう変更を行う場合は、影響を受けるコンシューマー、移行方法、およびリリース意図をドキュメントに明記し、呼び出し元、テスト、サンプル、およびユーザー向けドキュメントを更新してください。

コンパイラ、ランタイム、およびMoLEの依存関係は分離して管理します。ランタイムの状態は、ホストのコンパイルへの入力として使用しないでください。ランタイムパッケージの境界を変更する場合は、`sima-lmm-core`、`sima-lmm-dev`、および`sima-lmm-cli`の役割を維持し、適切なAPI/ABIの検証を含める必要があります。

### 実装の品質

- `pyproject.toml` に宣言されている、対象の C++ 20 および Python のバージョン。
- 周囲の書式、命名規則、およびグループ化に従い、広範な表現は避けてください。
  機械的な再フォーマット。
- インストールされているインターフェースの数を最小限に抑え、実装の詳細を非公開にしてください。
- 可能な限り、Python 型アノテーションを追加してください。
- 一見してわかりにくい契約条項、数値的な前提条件、およびハードウェアについて説明してください。
  制約：コードの内容を説明するような記述はしないでください。
- 抽象化を追加する前に、まず近くにある既存のヘルパー関数を再利用する。
- 決定的なモデル選択、グラフ構造、シリアライズ、および
  アーティファクト名。シード値を記録し、ファイルシステムやプロセスの順序による影響を避けてください。
- サポートされていない、または無効な入力があった場合は、その理由を明示して拒否し、元の原因を保持する。
  複数のレイヤーを横断し、決して静かに別の実行パスを選択することはありません。
- バウンドされたワーカーの連携と分解を行います。バッファー、ハンドル、スレッドを作成します。
  一時ファイルの所有権を明示的に設定し、部分的な作業を安全にクリーンアップする。
- 頻繁に実行される処理において、不要なメモリ割り当て、コピー、および同期処理は避けてください。

### モデル、アーティファクト、依存関係、および機密情報

試験で使用できる資料（再利用可能）：

- JSON 構成契約をレビューしました。
- バージョン管理された事例、初期データ、許容範囲、および比較ポリシー。
- 承認された不変の Hugging Face/ GGUF リビジョンのマニフェスト。

ダウンロードした重み、顧客データ、または生成されたONNX、NumPy、量子化されたデータ、MPK、ELF、またはランタイムモデルツリーをリポジトリにコミットしないでください。生成された出力は、無視されるディレクトリまたは一時ディレクトリに保存してください。

パッケージ/プラットフォームのバージョンには、`deps/manifest.json`を使用してください。`third_party/`は、サードパーティのコードとして扱い、意図的なサブモジュールの更新を分離し、ドキュメント化してください。ランタイムDebianパッケージにコンパイラ依存関係を追加しないでください。

トークン、SSH認証情報、プライベートリポジトリ、署名されたURL、または個人パスをリポジトリにコミットしたり、ログに記録したりしないでください。アクセス制御されたモデルの認証は、ソース管理の外部に置き、レポートでは公開モデルID、不変のバージョン、および編集されたログを使用してください。

## ドキュメント、スキル、プルリクエスト

最も関連性の高いユーザーガイドを更新してください。

- [システム要件](setup.md)
- [モデルのコンパイル](compilation_genai.md)
- [モデルのデプロイ](deployment.md)
- [ LLiMa CLI ](runtime.md)
- [MoLE](mole.md)

ルートの`CONTRIBUTING.md`をクイックスタートとして、このファイルを詳細なポリシーとして、そして`AGENTS.md`を強制可能なエージェントルールとして保持してください。スキルには、有効な`SKILL.md`、`playbook.yml`、およびエージェントのメタデータが含まれている必要があります。主要なワークフローは簡潔に保ち、条件付きの詳細を直接参照に移動してください。

リポジトリのルートから、Neat SDKを使用して、すべてのスキルのペイロードを検証し、インストールされたエージェントの状態を変更せずに実行してください。

```bash
playbooks_validation_dir="$(mktemp -d)"
CODEX_HOME="${playbooks_validation_dir}/codex" \
CLAUDE_HOME="${playbooks_validation_dir}/claude" \
SIMA_CLI_HOME="${playbooks_validation_dir}/sima-cli" \
sima-cli playbooks install ./skills
```

インストール概要には、`detected: 3`、`valid: 3`、および`discarded: 0`の結果を必ず含めてください。
各`playbook.yml`において、`sima-cli`のバージョンは、`min_cli_version`を満たしている必要があります。

プルリクエストの場合：

- 現在の`develop`から分岐させ、それをターゲットとする。
- コミットの内容を簡潔にするために、命令形を使用しましょう。
- `.github/PULL_REQUEST_TEMPLATE.md` を使用します。
- 解決済みの課題と、`Fixes #<issue>` をリンクします。
- リスク、互換性/移行、ドキュメントへの影響、再現可能なコマンドを報告する。
  モデル/パッケージのバージョン、ハードウェアの証拠、スキップされたチェック、および残存リスク。
- 認証情報と機密性の高い資産は除外してください。

関連するテストが意図しないスキップなしで正常に完了し、必要なパッケージとModalixのチェックが完了しているか、または明示的に利用できない状態になっている場合、互換性とドキュメントが適切に処理され、プルリクエストに再現可能な証拠が含まれている場合に、貢献として受け入れられます。
