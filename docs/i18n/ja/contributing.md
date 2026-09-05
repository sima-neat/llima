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

コンパイラの変更は通常、`sima_lmm/config/whisper_config.py`、`sima_lmm/model/whisper_*.py`、およびスクリプトに影響します。ランタイムの変更は、`sima_lmm/devkit/cpp/whisper_*` に影響します。`tests/README.md` に記載されているパッケージ化された C++ ASR ランタイムテストと、Modalix の代表的なオーディオを使用して検証します。これは、一般的な ASR アーキテクチャフレームワークではなく、Whisper に固有のパスです。

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

テストマトリックス、期待されるカウント、およびベースラインポリシーは、`tests/README.md`にあります。CIの呼び出しは、`.github/workflows/model-compiler-tests.yml`にあります。実行中にONNXと数値比較アーティファクトを生成し、バイナリベースラインをコミットするのではなく、それらを使用します。

### ランタイムでの検証

候補となるパッケージと、ランタイムでのテストに使用する追加コンポーネントをビルドします。

```bash
./build.sh --all --clean
```

これはビルドは行うものの、テストを実行しません。互換性のある候補であるLLiMaと、Modalix上の内部パッケージをインストールし、追加のアーカイブを抽出し、パッケージ化されたCTestとpytestを、`tests/README.md`に記載されているDevKitランタイムテストの手順に従って実行します。

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
