# 為 LLiMa 做出貢獻

LLiMa 包含一個主機端 GenAI 編譯器和一個用於 Modalix 的 C++ 執行階段。該執行階段透過封裝後的 CLI/HTTP/ZMQ 入口點進行操作；Python 是 CLI 編排工具，而不是一個獨立的公開執行階段 API。請將編譯器和執行階段環境及其依賴項分開。在程式碼庫的檢出過程中，`CONTRIBUTING.md` 提供快速入門指南，而 `AGENTS.md` 定義了特定代理程式的規則；本指南是詳細的貢獻者政策。

## 程式設計代理人的技能

將兩個 LLiMa 開發者技能作為標準開發者設定的一部分進行安裝。
它們不會被預設的 Neat SDK 腳本索引安裝，這是故意的：

```bash
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-contribute-to-llima
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-add-llima-model-support
```

一般貢獻者技能涵蓋整個程式碼庫的編譯器、執行階段、封裝、測試、檔案和技能變更。模型支援技能新增了對 LLM 和 VLM 架構、檢查點、張量佈局、權杖化器和提示合約的相容性和實作流程。請確保同時安裝這兩者，以便在貢獻跨越這些界線時，提供適當的指導。


## 儲存庫地圖

| 區域 | 路徑 | 責任 |
| --- | --- | --- |
| 設定 | `sima_lmm/config/` | LLM、VLM 和 ASR 的設定合約。 |
| 攝取 | `sima_lmm/hf/`, `sima_lmm/gguf/` | Hugging Face 和 GGUF 的載入和轉換。 |
| 編譯 | `sima_lmm/model/`, `sima_lmm/preproc/` | 模型組件、量化、圖表和預處理。 |
| 主機工具 | `sima_lmm/host/` | 編譯、部署、LoRA，以及進行基準測試，以評估程式的進入點。 |
| 評估 | `sima_lmm/mole/` | MoLE 工作流程 |
| 執行階段 CLI | `sima_lmm/devkit/` | Python CLI 流程協調與模型管理 |
| C++ 執行階段 | `sima_lmm/devkit/cpp/` | 模型、權杖化器、MLA、CLI/HTTP/ZMQ 的實作，以及內部 CLI 繫結。 |
| 測試 | `tests/` | 編譯器和 Modalix 執行階段測試 |
| 包裝 | `CMakeLists.txt`, `cmake/`, `build*.sh`, `tools/install_*.sh` | Debian、wheel 格式，以及成品組裝。 |
| 持續整合/快取 | `.github/workflows/`, `tools/ci/`, `tools/hf-safetensors/` | 建立、測試和快取模型。 |
| 檔案/技能 | `README.md`, `docs/`, `skills/` | 使用者、貢獻者和《Playbooks》指南 |

僅編譯時所需的相依性不應納入 `sima_lmm/devkit/` 或 Modalix 執行階段套件。

## 開發環境

### 執行階段和封裝

請使用 Neat SDK 作為支援的建置環境。建置所有執行階段套件和封裝的測試：

```bash
./build.sh --all --clean
```

標準的建置程序會處理其所需的設定，包括子模組；請勿為標準工作流程執行單獨的依賴項啟動步驟。

有用的更精簡的建置程序：

```bash
./build.sh --clean --core
./build.sh --clean --core --dev
./build.sh --clean --cli
./build.sh --no-dist
```

輸出內容會在 `build-deb/` 中產生，並在 `dist/` 中進行分階段處理。Modalix 是執行 MLA 以及進行實際 `llima run` 驗證所必需的。

### 編譯器開發

請使用由 Model Compiler 安裝的 Python 3.12 環境。依序搜尋：

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

```bash
source <model-compiler-venv>/bin/activate
python -m pip install -e '.[sdk_ext,tests]'
llima-compile --help
```

請勿建立另一個環境，以免遮蔽已安裝的編譯器套件。請使用以下方式建立發布設定檔：

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

他們在 `build/` 中使用輪式工具，並在 `dist/compiler/` 和 `dist/mole/` 中進行階段性輸出。

### Whisper 和 ASR 的開發

公開的 `llima-compile` 工作流程涵蓋 LLM 模型和 VLM 模型。現有的 Whisper 編譯流程改用貢獻者工具：`scripts/gen_models--openai--whisper.py`。

```bash
python scripts/gen_models--openai--whisper.py \
  --model_path /path/to/openai/whisper-small \
  --output /path/to/whisper-output \
  --part all
```

在 Model Compiler 環境中執行，並明確指定模型路徑。
`--part` 接受 `all`、`encoder`、`language_detect`、`init`、`single_pre`、`single_post`，以及 `single_cache`。新增 `--enable_log_probe` 以編譯啟用日誌探測功能的解碼器輸出；使用 `--part all --enable_log_probe` 以進行完整的日誌探測建置。

編譯器變更通常會影響 `sima_lmm/config/whisper_config.py`、`sima_lmm/model/whisper_*.py` 和腳本；執行階段變更會影響 `sima_lmm/devkit/cpp/whisper_*`。使用封裝的 C++ ASR 執行階段測試進行驗證，該測試的相關檔案位於 `tests/README.md` 中，並使用 Modalix 上的代表性音訊進行測試。這是一個 Whisper 專用的路徑，而不是一個通用的 ASR 架構框架。

## 測試

依據失效面選擇測試。建置作業並不能取代行為驗證，而且跳過的必要測試案例不視為通過。

### 氣密性測試

將純粹的設定、映射、序列化、驗證和數值邏輯與模型下載分開：

```bash
pytest -q <targeted-test-path>
```

### 以模型為基礎的編譯器測試

編譯器測試位於 `tests/compilation/`。請選擇受影響的群組，以及 `tests/README.md` 中描述的標記。例如：

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
python -P -m pytest \
  -c pytest.ini \
  tests/compilation/configuration \
  -m compiler_config \
  --strict-markers \
  -vv -ra
```

`--model-inputs-path` 和 `LLIMA_HF_MODELS_PATH` 選取已準備好的 Hugging Face/GGUF 輸入根目錄。CI 使用位於 `tools/hf-safetensors/` 中的資訊檔。

設定所需的輸入，而不是接受跳過測試。

測試矩陣、預期的計數和基準策略位於 `tests/README.md`；CI 執行位於 `.github/workflows/model-compiler-tests.yml`。在執行期間生成 ONNX 和數值比較成品，而不是提交二進位基準。

### 執行階段驗證

建立候選套件和執行階段測試的額外元件：

```bash
./build.sh --all --clean
```

這會編譯程式碼，但不會執行測試。請安裝相符的 LLiMa 候選模型，以及在 Modalix 上安裝相關的內部元件套件。接著，解壓縮額外的檔案，並按照 `tests/README.md` 中 DevKit 執行階段測試的說明，執行封裝後的 CTest 和 pytest。

當程式碼變更影響到模型載入、推論、分詞、多模態預處理、推測式解碼、CLI/HTTP/ZMQ 或資源生命週期時，請執行相關的硬體測試。必要時，新增一個具有代表性的簡短測試。

```bash
llima run <model_dir> --mode cli
```

對於 VLM 的變更，請包含一個基於圖像的提示。手動煙霧測試是補充，但不會取代受影響的套件覆蓋範圍。

Neat Core 會使用已安裝的 LLiMa 的 C++ API 和執行階段套件。當上述任何一個部分（或透過 Core 的 GenAI API 公開的行為）發生變更時，請使用候選的 `sima-lmm-core` 和 `sima-lmm-dev` 套件來建置 Core，而不是使用已發布或快取的 LLiMa 建置版本。在 Modalix 上執行受影響的 Core GenAI C++ 測試。此下游驗證對於獨立的編譯器、檔案或僅測試的變更而言，並非必要。

### 包裝驗證

建立每個已變更的設定檔：

```bash
./build.sh --all --clean
./build_compiler_wheel.sh
./build_mole_package.sh
```

驗證套件名稱、檔案擁有者、安裝資訊檔、相依性、校驗總和以及中繼資料。

## 程式碼規範

### 相容性與界限

將已安裝的 C++ 標頭檔、CLI 指令、序列化的組態、套件中繼資料，以及產生的成品佈局視為相容性介面。 優先採用增量變更。 如果需要進行重大變更，請記錄受影響的元件、移轉方式和發布意圖；更新呼叫者、測試、範例和使用者檔案。

將編譯器、執行階段和 MoLE 依賴項分開。 執行階段狀態不得作為主機編譯的輸入。 對執行階段套件邊界的變更必須保留 `sima-lmm-core`、`sima-lmm-dev` 和 `sima-lmm-cli` 的角色，並包含適當的 API/ABI 驗證。

### 實施品質

- 目標 C++20 歲 Python 在下列位置宣告的版本： `pyproject.toml`.
- 遵循周圍的格式、命名方式，並包含群組；避免過於寬泛。
  機械格式重置。
- 盡可能減少已安裝介面的數量，並將實作細節保持私密。
- 在適當的地方新增 Python 類型註解。
- 解釋那些不易理解的合約條款、數值假設，以及硬體規格。
  限制；請勿逐行解釋程式碼。
- 在新增抽象概念之前，先重複使用附近的輔助函式。
- 保留確定性模型選擇、圖形結構和序列化功能。
  成品名稱。記錄種子，並避免檔案系統/程序排序。
- 拒絕不支援或無效的輸入，並提供相關資訊；保留原始原因。
  跨越各層級，且永遠不會靜默地選擇另一個執行路徑。
- 協調並拆解綁定工作流程。建立緩衝區、處理程序、執行緒，以及
  明確指定暫存檔案的所有權；安全地清理部分已完成的工作。
- 在熱門程式碼路徑中，避免不必要的資源設定、複製和同步操作。

### 模型、成品、依賴項和機密資訊

允許保留的測試材料：

- 已審閱 JSON 設定合約；
- 受版本控制的案例、原始資料、容差值和比較原則；以及
- 用於已批准且不可變更的 Hugging Face/GGUF 版本。

請勿提交已下載的權重、客戶資料或產生的 ONNX、NumPy、量化模型、MPK、ELF 或執行階段模型樹。將產生的輸出儲存在已忽略或暫存目錄中。

使用 `deps/manifest.json` 來管理套件/平台的版本。將 `third_party/` 視為已包含的程式碼；隔離並記錄有意的子模組更新。請勿將編譯器依賴項新增到執行階段 Debian 套件中。

切勿提交或記錄任何 Token、SSH 憑證、私人儲存庫、已簽署的 URL 或個人路徑。將受控模型的授權權限置於版本控制系統之外，並在報告中使用公開的模型 ID、不可變的修訂版本和已刪除的日誌。

## 檔案、技能和程式碼變更請求

更新最接近的使用者指南：

- [系統需求](setup.md)
- [模型編譯](compilation_genai.md)
- [模型部署](deployment.md)
- [ LLiMa CLI ](runtime.md)
- [MoLE](mole.md)

將根目錄的 `CONTRIBUTING.md` 檔案保留為快速入門指南，將此檔案作為詳細的策略檔案，並將 `AGENTS.md` 檔案作為可執行的代理規則。技能必須包含有效的 `SKILL.md`、`playbook.yml` 和代理中繼資料；保持其主要工作流程簡潔，並將條件細節移至直接引用中。

在 Neat SDK 中，驗證來自儲存庫根目錄的所有技能負載，且不得更改已安裝的代理狀態：

```bash
playbooks_validation_dir="$(mktemp -d)"
CODEX_HOME="${playbooks_validation_dir}/codex" \
CLAUDE_HOME="${playbooks_validation_dir}/claude" \
SIMA_CLI_HOME="${playbooks_validation_dir}/sima-cli" \
sima-cli playbooks install ./skills
```

安裝摘要必須報告 `detected: 3`、`valid: 3` 和 `discarded: 0`。
每個 `playbook.yml` 中的 `sima-cli` 版本必須滿足 `min_cli_version`。

對於提交的程式碼變更：

- 從目前的分支切換到並以目前的分支作為目標分支：`develop`；
- 讓提交訊息更具重點，使用祈使語氣的動詞開頭；
- 使用 `.github/PULL_REQUEST_TEMPLATE.md`。
- 將已完成的問題與「`Fixes #<issue>`」連結；
- 回報風險、相容性／移轉問題、檔案影響、可重現的指令
  模型/套件版本、硬體證據、已跳過的檢查項目，以及剩餘風險；
- 排除憑證和私人資產。

當相關測試在沒有意外跳過的情況下通過，所需套件和 Modalix 檢查完成或明確指出無法使用，且已處理相容性和檔案問題，並且 PR 包含可重複驗證的證據時，即表示貢獻已準備就緒。
