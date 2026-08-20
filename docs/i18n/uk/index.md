# GenAI з використанням LLiMa

LLiMa — це набір інструментів GenAI у Model Compiler, призначений для компіляції, тестування, проведення тестів продуктивності, розгортання та запуску моделей LLM, VLM та ASR на платформі Modalix.

LLiMa підтримує три формати вхідних даних:

- **Hugging Face safetensors** — стандартні директорії для моделей LLM та VLM.
- **Файли GGUF** — моделі LLM, упаковані у формат GGUF.
- **Стиснені тензорні моделі** — попередньо квантовані моделі у форматі safetensor, що використовують методи GPTQ/AWQ.

SiMa.ai також публікує попередньо скомпільовані GenAI моделі на
[ Hugging Face ](https://huggingface.co/simaai). Почніть звідти, якщо вже існує відповідна модель.

Для перегляду конкретних демонстрацій GenAI, див. [ приклади ](https://developer.sima.ai/examples).

## LLiMa Наявність

Інструменти для компіляції LLiMa встановлюються за замовчуванням у Model Compiler.
LLiMa runtime встановлюється нативно в Modalix як частина Neat runtime.
Див. [Neat Framework installation](/getting-started/neat-library/) для отримання інформації про процес встановлення runtime.

## Беручи участь / Вносячи свій внесок

Автори, які вносять зміни безпосередньо до LLiMa, повинні дотримуватися
[Посібника для авторів LLiMa ](contributing.md) щодо структури репозиторію,
середовищ розробки, рівнів тестування, політики щодо вхідних даних для моделі та вимог до запитів на внесення змін.

## Підтримувані моделі

Наступна таблиця містить інформацію про підтримувані архітектури моделей та їхні можливості:

| Архітектура моделі. | Тип | Підтримувані розміри |
|----|----|----|
| [Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family) | LLM | [7b](https://huggingface.co/simaai/Llama-2-7b-chat-hf-a16w4) |
| [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31) | LLM | [8b](https://huggingface.co/simaai/Llama-3.1-8B-Instruct-a16w4) |
| [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32) | LLM | [1b](https://huggingface.co/simaai/Llama-3.2-1B-Instruct-GPTQ-a16w4), [3b](https://huggingface.co/simaai/Llama-3.2-3B-Instruct-a16w4) |
| [Gemma 1](https://huggingface.co/collections/google/gemma-release) | LLM | 2б, 7б |
| [Gemma 2](https://huggingface.co/collections/google/gemma-2-release) | LLM | 2б, 9б |
| [Gemma 3](https://huggingface.co/collections/google/gemma-3-release) | LLM | [1b](https://huggingface.co/simaai/gemma-3-1b-it-a16w4), [4b](https://huggingface.co/simaai/gemma-3-4b-it-a16w4) |
| [ Phi 3.5 mini ](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | LLM | [3.8b](https://huggingface.co/simaai/Phi-3.5-mini-instruct-a16w4) |
| [Qwen 2.5](https://huggingface.co/collections/Qwen/qwen25) | LLM | [0.5b](https://huggingface.co/simaai/Qwen2.5-0.5B-Instruct-GPTQ-a16w4), [1.5b](https://huggingface.co/simaai/Qwen2.5-1.5B-Instruct-GPTQ-a16w4), [3b](https://huggingface.co/simaai/Qwen2.5-3B-Instruct-GPTQ-a16w4), [7b](https://huggingface.co/simaai/Qwen2.5-7B-Instruct-GPTQ-a16w4) |
| [Qwen 3](https://huggingface.co/collections/Qwen/qwen3) | LLM | [0.6b](https://huggingface.co/simaai/Qwen3-0.6B-GPTQ-a16w4), [1.7b](https://huggingface.co/simaai/Qwen3-1.7B-GPTQ-a16w4), [4b](https://huggingface.co/simaai/Qwen3-4B-Instruct-2507-GPTQ-a16w4), [8b](https://huggingface.co/simaai/Qwen3-8B-GPTQ-a16w4) |
| [Mistral 1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | LLM | [7b](https://huggingface.co/simaai/Mistral-7B-Instruct-v0.3-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2) | LLM | [350 м, ](https://huggingface.co/simaai/LFM2-350M-a16w4), [1,2 млрд, ](https://huggingface.co/simaai/LFM2-1.2B-a16w4), [2,6 млрд, ](https://huggingface.co/simaai/LFM2-2.6B-a16w4). |
| [Llava 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | VLM | [7b](https://huggingface.co/simaai/llava-1.5-7b-hf-a16w4) |
| [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) | VLM | [3b](https://huggingface.co/simaai/paligemma-3b-pt-224-a16w8) |
| [Gemma 3](https://huggingface.co/simaai/gemma3-siglip448-a16w4) | VLM | [4b](https://huggingface.co/simaai/gemma3-siglip448-a16w4) |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4) | VLM | [E2B](https://huggingface.co/simaai/gemma4-E2B-it), [E4B](https://huggingface.co/simaai/gemma4-E4B-it) |
| [ Qwen 2.5 VL ](https://huggingface.co/collections/Qwen/qwen25-vl) | VLM | [3b](https://huggingface.co/simaai/Qwen2.5-VL-3B-Instruct-GPTQ-a16w4), [7b](https://huggingface.co/simaai/Qwen2.5-VL-7B-Instruct-GPTQ-a16w4) |
| [ Qwen 3 VL ](https://huggingface.co/collections/Qwen/qwen3-vl) | VLM | [2b](https://huggingface.co/simaai/Qwen3-VL-2B-Instruct-GPTQ-a16w4), [4b](https://huggingface.co/simaai/Qwen3-VL-4B-Instruct-GPTQ-a16w4), [8b](https://huggingface.co/simaai/Qwen3-VL-8B-Instruct-GPTQ-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2-vl) | VLM | [450 м, ](https://huggingface.co/simaai/LFM2-VL-450M-a16w4), [1,6 млрд, ](https://huggingface.co/simaai/LFM2-VL-1.6B-a16w4), [3 млрд, ](https://huggingface.co/simaai/LFM2-VL-3B-a16w4). |
| [Whisper](https://huggingface.co/openai/whisper-small) | ASR | [small](https://huggingface.co/simaai/whisper-small-a16w8) |

## Обмеження

| Тип обмеження | Опис. |
|----|----|
| Архітектура моделі. | Підтримуються лише моделі, що базуються на описаних вище архітектурах. |
| Параметри моделі. | Підтримуються лише моделі, кількість параметрів яких становить менше 10 мільярдів. |
| Моделі HF | Моделі мають бути доступні у вигляді локальної директорії Hugging Face, яка містить `config.json`, файли з вагами у форматі safetensor, а також файли токенізатора та процесора, необхідні для даної архітектури. `generation_config.json` є необов’язковим. |
| GGUF Моделі | Формат GGUF підтримується лише для LLM. VLM необхідно скомпілювати з формату Hugging Face safetensors. Зверніть увагу, що продуктивність може знизитися порівняно з компіляцією Hugging Face safetensor. |
| Стиснені тензорні моделі. | Підтримувані LLM та VLM можуть використовувати попередньо квантовані моделі safetensor (GPTQ/AWQ), створені за допомогою llm-compressor. Модель повинна використовувати симетричне квантування та підтримуваний формат compressed-tensors. Ці моделі можуть досягати кращої точності, ніж стандартне INT4 квантування, зберігаючи при цьому високу продуктивність. |
| Gemma3 VLM | Підтримується модифікований кодувальник зображень SigLip 448. |
| LLAMA 3.2. Можливості візуального сприйняття | Моделі комп’ютерного зору не підтримуються. |
