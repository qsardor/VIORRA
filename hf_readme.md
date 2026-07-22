---
base_model: google/gemma-4-E2B-it
tags:
- gemma4
- gemma
- gguf
- llama.cpp
- unsloth
- vision-language-model
- qat
---

# Viorra-Gemma-4-E2B-GGUF (QAT Editions)

This repository contains the fine-tuned Viorra reasoning models converted to GGUF format using [Unsloth](https://github.com/unslothai/unsloth). 

> **Important QAT Notice**: These models were exported natively using Unsloth's push-to-hub pipeline, meaning they are **Quantization-Aware Training (QAT) aligned**. They retain the dynamic grid alignment from the training process, preserving significantly higher accuracy and byte-exactness compared to naive post-training GGUF conversions.

## Available Model files & VRAM Requirements:

| Model File | Precision | File Size | Est. Peak VRAM (Context) | Recommended Use |
|---|---|---|---|---|
| `gemma-4-e2b-it.Q4_K_M.gguf` | 4-bit (QAT) | 3.43 GB | ~4.5 GB | Best balance of extreme speed and reasoning accuracy. Ideal for local deployment. |
| `gemma-4-e2b-it.Q8_0.gguf` | 8-bit (QAT) | 4.95 GB | ~6.0 GB | Maximum precision fallback for server-grade accuracy. |
| `gemma-4-e2b-it.BF16-mmproj.gguf` | 16-bit | 0.98 GB | N/A | Vision Encoder Projection (used for multimodal). |

**Example usage**:
- For text only LLMs:    `llama-cli -hf qsardor/Viorra-Gemma-4-E2B-GGUF --jinja`
- For multimodal models: `llama-mtmd-cli -hf qsardor/Viorra-Gemma-4-E2B-GGUF --jinja`

## ⚠️ Critical Formatting Requirements (Reasoning Engine)

Viorra 1.3 is a specialized reasoning model. It no longer uses the standard Gemma 4 chat template under the hood. If you are manually formatting prompts (bypassing `apply_chat_template`), you **must** use the following exact structure, otherwise the model will hallucinate or enter an infinite loop:

1. **You must prepend `<bos>`** to the very beginning of the prompt.
2. **Use `<|turn>user` (Not `<|turn|>user`)**: The tags are asymmetrical.
3. **Trigger the Reasoning Block**: To activate Viorra's internal thinking, include `<|think|>` in the system prompt. The model will respond with `<|channel>thought\n` before its final answer.

**Correct Manual Prompt Structure:**
```text
<bos><|turn>system
<|think|>
You are Viorra, a brutal essay reviewer.<turn|>
<|turn>user
Review my essay: ...<turn|>
<|turn>model
```

## ⚠️ Ollama Note for Vision Models
**Important:** Ollama currently does not support separate mmproj files for vision models.

To create an Ollama model from this vision model:
1. Place the `Modelfile` in the same directory as the finetuned bf16 merged model
3. Run: `ollama create model_name -f ./Modelfile`
   (Replace `model_name` with your desired name)

This will create a unified bf16 model that Ollama can use.
This was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth)
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
