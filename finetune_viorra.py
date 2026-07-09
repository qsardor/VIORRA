"""
Viorra Fine-Tuning Pipeline (Unsloth)
Target Model: unsloth/gemma-4-e2b-it
Target Dataset: Roman1111111/claude-sonnet-4.6-120000x

This script securely loads the Gemma 4 E2B model with 4-bit quantization, 
applies QLoRA to prevent amnesia, trains the model on the Claude Sonnet 
reasoning dataset (retaining <|think|> tokens), and automatically exports 
the final weights as a GGUF file.
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration
max_seq_length = 4096 # Gemma 4 supports up to 128K, but 4096 is safe for training VRAM
dtype = None # Auto-detect
load_in_4bit = True # Use 4-bit quantization to fit on consumer GPUs

# 2. Load Base Model (Gemma 4 E2B Instruction Tuned)
print("Loading Unsloth Gemma 4 E2B model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-4-e2b-it",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Configure QLoRA Adapters
print("Applying QLoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is optimized for Unsloth
    bias = "none",
    use_gradient_checkpointing = "unsloth", # CRITICAL: Gemma 4 requires 'unsloth' checkpointing
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Prepare the Claude Sonnet Dataset
print("Loading Filtered Humanities/Empathy Dataset...")
# Load the locally compiled and filtered dataset to prevent loading coding slop
dataset_path = r"C:\Users\Sardor\.gemini\antigravity\scratch\viorra_dataset\masonmac_filtered.jsonl"
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Gemma 4 Chat Template formatting function
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma",
)

def format_prompts(examples):
    convos = examples["messages"]
    # Preprocess to stitch the reasoning field into assistant content
    for convo in convos:
        for msg in convo:
            if msg.get("role") == "assistant" and "reasoning" in msg and msg.get("reasoning"):
                # Embed reasoning trace inside <think> tags in content
                # This prevents Unsloth's strip_thinking macro from deleting it
                msg["content"] = f"<think>\n{msg['reasoning']}\n</think>\n{msg['content']}"
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts }

# Map dataset (using batched processing for speed)
dataset = dataset.map(format_prompts, batched = True)

# 5. Training Setup
print("Configuring SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100, # Start with 100 steps for a dry-run test
        learning_rate = 2e-4,
        fp16 = not FastLanguageModel.is_bfloat16_supported(),
        bf16 = FastLanguageModel.is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Execute Training
print("Starting Fine-Tuning...")
trainer_stats = trainer.train()

# 7. Export to GGUF
print("Training Complete! Exporting to GGUF format for llama.cpp...")
# This automatically merges the LoRA adapters and exports to q4_k_m for extreme speed
model.save_pretrained_gguf("viorra-sonnet-e2b", tokenizer, quantization_method = "q4_k_m")
print("Export complete! File ready: viorra-sonnet-e2b-unsloth.Q4_K_M.gguf")
