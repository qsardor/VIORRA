import os
import time
import json
import re
import faiss
import numpy as np
import litert_lm
import threading
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, logging as hf_logging
from optimum.onnxruntime import ORTModelForSequenceClassification
from huggingface_hub import hf_hub_download
from rich.console import Console

# Suppress Hugging Face Symlink warning on Windows and internal deprecation warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"
hf_logging.set_verbosity_error()

console = Console()

# Globals for singleton loading
is_loaded = False
embedder = None
index = None
corpus_texts = []
corpus_scores = []
aes_tokenizer = None
aes_model = None
llm_engine = None
_model_lock = threading.Lock()

def get_sys_prompt_path():
    return os.path.join(os.path.dirname(__file__), "system_prompt.txt")

def get_config():
    import json
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except:
            pass
    return {"backend": "GPU", "language": "English"}

def ensure_models_loaded():
    global is_loaded, embedder, index, corpus_texts, corpus_scores, aes_tokenizer, aes_model, llm_engine
    
    with _model_lock:
        if is_loaded:
            return

        console.print("[bold cyan]🔄 Initializing VIORRA Engine (This may take a moment if downloading models)...[/bold cyan]")
    
    # 1. RAG
    console.print("--> Loading Vector Database...")
    dataset = load_dataset('nid989/EssayFroum-Dataset', split='train')
    corpus_texts = dataset['Cleaned Essay'][:2000] # Expanded sample for better semantic search
    corpus_scores = [5] * len(corpus_texts)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    corpus_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    # 2. AES ONNX
    console.print("--> Loading ONNX AES Model...")
    aes_model_id = "Kevintu/Engessay_grading_ML"
    aes_tokenizer = AutoTokenizer.from_pretrained(aes_model_id)
    aes_model = ORTModelForSequenceClassification.from_pretrained(aes_model_id, export=True)

    # 3. LiteRT
    console.print("--> Downloading/Verifying LiteRT Gemma (2.5GB)...")
    llm_path = hf_hub_download(repo_id="litert-community/gemma-4-E2B-it-litert-lm", filename="gemma-4-E2B-it.litertlm")
    
    import torch
    # Detect NVIDIA CUDA (Windows/Linux) or Apple Silicon MPS (Mac)
    if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        backend_choice = "GPU"
        backend = litert_lm.interfaces.GPU()
    else:
        backend_choice = "CPU"
        backend = litert_lm.interfaces.CPU()
        
    console.print(f"--> Initializing engine on {backend_choice} (Auto-detected)...")
    llm_engine = litert_lm.Engine(llm_path, backend=backend)
    
    is_loaded = True
    console.print("[bold green]✅ VIORRA Engine Ready![/bold green]")

def analyze_essay(test_text: str):
    ensure_models_loaded()
    
    if len(test_text) < 50:
        return {"error": "Essay too short. Please provide at least 50 characters."}
        
    start_time = time.time()
    
    # 1. RAG
    query_embedding = embedder.encode([test_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 2)
    rag_examples = ""
    for i, idx in enumerate(indices[0]):
        rag_examples += f"\n--- HISTORICAL EXAMPLE {i+1} ---\n"
        rag_examples += corpus_texts[idx][:400] + "...\n"
        
    # 2. AES
    aes_inputs = aes_tokenizer(test_text, return_tensors="np", truncation=True, max_length=512)
    aes_outputs = aes_model(**aes_inputs)
    logits = aes_outputs.logits[0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    classes = np.arange(len(probs), dtype=np.float32)
    raw_score = (probs * classes).sum().item()
    final_aes_score = round(max(1.0, min(5.0, raw_score)), 2)
    clarity_score = int((final_aes_score/5.0)*100)

    # 3. Construct Prompt
    try:
        with open(get_sys_prompt_path(), "r", encoding="utf-8") as f:
            sys_prompt = f.read()
    except FileNotFoundError:
        sys_prompt = "[STUDENT ESSAY]\n\"[[TEST_TEXT]]\"\n[INSTRUCTIONS]\nOutput empty JSON."
        
    config = get_config()
    lang = config.get("language", "English")
    if lang != "English":
        sys_prompt += f"\n\nCRITICAL DIRECTIVE: You MUST generate your entire response (including all feedback and analysis text fields) strictly in {lang}. Do not use English."
        
    sys_prompt = sys_prompt.replace("[[TEST_TEXT]]", test_text)
    sys_prompt = sys_prompt.replace("[[AES_SCORE]]", str(final_aes_score))
    sys_prompt = sys_prompt.replace("[[CLARITY_SCORE]]", str(clarity_score))
    sys_prompt = sys_prompt.replace("[[RAG_EXAMPLES]]", rag_examples)
    
    # 4. LiteRT Generate
    with llm_engine.create_conversation() as conversation:
        response = conversation.send_message(sys_prompt)
        output_text = response["content"][0]["text"]
            
    # Parse JSON
    try:
        json_str = output_text
        match = re.search(r'\{[\s\S]*\}', output_text)
        if match:
            json_str = match.group(0)
        data = json.loads(json_str)
        data["generation_time"] = round(time.time() - start_time, 2)
        return data
    except Exception as e:
        return {"error": "Failed to parse JSON.", "raw_output": output_text}

