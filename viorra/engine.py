"""
VIORRA CORE ENGINE
------------------
This module is the absolute heart of VIORRA. It handles:
1. Hardware Profiling: Natively querying the OS (WMI/sysctl/lspci/nvidia-smi) to find the best GPU backend.
2. Vector Database (RAG): Downloading and loading pre-compiled TurboVec indexes and pickled corpora from Hugging Face for instant retrieval.
3. FastEmbed: Generating mathematical vectors of the user's essays to match against the Ivy League database.
4. Llama.cpp: Initializing and binding the Gemma 4-E2B language model directly to the local GPU hardware.
5. Inference: Providing the `analyze_essay` (one-shot review) and `chat_with_viorra` (conversational loop) API endpoints.

NOTE: This file employs singletons (`_model_lock`, `is_loaded`) to ensure the massive AI models are only loaded into RAM/VRAM once across all threads.
"""

import os
import time
import re
import threading
import sys
from contextlib import contextmanager
import numpy as np


@contextmanager
def suppress_c_stderr():
    fd = sys.stderr.fileno()
    def_pad = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(def_pad, fd)
        os.close(def_pad)
        os.close(devnull)

# Suppress Hugging Face Symlink warning on Windows and internal deprecation warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"



# Globals for singleton loading
is_loaded = False
embedder = None
index = None
corpus_texts = []
corpus_feedback = []
llm_engine = None
boot_status_message = "Loading..."
_model_lock = threading.Lock()
_inference_lock = threading.Lock()

# Cache templates in memory to avoid disk I/O on every request
_cached_soul_content = None
_cached_evaluate_instructions = None
_cached_chat_prefix = None
_cached_chat_main = None
_cached_chat_suffix = None

def get_cached_prompts():
    global _cached_soul_content, _cached_evaluate_instructions, _cached_chat_prefix, _cached_chat_main, _cached_chat_suffix
    if _cached_soul_content is None:
        try:
            with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r", encoding="utf-8") as f:
                _cached_soul_content = f.read()
        except FileNotFoundError:
            _cached_soul_content = ""
            
    if _cached_evaluate_instructions is None:
        try:
            with open(os.path.join(os.path.dirname(__file__), "evaluate_instructions.txt"), "r", encoding="utf-8") as f:
                _cached_evaluate_instructions = f.read()
        except FileNotFoundError:
            _cached_evaluate_instructions = ""
            
    if _cached_chat_prefix is None or _cached_chat_main is None or _cached_chat_suffix is None:
        try:
            with open(os.path.join(os.path.dirname(__file__), "chat_instructions.txt"), "r", encoding="utf-8") as f:
                chat_content = f.read()
            # Split by section headers
            parts = chat_content.split("=== PREFIX ===")
            prefix_val = ""
            main_val = ""
            suffix_val = ""
            if len(parts) > 1:
                subparts = parts[1].split("=== MAIN ===")
                prefix_val = subparts[0].strip() + "\n\n"
                if len(subparts) > 1:
                    subparts2 = subparts[1].split("=== SUFFIX ===")
                    main_val = subparts2[0].strip()
                    if len(subparts2) > 1:
                        suffix_val = "\n\n" + subparts2[1].strip()
            _cached_chat_prefix = prefix_val
            _cached_chat_main = main_val
            _cached_chat_suffix = suffix_val
        except FileNotFoundError:
            _cached_chat_prefix = ""
            _cached_chat_main = ""
            _cached_chat_suffix = ""
            
    return _cached_soul_content, _cached_evaluate_instructions, _cached_chat_prefix, _cached_chat_main, _cached_chat_suffix

def _update_boot_status(msg):
    global boot_status_message
    boot_status_message = msg

def ensure_models_loaded():
    """Lazy-load the massive Llama.cpp LLM and TurboVec components into RAM."""
    global is_loaded, embedder, index, corpus_texts, corpus_feedback, llm_engine, boot_status_message
    
    with _model_lock:
        if is_loaded:
            return
            
        boot_status_message = "Initializing engine components..."
        
        # Monkeypatch global tqdm to broadcast FAISS database download progress to the frontend UI silently
        import tqdm
        _orig_tqdm = tqdm.tqdm
        class UItqdm(_orig_tqdm):
            def __init__(self, *args, **kwargs):
                # Force tqdm to print to the void so it doesn't ruin our terminal aesthetic
                kwargs['file'] = open(os.devnull, 'w')
                super().__init__(*args, **kwargs)
                
            def update(self, n=1):
                super().update(n)
                if hasattr(self, 'total') and self.total:
                    pct = int(self.n / self.total * 100)
                    global boot_status_message
                    boot_status_message = f"Downloading files... [{pct}%]"
        tqdm.tqdm = UItqdm
        import tqdm.auto
        tqdm.auto.tqdm = UItqdm
        import huggingface_hub.file_download
        huggingface_hub.file_download.tqdm = UItqdm
        
        from fastembed import TextEmbedding
        from viorra.hardware import profile_system_hardware, download_llm_native
    
        # --- 1. HARDWARE PROFILING & GPU ENFORCEMENT ---
        best_gpu_name, best_vram = profile_system_hardware()
        
        is_gpu_supported = False
        if best_gpu_name:
            gpu_name_upper = best_gpu_name.upper()
            if any(term in gpu_name_upper for term in ["NVIDIA", "AMD", "RADEON", "APPLE SILICON", "METAL"]):
                is_gpu_supported = True
                
        import llama_cpp
        if not is_gpu_supported or not getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)():
            raise RuntimeError(
                "VIORRA Error: No compatible GPU detected or llama-cpp-python was compiled without GPU offload support. "
                "To guarantee sub-second admissions feedback, VIORRA runs strictly in GPU acceleration mode. "
                "Running on CPU is disabled."
            )

        # --- 2. VECTOR DATABASE (RAG) LOADING ---
        import pickle
        from viorra.server import USER_DATA_DIR
        from viorra.hardware import download_file_native
        
        local_index_file = os.path.join(USER_DATA_DIR, "viorra_index_local.tv")
        local_corpus_file = os.path.join(USER_DATA_DIR, "viorra_corpus_local.pkl")
        
        if os.path.exists(local_index_file) and os.path.exists(local_corpus_file):
            index_file = local_index_file
            corpus_file = local_corpus_file
        else:
            index_file = os.path.join(USER_DATA_DIR, "viorra_index.tv")
            corpus_file = os.path.join(USER_DATA_DIR, "viorra_corpus.pkl")
            # Pull the pre-compiled database binaries natively
            download_file_native("https://huggingface.co/datasets/qsardor/viorra-admissions-essays/resolve/main/viorra_index.tv", index_file, status_callback=_update_boot_status)
            download_file_native("https://huggingface.co/datasets/qsardor/viorra-admissions-essays/resolve/main/viorra_corpus.pkl", corpus_file, status_callback=_update_boot_status)
        
        import turbovec
        index = turbovec.TurboQuantIndex.load(index_file)
        index.prepare() # Warm up the SIMD search caches
        
        with open(corpus_file, "rb") as f:
            corpus_texts, corpus_feedback = pickle.load(f)
            
        # FastEmbed is initialized strictly for embedding the user's live query, NOT the database.
        try:
            # Try offline first so it doesn't ping the internet and show 5 progress bars
            embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            import shutil
            import tempfile
            cache_dir = os.path.join(tempfile.gettempdir(), "fastembed_cache")
            if os.path.exists(cache_dir):
                
                shutil.rmtree(cache_dir, ignore_errors=True)
                
            from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
            disable_progress_bars()
            # [MEMORY PATCH]: Limit fastembed threads to 1 to prevent ONNX memory arena leaks under concurrency
            embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", threads=1)
            enable_progress_bars()

        # --- 3. LLAMA.CPP GEMMA 4 INITIALIZATION ---
        from viorra.server import USER_DATA_DIR
        llm_path = os.path.join(USER_DATA_DIR, "gemma-4-e2b-it.Q4_K_M.gguf")
        
        if not os.path.exists(llm_path):
            boot_status_message = "Downloading gemma-4-e2b-it.Q4_K_M.gguf..."
            from viorra.hardware import download_llm_native
            download_llm_native(llm_path, status_callback=_update_boot_status)
            
        # Restore original tqdm so we don't mess up other terminal apps
        huggingface_hub.file_download.tqdm = _orig_tqdm
        
        boot_status_message = "Starting AI Inference Engine (CUDA)..."
        
        from llama_cpp import Llama
        global llm_engine
        
        # We cap at 16384 context to avoid blowing up the 8GB RTX 4060 when MTP drafter heads are active.
        llm_engine = Llama(
            model_path=llm_path,
            n_ctx=16384,          
            n_gpu_layers=-1,      # Offload everything to GPU
            flash_attn=True,      # Fast attention
            n_threads=8,
            verbose=False
        )
        
        is_loaded = True
        boot_status_message = "Ready!"
        

def unload_models():
    """
    Called by the server's inactivity monitor.
    Drops the heavy C++ engine from GPU/RAM to prevent idle overheating and memory hoarding.
    """
    global is_loaded, embedder, index
    with _model_lock:
        if is_loaded:
            # Drop the heavy C++ engine from GPU/RAM to prevent idle overheating and memory hoarding.
            try:
                if llm_engine:
                    del llm_engine
                    llm_engine = None
            except Exception:
                pass
            
            embedder = None
            index = None
            import gc
            gc.collect()
            is_loaded = False




# --- CLAUDE ARCHITECTURE FIXES ---
import logging

BANNED_WORDS = [
    "delve", "testament", "intricate", "tapestry", "underscore",
    "crucial", "additionally", "actually", "vibrant", "breathtaking",
    "showcasing", "pivotal"
]

def audit_output(text: str) -> list:
    violations = [w for w in BANNED_WORDS if w in text.lower()]
    if violations:
        logging.warning(f"[VIORRA] Banned word violations: {violations}")
    return violations

ALLOWED_KEYS = {"quote", "feedback"}

def enforce_schema(raw_json: dict) -> dict:
    if "diagnostics" in raw_json:
        for item in raw_json["diagnostics"]:
            for k in list(item.keys()):
                if k not in ALLOWED_KEYS:
                    logging.warning(f"[VIORRA] Illegal JSON key stripped: {k}")
                    del item[k]
    return raw_json

def extract_response(raw_output: str) -> str:
    separators = ["<channel|>", "<|channel>thought", "[DIAGNOSIS]"]
    for sep in separators:
        if sep in raw_output:
            raw_output = raw_output.split(sep)[-1].strip()
    return raw_output

def check_hallucination(output: str, essay_text: str) -> bool:
    WHITELIST = {"VIORRA", "Ivy", "League", "Barnaby"}
    output_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', output))
    essay_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', essay_text))
    foreign = output_nouns - essay_nouns - WHITELIST
    if foreign:
        logging.warning(f"[VIORRA] Possible hallucination ΓÇö unknown nouns: {foreign}")
        return True
    return False

def build_prompt(template: str, mode: str, essay: str, rag: str, feedback: str) -> str:
    soul_content, evaluate_inst, chat_prefix, chat_main, chat_suffix = get_cached_prompts()
    
    prompt = (template
        .replace("[[MODE]]", mode)
        .replace("[[ESSAY_TEXT]]", essay)
        .replace("[[RAG_EXAMPLES]]", rag)
        .replace("[[PREVIOUS_FEEDBACK]]", feedback))

    if mode == "EVALUATE":
        prompt = prompt.replace("[[CHAT_PREFIX]]", "")
        prompt = prompt.replace("[[EVALUATE_INSTRUCTIONS]]", evaluate_inst)
        prompt = prompt.replace("[[CHAT_SUFFIX]]", "")
    elif mode == "CHAT":
        prompt = prompt.replace("[[CHAT_PREFIX]]", chat_prefix)
        prompt = prompt.replace("[[EVALUATE_INSTRUCTIONS]]", chat_main)
        prompt = prompt.replace("[[CHAT_SUFFIX]]", chat_suffix)
    assert "[[" not in prompt, "Unsubstituted placeholder remaining in prompt: " + str(re.findall(r'\[\[.*?\]\]', prompt))
    return prompt
# ---------------------------------

def analyze_essay(test_text: str, debug_mode: bool = False):
    with _inference_lock:
        return _analyze_essay_impl(test_text, debug_mode)

def _analyze_essay_impl(test_text: str, debug_mode: bool = False):
    """
    Main entry point for generating the "Admissions Mentor Summary".
    Executes the complete pipeline: RAG Search -> Persona Injection -> Llama.cpp Generation.
    """
    ensure_models_loaded()
    import os
    global llm_engine

    # --- SANITY CHECKS ---
    word_count = len(test_text.split())
    if word_count < 150:
        return {"error": f"Oops! Your essay is only {word_count} words. The Common App personal statement has a strict minimum of 150 words. Please check up and upload the correct essay."}
    if word_count > 1000:
        return {"error": f"Oops! Your essay is {word_count} words, which exceeds the strict 1,000-word maximum limit of VIORRA. Please edit it down before requesting Ivy League analysis."}
        

    start_time = time.time()
    
    # --- 1. RAG (RETRIEVAL AUGMENTED GENERATION) ---
    # FastEmbed converts the user's 500-word essay into a dense mathematical vector.
    query_embedding = np.array(list(embedder.embed([test_text])), dtype=np.float32)
    
    # Turbovec does a nearest-neighbor search through the thousands of Ivy League essays in microseconds
    distances, indices = index.search(query_embedding, 2) # Get top 2 most similar historical essays
    rag_examples_text = ""
    retrieved_docs = []
    
    for i, idx in enumerate(indices[0]):
        excerpt = corpus_texts[idx][:500]
        fb = corpus_feedback[idx]
        rag_examples_text += f"\n--- SIMILAR ADMISSIONS ESSAY {i+1} ---\n"
        rag_examples_text += "EXCERPT: " + excerpt + "...\n"
        rag_examples_text += "ADMISSIONS FEEDBACK: " + fb + "\n"
        
        retrieved_docs.append({
            "id": i+1,
            "excerpt": excerpt,
            "full_text": corpus_texts[idx],
            "feedback": fb
        })
        
    # --- 2. PROMPT CONSTRUCTION (IDENTITY INJECTION) ---
    soul_content, _, _, _, _ = get_cached_prompts()
    
    # We now strictly use the Claude SOUL architecture template
    sys_prompt = build_prompt(
        template=soul_content,
        mode="EVALUATE",
        essay=test_text,
        rag=rag_examples_text,
        feedback="None"
    )
    try:
        from viorra.memory_agent import load_memory
        current_memory = load_memory()
        if current_memory:
            memory_injection = "\n## PERMANENT USER MEMORY (KNOWLEDGE ABOUT THE USER):\n"
            for m in current_memory:
                memory_injection += f"- {m}\n"
            sys_prompt = memory_injection + "\n\n" + sys_prompt
    except Exception as e:
        pass

    # --- 3. NATIVE LLAMA.CPP INFERENCE ---
    # Manually construct Gemma 4 formatting to avoid template fragility
    raw_prompt = f"<|turn>system\n<|think|>\n{sys_prompt}<turn|>\n<|turn>user\nAnalyze my essay and provide your feedback.<turn|>\n<|turn>model\n"
    
    infer_start = time.time()
    response = llm_engine.create_completion(
        prompt=raw_prompt,
        max_tokens=2048,
        temperature=0.0
    )
    output_text = response["choices"][0]["text"]
    infer_end = time.time()
    infer_time = infer_end - infer_start
            
    # --- 4. JSON PARSING & RESPONSE ---
    import json_repair
    
    try:
        text_to_parse = extract_response(output_text)
        audit_output(text_to_parse)
        check_hallucination(text_to_parse, test_text)
        
        # 2. Try to extract markdown block first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text_to_parse, re.DOTALL)
        if json_match:
            parsed = enforce_schema(json_repair.loads(json_match.group(1)))
        else:
            # 3. Fallback: extract the outermost curly braces in case of raw output
            fallback_match = re.search(r'(\{.*\})', text_to_parse, re.DOTALL)
            if fallback_match:
                parsed = enforce_schema(json_repair.loads(fallback_match.group(1)))
            else:
                parsed = enforce_schema(json_repair.loads(text_to_parse))
                
        parsed["generation_time"] = round(time.time() - start_time, 2)
        parsed["raw_output"] = output_text
        parsed["retrieved_docs"] = retrieved_docs
        
        if debug_mode:
            import psutil
            import os
            out_words = len(output_text.split())
            tokens = out_words * 1.3
            tps = tokens / infer_time if infer_time > 0 else 0
            mem_info = psutil.Process(os.getpid()).memory_info()
            ram_mb = mem_info.rss / (1024 ** 2)
            
            parsed["benchmark"] = {
                "infer_time": round(infer_time, 2),
                "tokens_sec": round(tps, 2),
                "ram_mb": round(ram_mb, 2)
            }
            
        return parsed
    except Exception as e:
        
        return {"error": "Failed to parse JSON. Please check the terminal logs.", "raw_output": output_text, "retrieved_docs": retrieved_docs}

def chat_with_viorra(essay_text: str, previous_feedback: str, chat_history: list, new_message: str, retrieved_docs: list = None):
    with _inference_lock:
        return _chat_with_viorra_impl(essay_text, previous_feedback, chat_history, new_message, retrieved_docs)

def _chat_with_viorra_impl(essay_text: str, previous_feedback: str, chat_history: list, new_message: str, retrieved_docs: list = None):
    ensure_models_loaded()
    

    
    soul_content, _, _, _, _ = get_cached_prompts()
    
    rag_injection = "None"
    if retrieved_docs:
        rag_injection = "\n\n--- Reference: Successful Admissions Essays ---\n"
        for i, doc in enumerate(retrieved_docs):
            rag_injection += f"Example {i+1}:\n"
            rag_injection += f"Excerpt: {doc.get('excerpt', '')}\n"
            rag_injection += f"Feedback: {doc.get('feedback', '')}\n\n"
            
    chat_sys_prompt = build_prompt(
        template=soul_content,
        mode="CHAT",
        essay=essay_text,
        rag=rag_injection,
        feedback=previous_feedback
    )
    
    try:
        from viorra.memory_agent import load_memory
        current_memory = load_memory()
        if current_memory:
            memory_injection = "\n## PERMANENT USER MEMORY (KNOWLEDGE ABOUT THE USER):\n"
            for m in current_memory:
                memory_injection += f"- {m}\n"
            chat_sys_prompt = memory_injection + "\n\n" + chat_sys_prompt
    except Exception as e:
        pass

    # Build the conversation manually for Gemma 4 native format
    global llm_engine
    
    # [CONTEXT WINDOW PATCH]: Truncate chat history to the last 10 messages to prevent 128K VRAM OOM crashes on the GPU
    safe_chat_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
    
    raw_prompt = f"<|turn>system\n<|think|>\n{chat_sys_prompt}<turn|>\n"
    for msg in safe_chat_history:
        # DeepMind Guideline: Strip out internal <|channel>thought tokens to save context window space
        # and prevent constraint degradation over long conversations.
        clean_content = msg["content"]
        import re
        clean_content = re.sub(r'<\|channel\|?>thought.*?<channel\|>', '', clean_content, flags=re.DOTALL)
        clean_content = re.sub(r'<think>.*?</think>', '', clean_content, flags=re.DOTALL)
        clean_content = clean_content.strip()
        
        if msg["role"] == "user":
            raw_prompt += f"<|turn>user\n{clean_content}<turn|>\n"
        else:
            raw_prompt += f"<|turn>model\n{clean_content}<turn|>\n"
            
    raw_prompt += f"<|turn>user\n{new_message.strip()}<turn|>\n<|turn>model\n"
    
    try:
        response = llm_engine.create_completion(
            prompt=raw_prompt,
            max_tokens=1024,
            temperature=0.7,
            stop=["<turn|>", "<|turn>", "<|startoftext|>", "<|endoftext|>", "<|im_end|>"]
        )
        output_text = response["choices"][0]["text"]
        
        # Strip MTP Draft Channel Bleed and leaked chat/think tokens
        if "<channel|>" in output_text:
            output_text = output_text.split("<channel|>")[-1].strip()
        
        # Scrub any leaked internal tokens that bleed through on edge cases
        import re as _re
        output_text = _re.sub(r'<\|think\|?>.*?(<\|/think\|?>|$)', '', output_text, flags=_re.DOTALL)
        output_text = _re.sub(r'<\|startoftext\|>', '', output_text)
        output_text = _re.sub(r'<\|turn\|>.*', '', output_text, flags=_re.DOTALL)
        # Scrub markdown headers (#, ##, ###) to enforce natural human paragraph formatting
        output_text = _re.sub(r'^#+\s*', '', output_text, flags=_re.MULTILINE)
        output_text = output_text.strip()
        
        # [COMMUNITY RESEARCH APPLIED]: The Anti-Slop filter natively scrubs generic AI buzzwords
        # because 2B parameters will inevitably regress to training slop over long contexts.
        import re
        slop_map = {
            r"\bdelve\b": "explore",
            r"\btapestry\b": "structure",
            r"\btestament to\b": "proof of",
            r"\bmultifaceted\b": "complex",
            r"\bnuanced\b": "detailed",
            r"\bleverage\b": "use",
            r"\bactually\b": "in truth",
            r"\bcrucial\b": "essential",
            r"\badditionally\b": "also",
            r"\bintricate\b": "detailed",
            r"\bunderscore\b": "highlight",
            r"\bvibrant\b": "lively",
            r"\bbreathtaking\b": "striking",
            r"\bshowcasing\b": "showing",
            r"\bpivotal\b": "key"
        }
        for slop, replacement in slop_map.items():
            output_text = re.sub(slop, replacement, output_text, flags=re.IGNORECASE)
            
        return {"response": output_text.strip()}
    except Exception as e:
        import logging
        import traceback
        logging.error(f"[VIORRA CHAT ENGINE ERROR] {e}\n{traceback.format_exc()}")
        error_msg = str(e).lower()
        if "context" in error_msg or "token" in error_msg or "size" in error_msg or "exceed" in error_msg or "length" in error_msg or "allocate" in error_msg:
            return {"error": "CONTEXT_LIMIT_REACHED"}
        return {"error": "ENGINE_ERROR", "details": str(e)}
