"""
VIORRA CORE ENGINE
------------------
This module is the absolute heart of VIORRA. It handles:
1. Hardware Profiling: Natively querying the OS (WMI/sysctl/lspci/nvidia-smi) to find the best GPU backend.
2. Vector Database (RAG): Downloading and loading pre-compiled FAISS indexes and pickled corpora from Hugging Face for instant retrieval.
3. FastEmbed: Generating mathematical vectors of the user's essays to match against the Ivy League database.
4. LiteRT (TensorFlow Lite C++): Initializing and binding the Gemma 4-E2B language model directly to the local GPU hardware via WebGPU/DirectX12.
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

def get_sys_prompt_path():
    return os.path.join(os.path.dirname(__file__), "system_prompt.txt")

def get_chat_prompt_path():
    return os.path.join(os.path.dirname(__file__), "chat_prompt.txt")

# Cache prompts in memory to avoid disk I/O on every request
_cached_sys_prompt = None
_cached_chat_sys_prompt = None
_cached_soul_content = None

def get_cached_prompts():
    global _cached_sys_prompt, _cached_chat_sys_prompt, _cached_soul_content
    if _cached_soul_content is None:
        try:
            with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r", encoding="utf-8") as f:
                _cached_soul_content = f.read()
        except FileNotFoundError:
            _cached_soul_content = ""
            
    if _cached_sys_prompt is None:
        try:
            with open(get_sys_prompt_path(), "r", encoding="utf-8") as f:
                _cached_sys_prompt = f.read()
        except FileNotFoundError:
            _cached_sys_prompt = "[STUDENT ESSAY]\n\"[[TEST_TEXT]]\"\n[INSTRUCTIONS]\nOutput empty JSON."
            
    if _cached_chat_sys_prompt is None:
        try:
            with open(get_chat_prompt_path(), "r", encoding="utf-8") as f:
                _cached_chat_sys_prompt = f.read()
        except FileNotFoundError:
            _cached_chat_sys_prompt = "You are VIORRA, an Ivy League Admissions Coach."
            
    return _cached_sys_prompt, _cached_chat_sys_prompt, _cached_soul_content

def _update_boot_status(msg):
    global boot_status_message
    boot_status_message = msg

def ensure_models_loaded():
    """Lazy-load the massive LiteRT LLM and FAISS components into RAM."""
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
        
        import faiss
        from fastembed import TextEmbedding
        from huggingface_hub import hf_hub_download
        from viorra.hardware import profile_system_hardware, download_llm_native
    
        # --- 1. HARDWARE PROFILING & BACKEND SELECTION ---
        best_gpu_name, best_vram = profile_system_hardware()
    
        # Backend selection is deferred to _instantiate_engine_with_autofix()

        # --- 2. VECTOR DATABASE (RAG) LOADING ---
        import pickle
        
        # Pull the pre-compiled database binaries, bypassing the massive local compilation tax
        try:
            # Try offline mode first to prevent 15-second timeout delays
            index_file = hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_faiss.index", repo_type="dataset", local_files_only=True)
            corpus_file = hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_corpus.pkl", repo_type="dataset", local_files_only=True)
        except Exception:
            # If local files are missing, we must be online to download them
            index_file = hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_faiss.index", repo_type="dataset")
            corpus_file = hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_corpus.pkl", repo_type="dataset")
        
        index = faiss.read_index(index_file)
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
            embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            enable_progress_bars()

        # --- 3. LITERT GEMMA 4 INITIALIZATION (NATIVE DOWNLOAD) ---
        from viorra.server import USER_DATA_DIR
        llm_path = os.path.join(USER_DATA_DIR, "gemma-4-E2B-it.litertlm")
        
        # Check if the file is completely downloaded using the exact content-length
        expected_size = 2588147712
        if not os.path.exists(llm_path) or os.path.getsize(llm_path) != expected_size:
            boot_status_message = "Starting download..."
            download_llm_native(llm_path, status_callback=_update_boot_status)
        
        # Restore original tqdm so we don't mess up other terminal apps
        huggingface_hub.file_download.tqdm = _orig_tqdm
        
        boot_status_message = "Loading..."
        
        # Create the high-performance C++ inference engine using native DirectX/Vulkan backends
        import litert_lm
        try:
            litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        except Exception:
            pass

        def _instantiate_engine_with_autofix():
            backend = litert_lm.Backend.GPU() if best_gpu_name else litert_lm.Backend.CPU()
            try:
                return litert_lm.Engine(str(llm_path), backend=backend)
            except Exception as e:
                # Auto-Fixer: If LiteRT crashes due to corrupted WebGPU cache (driver update, hard reboot, etc.)
                import glob
                import os
                error_str = str(e).lower()
                if "cache" in error_str or "invalid argument" in error_str or "delegate" in error_str or "failed to create" in error_str:
                    global boot_status_message
                    boot_status_message = "Corrupted cache detected. Auto-fixing..."
                    cache_dir = os.path.dirname(llm_path)
                    for cache_file in glob.glob(os.path.join(cache_dir, "*.bin")):
                        try:
                            os.remove(cache_file)
                        except Exception:
                            pass
                    # Retry instantiation after wiping the corrupted caches
                    boot_status_message = "Rebuilding engine cache..."
                    return litert_lm.Engine(str(llm_path), backend=backend)
                raise e

        llm_engine = _instantiate_engine_with_autofix()
    
        is_loaded = True
        boot_status_message = "Ready!"
        

def unload_models():
    """
    Called by the server's inactivity monitor.
    Drops the heavy C++ engine from GPU/RAM to prevent idle overheating and memory hoarding.
    """
    global is_loaded, llm_engine, embedder, index
    with _model_lock:
        if is_loaded:
            
            llm_engine = None
            embedder = None
            index = None
            import gc
            gc.collect()
            is_loaded = False



def analyze_essay(test_text: str, debug_mode: bool = False):
    with _inference_lock:
        return _analyze_essay_impl(test_text, debug_mode)

def _analyze_essay_impl(test_text: str, debug_mode: bool = False):
    """
    Main entry point for generating the "Admissions Mentor Summary".
    Executes the complete pipeline: RAG Search -> Persona Injection -> LiteRT Generation.
    """
    ensure_models_loaded()
    
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
    
    # FAISS does a nearest-neighbor search through the thousands of Ivy League essays in milliseconds
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
    sys_prompt, _, soul_content = get_cached_prompts()
    if soul_content:
        sys_prompt = soul_content + "\n\n" + sys_prompt

    # Dynamic injection of student identity for conversational realism
    sys_prompt += "\n\nCRITICAL DIRECTIVE: You are talking directly to the student. Always address them generally as 'you' and 'your'. Example of GOOD phrasing: 'Your essay...' or 'You need to fix...'. Example of BAD phrasing: 'The student's essay demonstrates...' or 'The student writes...'. NEVER refer to them in the third person."
    
    sys_prompt = sys_prompt.replace("[[TEST_TEXT]]", test_text)
    sys_prompt = sys_prompt.replace("[[RAG_EXAMPLES]]", rag_examples_text)
    
    # --- 3. LITERT INFERENCE (NATIVE GEMMA 4 FORMAT) ---
    import litert_lm
    system_messages = [litert_lm.Message.system(sys_prompt)]
    
    infer_start = time.time()
    with llm_engine.create_conversation(messages=system_messages) as conversation:
        response = conversation.send_message("Analyze my essay and provide your feedback.")
        output_text = response["content"][0]["text"]
    infer_end = time.time()
    infer_time = infer_end - infer_start
            
    # --- 4. JSON PARSING & RESPONSE ---
    import json_repair
    
    try:
        # 1. Strip reasoning blocks out entirely (both legacy <think> and native Gemma 4 channel tokens)
        text_to_parse = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
        text_to_parse = re.sub(r'<\|channel\|?>thought.*?<channel\|>', '', text_to_parse, flags=re.DOTALL)
        text_to_parse = text_to_parse.strip()
        
        # 2. Try to extract markdown block first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text_to_parse, re.DOTALL)
        if json_match:
            parsed = json_repair.loads(json_match.group(1))
        else:
            # 3. Fallback: extract the outermost curly braces in case of raw output
            fallback_match = re.search(r'(\{.*\})', text_to_parse, re.DOTALL)
            if fallback_match:
                parsed = json_repair.loads(fallback_match.group(1))
            else:
                parsed = json_repair.loads(text_to_parse)
                
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
    

    
    # Construct a stateful prompt using cached memory
    _, chat_sys_prompt, soul_content = get_cached_prompts()
    if soul_content:
        chat_sys_prompt = soul_content + "\n\n" + chat_sys_prompt
        
    chat_sys_prompt = chat_sys_prompt.replace("[[ESSAY_TEXT]]", essay_text)
    chat_sys_prompt = chat_sys_prompt.replace("[[PREVIOUS_FEEDBACK]]", previous_feedback)
    
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
    
    if retrieved_docs:
        rag_injection = "\n\n--- Reference: Successful Admissions Essays ---\n"
        for i, doc in enumerate(retrieved_docs):
            rag_injection += f"Example {i+1}:\n"
            rag_injection += f"Excerpt: {doc.get('excerpt', '')}\n"
            rag_injection += f"Feedback: {doc.get('feedback', '')}\n\n"
        chat_sys_prompt += rag_injection

    # Build the conversation using Gemma 4's native system role and message history
    import litert_lm
    system_messages = [litert_lm.Message.system(chat_sys_prompt)]
    for msg in chat_history:
        if msg['role'] == 'user':
            system_messages.append(litert_lm.Message.user(msg['content']))
        else:
            system_messages.append(litert_lm.Message.model(litert_lm.Contents.of(msg['content'])))
    
    try:
        with llm_engine.create_conversation(messages=system_messages) as conversation:
            response = conversation.send_message(new_message)
            output_text = response["content"][0]["text"]
            
        return {"response": output_text.strip()}
    except Exception as e:
        error_msg = str(e).lower()
        if "context" in error_msg or "token" in error_msg or "size" in error_msg or "exceed" in error_msg or "length" in error_msg or "allocate" in error_msg:
            return {"error": "CONTEXT_LIMIT_REACHED"}
        return {"error": "ENGINE_ERROR", "details": str(e)}
