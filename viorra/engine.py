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
import json
import re
import threading
import sys
from contextlib import contextmanager
import numpy as np
from rich.console import Console
import faiss

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

console = Console()

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


def profile_system_hardware():
    """
    Scans the host system to determine the most powerful available GPU and its VRAM.
    This bypasses standard PyTorch/CUDA checks and queries the OS natively (WMI for Windows, 
    sysctl for Mac, lspci for Linux) to ensure LiteRT WebGPU can latch onto the correct hardware.
    
    Returns:
        tuple: (best_gpu_name (str), best_vram_in_bytes (int))
    """
    import platform
    import subprocess
    
    system = platform.system()
    best_gpu_name = None
    best_vram = 0
    
    # [WINDOWS NATIVE PROFILING]
    if system == "Windows":
        try:
            # Poll WMI for all VideoControllers to find dedicated GPUs
            out = subprocess.check_output(
                ['powershell', '-Command', 'Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            if out:
                data = json.loads(out)
                if isinstance(data, dict):
                    data = [data]
                    
                for gpu in data:
                    name = gpu.get("Name", "")
                    ram = gpu.get("AdapterRAM") or 0
                    
                    # WMI caps AdapterRAM at 4GB (32-bit legacy integer limit).
                    # If it's an NVIDIA card, we bypass WMI and directly query nvidia-smi for true VRAM.
                    if "NVIDIA" in name.upper():
                        try:
                            smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW).decode().strip()
                            for mem_line in smi.splitlines():
                                true_vram = int(mem_line.strip()) * (1024**2)
                                if true_vram > ram:
                                    ram = true_vram
                        except Exception:
                            pass
                            
                    if ram > best_vram:
                        best_vram = ram
                        best_gpu_name = name
        except Exception:
            pass
            
    # [MACOS NATIVE PROFILING]
    elif system == "Darwin":
        try:
            # Apple Silicon Unified Memory is shared, so we query total system RAM
            out = subprocess.check_output(['sysctl', 'hw.memsize'], stderr=subprocess.DEVNULL).decode().strip()
            if "hw.memsize:" in out:
                ram = int(out.split(":")[1].strip())
                best_vram = ram
                best_gpu_name = "Apple Silicon GPU (Unified)"
        except Exception:
            pass
            
    # [LINUX NATIVE PROFILING]
    elif system == "Linux":
        try:
            # Use lspci to find VGA compatible controllers
            out = subprocess.check_output('lspci | grep -i "vga\\|3d\\|display"', shell=True, stderr=subprocess.DEVNULL).decode().strip()
            if out:
                for line in out.splitlines():
                    line_upper = line.upper()
                    if "NVIDIA" in line_upper:
                        best_gpu_name = "NVIDIA GPU (Linux)"
                        # Query nvidia-smi for precise VRAM allocation
                        try:
                            smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL).decode().strip()
                            for mem_line in smi.splitlines():
                                vram = int(mem_line.strip()) * (1024**2)
                                if vram > best_vram:
                                    best_vram = vram
                        except Exception:
                            best_vram = max(best_vram, 1)
                        break
                    elif "AMD" in line_upper or "RADEON" in line_upper:
                        best_gpu_name = "AMD GPU (Linux)"
                        best_vram = max(best_vram, 1)
                        break
        except Exception:
            pass
            
    return best_gpu_name, best_vram

def ensure_models_loaded():
    """
    Singleton loader for the core AI engines.
    1. Detects hardware and spins up LiteRT WebGPU backend.
    2. Downloads the pre-compiled FAISS RAG database from Hugging Face.
    3. Mounts the Gemma 4 LLM into VRAM.
    """
    global is_loaded, embedder, index, corpus_texts, corpus_feedback, llm_engine, boot_status_message
    
    with _model_lock:
        if is_loaded:
            return

        boot_status_message = "Loading..."
    
        import litert_lm
        from datasets import load_dataset
        from fastembed import TextEmbedding
        from huggingface_hub import hf_hub_download
    
        # --- 1. HARDWARE PROFILING & BACKEND SELECTION ---
        best_gpu_name, best_vram = profile_system_hardware()
    
        if best_gpu_name and best_vram > 0:
            vram_gb = best_vram / (1024**3)
            backend_choice = "GPU"
            backend = litert_lm.interfaces.GPU() # Connects via WebGPU/DirectX12
        else:
            
            backend_choice = "CPU"
            backend = litert_lm.interfaces.CPU()

        # --- 2. VECTOR DATABASE (RAG) LOADING ---
        import hashlib
        import pickle
        
        from huggingface_hub import hf_hub_download
        
        # Pull the pre-compiled database binaries, bypassing the massive local compilation tax
        try:
            index_file = hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_faiss.index", repo_type="dataset")
            corpus_file = hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_corpus.pkl", repo_type="dataset")
        except Exception:
            # Offline fallback if disconnected
            index_file = hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_faiss.index", repo_type="dataset", local_files_only=True)
            corpus_file = hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_corpus.pkl", repo_type="dataset", local_files_only=True)
        
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

        # --- 3. LITERT GEMMA 4 INITIALIZATION ---
        boot_status_message = "Checking for updates..."
        try:
            # Check HuggingFace for new model hashes
            llm_path = hf_hub_download(repo_id="litert-community/gemma-4-E2B-it-litert-lm", filename="gemma-4-E2B-it.litertlm")
        except Exception:
            # Offline fallback if disconnected
            
            llm_path = hf_hub_download(repo_id="litert-community/gemma-4-E2B-it-litert-lm", filename="gemma-4-E2B-it.litertlm", local_files_only=True)
        
        boot_status_message = "Loading..."
        
        # Maps the Gemma 4 neural network directly to hardware
        with suppress_c_stderr():
            llm_engine = litert_lm.Engine(llm_path, backend=backend)
    
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



def analyze_essay(test_text: str):
    with _inference_lock:
        return _analyze_essay_impl(test_text)

def _analyze_essay_impl(test_text: str):
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
        
    try:
        from langdetect import detect
        lang = detect(test_text)
        if lang != 'en':
            return {"error": "Oops! VIORRA currently only supports essays written in English. Please upload an English personal statement."}
    except Exception:
        return {"error": "Oops! We couldn't recognize the text you uploaded. It appears to be gibberish or invalid characters. Please paste a real essay."}
        
    # --- UK/UCAS REJECTION ---
    uk_keywords = ["ucas", "a-levels", "a levels", "gcse", "bmat", "ukcat", "oxbridge"]
    test_text_lower = test_text.lower()
    for kw in uk_keywords:
        if kw in test_text_lower:
            return {"error": "Oops! VIORRA's database is currently trained exclusively on US Ivy League and Common App essays. UK (UCAS) support will be available in a future update once our dataset expands."}
            
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
    try:
        with open(get_sys_prompt_path(), "r", encoding="utf-8") as f:
            sys_prompt = f.read()
    except FileNotFoundError:
        sys_prompt = "[STUDENT ESSAY]\n\"[[TEST_TEXT]]\"\n[INSTRUCTIONS]\nOutput empty JSON."
        
    try:
        # Prepend the SOUL.md file to fundamentally warp the model's persona
        soul_path = os.path.join(os.path.dirname(__file__), "SOUL.md")
        with open(soul_path, "r", encoding="utf-8") as f:
            soul_content = f.read()
            sys_prompt = soul_content + "\n\n" + sys_prompt
    except FileNotFoundError:
        pass

    # Dynamic injection of student identity for conversational realism
    sys_prompt += "\n\nCRITICAL DIRECTIVE: You are talking directly to the student. Always address them generally as 'you' and 'your'. Example of GOOD phrasing: 'Your essay...' or 'You need to fix...'. Example of BAD phrasing: 'The student's essay demonstrates...' or 'The student writes...'. NEVER refer to them in the third person."
    
    sys_prompt = sys_prompt.replace("[[TEST_TEXT]]", test_text)
    sys_prompt = sys_prompt.replace("[[RAG_EXAMPLES]]", rag_examples_text)
    
    # --- 3. LITERT INFERENCE ---
    formatted_prompt = f"<|turn|>user\n{sys_prompt}<turn|>\n<|turn|>model\n"
    with llm_engine.create_conversation() as conversation:
        response = conversation.send_message(formatted_prompt)
        output_text = response["content"][0]["text"]
            
    # --- 4. JSON PARSING & RESPONSE ---
    import json_repair
    
    try:
        # 1. Strip reasoning blocks out entirely
        text_to_parse = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
        
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
        return parsed
    except Exception as e:
        
        return {"error": "Failed to parse JSON. Please check the terminal logs.", "raw_output": output_text, "retrieved_docs": retrieved_docs}

def chat_with_viorra(essay_text: str, previous_feedback: str, chat_history: list, new_message: str, retrieved_docs: list = None):
    with _inference_lock:
        return _chat_with_viorra_impl(essay_text, previous_feedback, chat_history, new_message, retrieved_docs)

def _chat_with_viorra_impl(essay_text: str, previous_feedback: str, chat_history: list, new_message: str, retrieved_docs: list = None):
    ensure_models_loaded()
    

    
    # Construct a stateful prompt manually if we don't have true conversational state memory in the wrapper
    try:
        with open(get_chat_prompt_path(), "r", encoding="utf-8") as f:
            chat_sys_prompt = f.read()
    except FileNotFoundError:
        chat_sys_prompt = "You are VIORRA, an Ivy League Admissions Coach."

    try:
        soul_path = os.path.join(os.path.dirname(__file__), "SOUL.md")
        with open(soul_path, "r", encoding="utf-8") as f:
            soul_content = f.read()
            chat_sys_prompt = soul_content + "\n\n" + chat_sys_prompt
    except FileNotFoundError:
        pass
        
    chat_sys_prompt = chat_sys_prompt.replace("[[ESSAY_TEXT]]", essay_text)
    chat_sys_prompt = chat_sys_prompt.replace("[[PREVIOUS_FEEDBACK]]", previous_feedback)
    
    if retrieved_docs:
        rag_injection = "\n\n--- Reference: Successful Admissions Essays ---\n"
        for i, doc in enumerate(retrieved_docs):
            rag_injection += f"Example {i+1}:\n"
            rag_injection += f"Excerpt: {doc.get('excerpt', '')}\n"
            rag_injection += f"Feedback: {doc.get('feedback', '')}\n\n"
        chat_sys_prompt += rag_injection

    # Build the conversation payload manually using Gemma 4 Control Tokens
    full_prompt = f"<|turn|>system\n{chat_sys_prompt}<turn|>\n"
    for msg in chat_history:
        role_token = "user" if msg['role'] == "user" else "model"
        full_prompt += f"<|turn|>{role_token}\n{msg['content']}<turn|>\n"
            
    full_prompt += f"<|turn|>user\n{new_message}<turn|>\n<|turn|>model\n"
    
    try:
        with llm_engine.create_conversation() as conversation:
            response = conversation.send_message(full_prompt)
            output_text = response["content"][0]["text"]
            
        return {"response": output_text.strip()}
    except Exception as e:
        error_msg = str(e).lower()
        if "context" in error_msg or "token" in error_msg or "size" in error_msg or "exceed" in error_msg or "length" in error_msg or "allocate" in error_msg:
            return {"error": "CONTEXT_LIMIT_REACHED"}
        return {"error": "ENGINE_ERROR", "details": str(e)}
