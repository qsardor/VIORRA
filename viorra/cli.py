"""
VIORRA Command Line Interface.
Provides a clean, aesthetic boot sequence for students.
"""

import os
import sys
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def _ensure_dependencies():
    try:
        import uvicorn  # noqa: F401
        import rich  # noqa: F401
        import fastapi  # noqa: F401
        import faiss  # noqa: F401
    except ModuleNotFoundError as e:
        print(f"Missing dependency detected: {e}. Auto-installing VIORRA dependencies...")
        try:
            # Check if running in development mode from source
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if os.path.exists(os.path.join(project_root, "pyproject.toml")):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", project_root])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "viorra"])
            print("Dependencies installed successfully. Restarting VIORRA...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as install_err:
            print(f"Failed to auto-install dependencies: {install_err}")
            sys.exit(1)

_ensure_dependencies()

import argparse
import uvicorn
import webbrowser
import threading
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text  # noqa: F401

# Ensure 'viorra' package can be imported when running cli.py directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize console to use the original stdout, even if we redirect later
console = Console()

def setup_quiet_logging():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "2"
    
    from viorra.server import USER_DATA_DIR
    log_file = os.path.join(USER_DATA_DIR, "viorra_logs.txt")
    import logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
    
    try:
        from loguru import logger
        logger.remove()
        logger.add(log_file, level="INFO")
    except Exception:
        pass
    return log_file

def main():
    import os
    from viorra.server import USER_DATA_DIR
    import viorra
    version_file = os.path.join(USER_DATA_DIR, "version.txt")
    current_version = viorra.__version__
    
    if os.path.exists(USER_DATA_DIR):
        should_reset = False
        if os.path.exists(version_file):
            with open(version_file, "r") as f:
                saved_version = f.read().strip()
            if saved_version != current_version:
                should_reset = True
        else:
            sessions_dir = os.path.join(USER_DATA_DIR, "Sessions")
            if os.path.exists(sessions_dir) and len(os.listdir(sessions_dir)) > 0:
                should_reset = True
                
        if should_reset:
            import shutil
            try:
                shutil.rmtree(USER_DATA_DIR)
            except Exception:
                pass
                
        os.makedirs(os.path.join(USER_DATA_DIR, "Sessions"), exist_ok=True)
        with open(version_file, "w") as f:
            f.write(current_version)

    parser = argparse.ArgumentParser(description="VIORRA: The elite personal statement coach, running locally.")
    parser.add_argument("command", nargs="?", default="start", help="Command to run (default: start the web server)")
    parser.add_argument("--cli", type=str, help="Analyze an essay file directly via CLI without starting the server", metavar="FILE")
    parser.add_argument("--factory-reset", action="store_true", help="Delete all local user data and logs")
    parser.add_argument("--clear-cache", action="store_true", help="Purge HuggingFace and FastEmbed model caches to free up disk space")
    parser.add_argument("--status", action="store_true", help="Check system hardware compatibility for VIORRA")
    parser.add_argument("--update", action="store_true", help="Force redownload of the latest RAG database and models")
    parser.add_argument("--benchmark", action="store_true", help="Run a real-time hardware performance test for the AI engine")
    parser.add_argument("--test-raw", action="store_true", help="Run a direct diagnostic test against the raw GGUF model (bypasses middleware)")
    args = parser.parse_args()

    if args.factory_reset:
        from viorra.server import USER_DATA_DIR
        import shutil
        console.print(f"[bold red]WARNING: Initiating Factory Reset...[/bold red]")
        if os.path.exists(USER_DATA_DIR):
            try:
                shutil.rmtree(USER_DATA_DIR)
                console.print(f"[bold green]Successfully deleted local user data at {USER_DATA_DIR}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Failed to delete user data: {e}[/bold red]")
        else:
            console.print(f"[dim]No user data found at {USER_DATA_DIR}.[/dim]")
        sys.exit(0)

    if args.clear_cache:
        import shutil
        import tempfile
        console.print(f"[bold yellow]Purging AI model caches...[/bold yellow]")
        
        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        if os.path.exists(hf_cache):
            try:
                shutil.rmtree(hf_cache)
                console.print(f"[bold green]Cleared HuggingFace Hub Cache: {hf_cache}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Failed to clear HuggingFace cache: {e}[/bold red]")
        else:
            console.print(f"[dim]HuggingFace cache already empty.[/dim]")
            
        fe_cache = os.path.join(tempfile.gettempdir(), "fastembed_cache")
        if os.path.exists(fe_cache):
            try:
                shutil.rmtree(fe_cache)
                console.print(f"[bold green]Cleared FastEmbed Cache: {fe_cache}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Failed to clear FastEmbed cache: {e}[/bold red]")
        else:
            console.print(f"[dim]FastEmbed cache already empty.[/dim]")
            
        from viorra.server import USER_DATA_DIR
        native_llm = os.path.join(USER_DATA_DIR, "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf")
        if os.path.exists(native_llm):
            try:
                os.remove(native_llm)
                console.print(f"[bold green]Deleted Native LLM Engine: {native_llm}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Failed to delete Native LLM: {e}[/bold red]")
        else:
            console.print(f"[dim]No native LLM found to delete.[/dim]")
            
        sys.exit(0)

    if args.status:
        from viorra.hardware import profile_system_hardware
        console.print(f"[bold cyan]VIORRA Hardware Diagnostics[/bold cyan]")
        console.print(f"Scanning host system for LiteRT compatibility...\n")
        
        gpu, vram = profile_system_hardware()
        if gpu and vram > 0:
            console.print(f"[bold green]DETECTED GPU:[/bold green] {gpu}")
            console.print(f"[bold green]AVAILABLE VRAM:[/bold green] {vram / (1024**3):.2f} GB")
            console.print(f"\n[bold green]VERDICT: Hardware is fully compatible with GPU acceleration.[/bold green]")
        else:
            console.print(f"[bold red]DETECTED GPU:[/bold red] None found or unsupported.")
            console.print(f"\n[bold red]VERDICT: Hardware is incompatible. VIORRA requires a dedicated GPU (NVIDIA CUDA, AMD, or Apple Silicon) to run. CPU mode is disabled.[/bold red]")
        sys.exit(0)

    if args.benchmark:
        console.print(f"[bold cyan]=== VIORRA PERFORMANCE BENCHMARK ===[/bold cyan]")
        import psutil
        import numpy as np
        import viorra.engine
        
        console.print("\n[bold yellow][1] Starting Cold Boot (Loading Models to GPU/RAM)...[/bold yellow]")
        boot_start = time.time()
        viorra.engine.ensure_models_loaded()
        boot_end = time.time()
        console.print(f"[dim]-> Boot Time: {boot_end - boot_start:.2f} seconds[/dim]")
        
        test_essay = "My entire life changed when I decided to volunteer at the local animal shelter. At first, I was just looking for a way to complete my high school community service hours. I didn't care about animals. But when I met a three-legged dog named Barnaby, everything clicked. I spent 400 hours redesigning the shelter's adoption website, which increased their adoption rate by 50%. This experience taught me about empathy, digital marketing, and the power of giving back. I want to study Computer Science so I can build software that helps non-profits scale their impact globally."
        
        console.print("\n[bold yellow][2] Measuring RAG Pipeline (FastEmbed + TurboVec (SIMD))...[/bold yellow]")
        rag_start = time.time()
        query_embedding = np.array(list(viorra.engine.embedder.embed([test_essay])), dtype=np.float32)
        distances, indices = viorra.engine.index.search(query_embedding, 2)
        rag_end = time.time()
        console.print(f"[dim]-> RAG Execution Time: {rag_end - rag_start:.4f} seconds[/dim]")
        
        console.print("\n[bold yellow][3] Measuring LLM Inference (Gemma 4 Llama.cpp)...[/bold yellow]")
        soul_content, _, _, _, _ = viorra.engine.get_cached_prompts()
        sys_prompt = viorra.engine.build_prompt(
            template=soul_content,
            mode="EVALUATE",
            essay=test_essay,
            rag="DUMMY RAG EXAMPLES",
            feedback="None"
        )
        formatted_prompt = f"<|turn|>system\n{sys_prompt}<|turn|>user\nAnalyze my essay.<|turn|>model\n"
        
        infer_start = time.time()
        response = viorra.engine.llm_engine.create_completion(
            prompt=formatted_prompt,
            max_tokens=1024,
            temperature=0.0
        )
        output_text = response["choices"][0]["text"]
        infer_end = time.time()
        infer_time = infer_end - infer_start
        console.print(f"[dim]-> LLM Inference Time: {infer_time:.2f} seconds[/dim]")
        
        out_words = len(output_text.split())
        tokens = out_words * 1.3 
        console.print(f"[bold green]-> Speed: ~{tokens / infer_time:.2f} Tokens per second[/bold green]")
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        console.print(f"\n[bold yellow][4] System Memory Footprint:[/bold yellow]")
        console.print(f"[dim]-> Active RAM Usage: {mem_info.rss / (1024 ** 2):.2f} MB[/dim]")
        
        console.print("\n[bold cyan]=== BENCHMARK COMPLETE ===[/bold cyan]")
        sys.exit(0)

    if args.update:
        console.print(f"[bold cyan]Forcing Model Update...[/bold cyan]")
        console.print(f"Purging old cache and downloading latest indices from HuggingFace.\n")
        
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        from huggingface_hub import hf_hub_download
        import shutil
        
        # Purge the specific repo caches to prevent orphaned model bloat
        hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        repos_to_clear = [
            "datasets--qsardor--viorra-admissions-essays",
            "models--unsloth--gemma-4-E2B-it-qat-GGUF"
        ]
        
        for repo in repos_to_clear:
            repo_path = os.path.join(hf_cache_dir, repo)
            if os.path.exists(repo_path):
                try:
                    shutil.rmtree(repo_path)
                    console.print(f"[dim]Purged outdated cache: {repo}[/dim]")
                except Exception:
                    pass
        
        try:
            hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_index.tv", repo_type="dataset", force_download=True)
            hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_corpus.pkl", repo_type="dataset", force_download=True)
            console.print(f"[bold green]Successfully updated RAG Database.[/bold green]")
            
            from viorra.server import USER_DATA_DIR
            from viorra.hardware import download_llm_native
            native_llm = os.path.join(USER_DATA_DIR, "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf")
            if os.path.exists(native_llm):
                os.remove(native_llm)
            
            console.print(f"Downloading Native LLM Engine (~2.6GB)...")
            download_llm_native(native_llm)
            console.print(f"[bold green]Successfully updated GGUF Engine.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Update failed. Ensure you have an active internet connection. Error: {e}[/bold red]")
        sys.exit(0)

    if args.test_raw:
        console.print(f"[bold cyan]=== DIRECT LLAMA.CPP DIAGNOSTIC TEST ===[/bold cyan]")
        console.print("[dim]Bypassing Viorra middleware... connecting directly to native Llama instance.[/dim]\n")
        
        try:
            from viorra.server import USER_DATA_DIR
            import llama_cpp
            llm_path = os.path.join(USER_DATA_DIR, "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf")
            if not os.path.exists(llm_path):
                console.print(f"[bold red]ERROR: GGUF model not found at {llm_path}[/bold red]")
                sys.exit(1)
                
            llm = llama_cpp.Llama(model_path=llm_path, n_gpu_layers=-1, n_ctx=8192, verbose=False)

            def test(label, messages):
                console.print(f"\n[bold yellow]{'='*70}[/bold yellow]")
                console.print(f"[bold yellow]PROMPT {label}[/bold yellow]")
                raw_prompt = ""
                for m in messages:
                    console.print(f"  [bold white][{m['role'].upper()}]:[/bold white] {m['content']}")
                    raw_prompt += f"<|turn|>{m['role']}\n{m['content']}"
                raw_prompt += "<|turn|>model\n"
                
                console.print(f"[bold yellow]{'='*70}[/bold yellow]")
                try:
                    resp = llm.create_completion(prompt=raw_prompt, max_tokens=1024, temperature=0.0)
                    raw = resp["choices"][0]["text"]
                    console.print(f"\n[bold green][RAW GEMMA 4 RESPONSE]:[/bold green]\n{raw}")
                except Exception as e:
                    console.print(f"\n[bold red][ERROR]: {e}[/bold red]")

            test("1 (Empathy Test)", [
                {"role": "system", "content": "You are a warm, empathetic Ivy League admissions mentor."},
                {"role": "user",   "content": "I worked so hard on this essay and my counselor said it was bad. I feel like giving up."}
            ])

            test("2 (Persona Test)", [
                {"role": "system", "content": "You are Viorra, a deeply human AI mentor."},
                {"role": "user",   "content": "Who are you?"}
            ])

            test("3 (Essay Critique Test)", [
                {"role": "user", "content": 'Here is my college essay opening: "I have always been passionate about making a difference in the world." Please give me brutal, honest feedback.'}
            ])

        except ImportError as e:
            console.print(f"[bold red]Failed to load llama_cpp. Error: {e}[/bold red]")
            
        sys.exit(0)

    if args.cli:
        setup_quiet_logging()
        from viorra.engine import analyze_essay
        
        file_path = args.cli
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File '{file_path}' not found.")
            sys.exit(1)
            
        with open(file_path, "r", encoding="utf-8") as f:
            essay_text = f.read()
            
        console.print(f"[bold cyan]VIORRA Engine Booting...[/bold cyan]")
        console.print(f"Loading RAG database and Llama.cpp models to analyze {len(essay_text.split())} words...\n")
        
        start_time = time.time()
        result = analyze_essay(essay_text)
        elapsed = time.time() - start_time
        
        console.print(f"[bold green]--- ANALYSIS COMPLETE ({elapsed:.2f}s) ---[/bold green]\n")
        
        if "error" in result:
            console.print(f"[bold red]ERROR:[/bold red] {result['error']}")
        else:
            # Check for keys, handling fallback if prompt changed
            summary = result.get("mentor_summary") or result.get("The Verdict") or "N/A"
            console.print(Panel(summary, title="[bold yellow]THE VERDICT[/bold yellow]", border_style="yellow"))
            
            # Print raw output if we want to expose reasoning blocks
            raw = result.get("raw_output", "")
            # Check for both legacy <think> and native Gemma 4 channel reasoning
            think_block = ""
            if "<think>" in raw:
                think_block = raw.split("<think>")[1].split("</think>")[0].strip()
            elif "<|channel" in raw:
                import re
                match = re.search(r'<\|channel\|?>thought(.*?)<channel\|>', raw, re.DOTALL)
                if match:
                    think_block = match.group(1).strip()
            if think_block:
                console.print(Panel(think_block, title="[bold magenta]INTERNAL REASONING[/bold magenta]", border_style="magenta"))
            
            console.print("\n[bold cyan]RAG MATCHES (Ivy League Database)[/bold cyan]")
            for doc in result.get("retrieved_docs", []):
                console.print(f"\n[bold white]Match #{doc['id']}:[/bold white]")
                console.print(f"[dim]Excerpt:[/dim] {doc['excerpt']}...")
                console.print(f"[italic green]Feedback:[/italic green] {doc['feedback']}")
                
            # Enter interactive conversational loop
            console.print("\n[bold green]=== INTERACTIVE MENTOR SESSION ===[/bold green]")
            console.print("[dim]Ask Viorra questions about your essay or the feedback report. Type 'exit' or 'quit' to end the session.[/dim]\n")
            
            chat_history = []
            previous_feedback = summary
            retrieved_docs = result.get("retrieved_docs", [])
            
            from viorra.engine import chat_with_viorra
            
            while True:
                try:
                    user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["exit", "quit"]:
                        console.print("[bold yellow]Session ended. Good luck with your drafts![/bold yellow]")
                        break
                    
                    console.print("[dim]Thinking...[/dim]")
                    
                    # Run conversational inference
                    chat_result = chat_with_viorra(
                        essay_text=essay_text,
                        previous_feedback=previous_feedback,
                        chat_history=chat_history,
                        new_message=user_input,
                        retrieved_docs=retrieved_docs
                    )
                    
                    if "error" in chat_result:
                        console.print(f"[bold red]Error: {chat_result['error']}[/bold red]")
                    else:
                        response_text = chat_result.get("response", "No response received.")
                        console.print(f"\n[bold green]Viorra >[/bold green] {response_text}\n")
                        
                        # Update conversation history
                        chat_history.append({"role": "user", "content": user_input})
                        chat_history.append({"role": "assistant", "content": response_text})
                        
                        # Trigger background memory agent in a separate thread
                        try:
                            from viorra.memory_agent import run_memory_agent_async
                            import threading
                            threading.Thread(
                                target=run_memory_agent_async, 
                                args=(user_input, response_text), 
                                daemon=True
                            ).start()
                        except Exception:
                            pass
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[bold yellow]Session ended cleanly. Good luck with your drafts![/bold yellow]")
                    import os
                    os._exit(0)
                
        sys.exit(0)

    if args.command == "debug":
        console.print(f"[bold cyan]=== VIORRA AUTOMATED QA DIAGNOSTICS ===[/bold cyan]")
        import threading
        import requests
        import uvicorn
        
        def run_server():
            import logging
            uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
            uvicorn_log_config["loggers"]["uvicorn"]["level"] = "ERROR"
            uvicorn_log_config["loggers"]["uvicorn.access"]["level"] = "ERROR"
            uvicorn_log_config["loggers"]["uvicorn.error"]["level"] = "ERROR"
            uvicorn.run("viorra.server:app", host="127.0.0.1", port=8001, reload=False, log_config=uvicorn_log_config)
            
        console.print("[dim]Booting isolated diagnostic web server on port 8001...[/dim]")
        import threading
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        server_up = False
        for _ in range(60):
            try:
                if requests.get("http://127.0.0.1:8001/api/status", timeout=1).status_code == 200:
                    server_up = True
                    break
            except Exception:
                time.sleep(1)
                
        if not server_up:
            console.print("[bold red]FATAL: Diagnostic server failed to boot.[/bold red]")
            import os
            os._exit(1)
            
        console.print("[bold green]✔ Server Online.[/bold green]\n")
        
        test_essay = "If you told me I would be playing a sport called squash at 11 years old, I would call you crazy. But in seventh grade, I was at a new school 10 times bigger than my last one. I felt like a little fish in a big pond. I was quiet, withdrawn, and very introverted. A lot of the time, I stayed where I was comfortable. During the first week of school, a group of people visited the school and they introduced themselves as Squashbusters. At that time, I’d only heard of Squash once before, but I didn’t really know what it was. I was curious, so I decided to try it out. I didn’t know it then, but this sport would completely change my life. It forced me out of my shell, taught me the true meaning of discipline, and showed me that I was capable of pushing my physical and mental limits far beyond what I ever thought was possible. Through hours of grueling practice, I learned how to stand my ground and fight for every single point on the court."

        success = True
        
        # --- TEST 1: /api/analyze ---
        console.print("[bold yellow][1] Testing /api/analyze (JSON Generation & Schema Enforcement)...[/bold yellow]")
        try:
            start = time.time()
            response = requests.post("http://127.0.0.1:8001/api/analyze", json={"text": test_essay, "debug_mode": True})
            elapsed = time.time() - start
            res_data = response.json()
            
            if response.status_code != 200:
                console.print(f"[bold red]FAILED:[/bold red] HTTP {response.status_code}")
                console.print(response.text)
                success = False
            elif "error" in res_data:
                console.print(f"[bold red]FAILED:[/bold red] Analysis returned error: {res_data['error']}")
                success = False
            elif "diagnostics" not in res_data:
                console.print("[bold red]FAILED:[/bold red] Missing structured JSON keys in response.")
                success = False
            else:
                console.print(f"[bold green]✔ Passed[/bold green] [dim]({elapsed:.2f}s)[/dim]")
                console.print(f"[dim]Mentor Summary: {res_data['mentor_summary'][:100]}...[/dim]\n")
        except Exception as e:
            console.print(f"[bold red]FAILED:[/bold red] Exception {e}")
            success = False
            
        # --- TEST 2: /api/chat ---
        console.print("[bold yellow][2] Testing /api/chat (Conversational Formatting & Banned Words)...[/bold yellow]")
        try:
            payload = {
                "session_id": "debug_123",
                "essay_text": test_essay,
                "previous_feedback": "None",
                "chat_history": [{"role": "user", "content": "Hello! I am ready to improve."}],
                "new_message": "Yo what's up bro?",
                "retrieved_docs": []
            }
            start = time.time()
            response = requests.post("http://127.0.0.1:8001/api/chat", json=payload)
            elapsed = time.time() - start
            res_data = response.json()
            
            if response.status_code != 200:
                console.print(f"[bold red]FAILED:[/bold red] HTTP {response.status_code}")
                console.print(response.text)
                success = False
            elif "error" in res_data:
                console.print(f"[bold red]FAILED:[/bold red] Chat returned error: {res_data['error']}")
                success = False
            else:
                reply = res_data.get("response", "")
                console.print(f"[bold green]✔ Passed[/bold green] [dim]({elapsed:.2f}s)[/dim]")
                console.print(f"[dim]Viorra: {reply[:150]}...[/dim]\n")
                
                if "{" in reply and "mentor_summary" in reply:
                    console.print("[bold red]FAILED:[/bold red] Chatbot outputted JSON schema instead of natural conversation!")
                    success = False
                    
                banned_words = ["delve", "testament", "intricate", "tapestry", "underscore", "crucial", "additionally", "actually", "vibrant", "breathtaking", "showcasing", "pivotal"]
                for word in banned_words:
                    if word in reply.lower():
                        console.print(f"[bold red]FAILED:[/bold red] Hallucinated banned word '{word}' in chat mode!")
                        success = False
        except Exception as e:
            console.print(f"[bold red]FAILED:[/bold red] Exception {e}")
            success = False
            
        if success:
            console.print("\n[bold green]=== ALL INTEGRATION TESTS PASSED: VIORRA IS 100% STABLE ===[/bold green]")
            import os
            os._exit(0)
        else:
            console.print("\n[bold red]=== INTEGRATION TESTS FAILED ===[/bold red]")
            import os
            os._exit(1)

    elif args.command == "start":
        setup_quiet_logging()
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        banner = """
██╗   ██╗██╗ ██████╗ ██████╗ ██████╗  █████╗ 
██║   ██║██║██╔═══██╗██╔══██╗██╔══██╗██╔══██╗
██║   ██║██║██║   ██║██████╔╝██████╔╝███████║
╚██╗ ██╔╝██║██║   ██║██╔══██╗██╔══██╗██╔══██║
 ╚████╔╝ ██║╚██████╔╝██║  ██║██║  ██║██║  ██║
  ╚═══╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝"""
        
        from rich.align import Align
        from rich.table import Table
        from rich.box import DOUBLE_EDGE
        
        # Create a beautiful bordered table layout
        table = Table(box=DOUBLE_EDGE, border_style="bold green", show_header=False, width=70)
        
        header_text = Text(banner, style="bold green", justify="center")
        subtitle = Text("Ivy League Admissions Engine", style="italic bright_white", justify="center")
        
        table.add_row(Align.center(header_text))
        table.add_row(Align.center(subtitle))
        
        console.print("\n")
        console.print(Align.center(table))
        console.print("\n")

        from viorra.server import USER_DATA_DIR
        boot_flag = os.path.join(USER_DATA_DIR, ".first_boot_done")
        
        # Enforce GPU-only mode
        from viorra.hardware import profile_system_hardware
        gpu_name, vram = profile_system_hardware()
        
        is_gpu_supported = False
        if gpu_name:
            gpu_name_upper = gpu_name.upper()
            if any(term in gpu_name_upper for term in ["NVIDIA", "AMD", "RADEON", "APPLE SILICON", "METAL"]):
                is_gpu_supported = True
                
        if not is_gpu_supported or vram < (2 * 1024**3):
            console.print(Align.center("\n[bold red]FATAL ERROR: NO COMPATIBLE DEDICATED GPU DETECTED[/bold red]"))
            console.print(Align.center("[red]VIORRA requires a dedicated GPU (NVIDIA CUDA, AMD, or Apple Silicon) to run. CPU mode is disabled.[/red]\n"))
            sys.exit(1)
            
        if not os.path.exists(boot_flag):
            os.makedirs(USER_DATA_DIR, exist_ok=True)
            with open(boot_flag, "w") as f:
                f.write("done")

        def wait_for_enter():
            import urllib.request
            
            while True:
                try:
                    # Use a long timeout so we don't abort the connection while Uvicorn is busy loading the model
                    urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=15)
                    break
                except Exception:
                    time.sleep(0.5)
            
            console.print(Align.center("[bold green]✔ VIORRA ONLINE[/bold green]"))
            console.print(Align.center("[dim]Press CTRL+C to safely shutdown[/dim]\n"))
            
            try:
                console.print(Align.center("[bold white on green]  PRESS ENTER TO OPEN IN BROWSER  [/bold white on green]"))
                if sys.stdin.isatty():
                    input("")
                    webbrowser.open("http://127.0.0.1:8000/")
            except KeyboardInterrupt:
                import os
                os._exit(0)
            except EOFError:
                pass
            except Exception:
                pass
            
        import threading
        threading.Thread(target=wait_for_enter, daemon=True).start()
        
        import logging  # noqa: F401
        import uvicorn
        uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
        uvicorn_log_config["loggers"]["uvicorn"]["level"] = "WARNING"
        uvicorn_log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
        uvicorn_log_config["loggers"]["uvicorn.error"]["level"] = "ERROR"
        
        # Suppress standard print output from random libraries, but keep console.print working
        # by letting uvicorn handle its own logs via the config above.
        
        uvicorn.run("viorra.server:app", host="127.0.0.1", port=8000, reload=False, log_config=uvicorn_log_config, log_level="error")
    else:
        console.print(f"[bold yellow]Unknown command: {args.command}[/bold yellow]")

if __name__ == "__main__":
    main()
