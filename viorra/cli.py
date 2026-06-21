"""
VIORRA Command Line Interface.
Provides a clean, aesthetic boot sequence for students.
"""

import os
import sys
import subprocess

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
        native_llm = os.path.join(USER_DATA_DIR, "gemma-4-E2B-it.litertlm")
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
            console.print(f"\n[bold green]VERDICT: Hardware is fully compatible with WebGPU acceleration.[/bold green]")
        else:
            console.print(f"[bold red]DETECTED GPU:[/bold red] None found or unsupported.")
            console.print(f"\n[bold yellow]VERDICT: System will fallback to CPU processing (slower performance).[/bold yellow]")
        sys.exit(0)

    if args.benchmark:
        console.print(f"[bold cyan]=== VIORRA PERFORMANCE BENCHMARK ===[/bold cyan]")
        import psutil
        import time
        import numpy as np
        import viorra.engine
        
        console.print("\n[bold yellow][1] Starting Cold Boot (Loading Models to GPU/RAM)...[/bold yellow]")
        boot_start = time.time()
        viorra.engine.ensure_models_loaded()
        boot_end = time.time()
        console.print(f"[dim]-> Boot Time: {boot_end - boot_start:.2f} seconds[/dim]")
        
        test_essay = "My entire life changed when I decided to volunteer at the local animal shelter. At first, I was just looking for a way to complete my high school community service hours. I didn't care about animals. But when I met a three-legged dog named Barnaby, everything clicked. I spent 400 hours redesigning the shelter's adoption website, which increased their adoption rate by 50%. This experience taught me about empathy, digital marketing, and the power of giving back. I want to study Computer Science so I can build software that helps non-profits scale their impact globally."
        
        console.print("\n[bold yellow][2] Measuring RAG Pipeline (FastEmbed + FAISS)...[/bold yellow]")
        rag_start = time.time()
        query_embedding = np.array(list(viorra.engine.embedder.embed([test_essay])), dtype=np.float32)
        distances, indices = viorra.engine.index.search(query_embedding, 2)
        rag_end = time.time()
        console.print(f"[dim]-> RAG Execution Time: {rag_end - rag_start:.4f} seconds[/dim]")
        
        console.print("\n[bold yellow][3] Measuring LLM Inference (Gemma 4 LiteRT)...[/bold yellow]")
        sys_prompt, _, _ = viorra.engine.get_cached_prompts()
        sys_prompt = sys_prompt.replace("[[TEST_TEXT]]", test_essay)
        sys_prompt = sys_prompt.replace("[[RAG_EXAMPLES]]", "DUMMY RAG EXAMPLES")
        formatted_prompt = f"<|turn|>user\n{sys_prompt}<turn|>\n<|turn|>model\n"
        
        infer_start = time.time()
        with viorra.engine.llm_engine.create_conversation() as conversation:
            response = conversation.send_message(formatted_prompt)
            output_text = response["content"][0]["text"]
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
            "models--litert-community--gemma-4-E2B-it-litert-lm"
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
            hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_faiss.index", repo_type="dataset", force_download=True)
            hf_hub_download(repo_id="qsardor/viorra-admissions-essays", filename="viorra_corpus.pkl", repo_type="dataset", force_download=True)
            console.print(f"[bold green]Successfully updated RAG Database.[/bold green]")
            
            from viorra.server import USER_DATA_DIR
            from viorra.hardware import download_llm_native
            native_llm = os.path.join(USER_DATA_DIR, "gemma-4-E2B-it.litertlm")
            if os.path.exists(native_llm):
                os.remove(native_llm)
            
            console.print(f"Downloading Native LLM Engine (~2.5GB)...")
            download_llm_native(native_llm)
            console.print(f"[bold green]Successfully updated LiteRT Engine.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Update failed. Ensure you have an active internet connection. Error: {e}[/bold red]")
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
        console.print(f"Loading RAG database and LiteRT models to analyze {len(essay_text.split())} words...\n")
        
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
                
        sys.exit(0)

    if args.command == "start":
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
        
        if not os.path.exists(boot_flag):
            from viorra.hardware import profile_system_hardware
            gpu_name, vram = profile_system_hardware()
            
            if not gpu_name or vram < (2 * 1024**3):
                console.print(Align.center("\n[bold red]WARNING: NO DEDICATED GPU DETECTED[/bold red]"))
                console.print(Align.center("[dim]VIORRA will download ~3GB of models and run on CPU when you start analysis.[/dim]\n"))
            
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
                input("")
                webbrowser.open("http://127.0.0.1:8000/")
            except (KeyboardInterrupt, EOFError):
                import os, signal
                os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                pass
            
        threading.Thread(target=wait_for_enter, daemon=True).start()
        
        import logging  # noqa: F401
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
