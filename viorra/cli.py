"""
VIORRA Command Line Interface.
Provides a clean, aesthetic boot sequence for students.
"""

import os
import sys
import argparse
import uvicorn
import webbrowser
import threading
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

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
    parser = argparse.ArgumentParser(description="VIORRA: The elite personal statement coach, running locally.")
    parser.add_argument("command", nargs="?", default="start", help="Command to run (default: start the web server)")
    parser.add_argument("--cli", type=str, help="Analyze an essay file directly via CLI without starting the server", metavar="FILE")
    parser.add_argument("--factory-reset", action="store_true", help="Delete all local user data and logs")
    parser.add_argument("--clear-cache", action="store_true", help="Purge HuggingFace and FastEmbed model caches to free up disk space")
    parser.add_argument("--status", action="store_true", help="Check system hardware compatibility for VIORRA")
    parser.add_argument("--update", action="store_true", help="Force redownload of the latest RAG database and models")
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
        sys.exit(0)

    if args.status:
        from viorra.engine import profile_system_hardware
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
            
            hf_hub_download(repo_id="litert-community/gemma-4-E2B-it-litert-lm", filename="gemma-4-E2B-it.litertlm", force_download=True)
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
            
            # Print raw output if we want to expose <think> tags natively
            raw = result.get("raw_output", "")
            if "<think>" in raw:
                think_block = raw.split("<think>")[1].split("</think>")[0].strip()
                console.print(Panel(think_block, title="[bold magenta]INTERNAL REASONING (<think>)[/bold magenta]", border_style="magenta"))
            
            console.print("\n[bold cyan]RAG MATCHES (Ivy League Database)[/bold cyan]")
            for doc in result.get("retrieved_docs", []):
                console.print(f"\n[bold white]Match #{doc['id']}:[/bold white]")
                console.print(f"[dim]Excerpt:[/dim] {doc['excerpt']}...")
                console.print(f"[italic green]Feedback:[/italic green] {doc['feedback']}")
                
        sys.exit(0)

    if args.command == "start":
        setup_quiet_logging()
        
        # Print beautiful ASCII UI
        os.system('cls' if os.name == 'nt' else 'clear')
        title = "             [bold cyan]VIORRA[/bold cyan]             "
        subtitle = "[italic white]Intelligent coaching. Beautifully delivered.[/italic white]"
        panel = Panel(f"{title}\n{subtitle}", border_style="cyan", expand=False, padding=(1, 3))
        
        print("")
        console.print(panel)
        print("")

        def wait_for_enter():
            import urllib.request
            
            with console.status(" [bold cyan]VIORRA is starting...[/bold cyan]", spinner="bouncingBar"):
                while True:
                    try:
                        # Use a long timeout so we don't abort the connection while Uvicorn is busy loading the model
                        urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=15)
                        break
                    except Exception:
                        time.sleep(0.5)
            
            time.sleep(0.5)
            console.print("  [bold green]Ready![/bold green]")
            print("")
            console.print("  [dim]For shutdown application press CTRL+C[/dim]\n")
            
            try:
                input("  [ PRESS ENTER TO OPEN ]\n")
                webbrowser.open("http://127.0.0.1:8000/")
            except Exception:
                pass
            
        threading.Thread(target=wait_for_enter, daemon=True).start()
        
        import logging
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
