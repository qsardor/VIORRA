import argparse
import uvicorn
import webbrowser
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description="VIORRA - Local AI Statement Reviewer")
    parser.add_argument("command", nargs="?", default="start", help="Command to run (e.g. 'start')")
    args = parser.parse_args()

    if args.command == "start":
        console.print("[bold red]🚀 Booting VIORRA Local Server...[/bold red]")
        
        import threading
        import time
        
        def wait_for_enter():
            import urllib.request
            
            # Poll until server is fully booted and responding
            while True:
                try:
                    urllib.request.urlopen("http://127.0.0.1:8000/", timeout=1)
                    break
                except Exception:
                    time.sleep(0.5)
            
            # Wait an extra half second to let Uvicorn's boot logs print first
            time.sleep(0.5)
            print("\n" + "="*55)
            try:
                input(" 👉 Press ENTER to open VIORRA in your browser 👈\n" + "="*55 + "\n")
                webbrowser.open("http://127.0.0.1:8000/")
            except (EOFError, KeyboardInterrupt):
                print("\n✅ Application is shut down.")
            
        threading.Thread(target=wait_for_enter, daemon=True).start()
        
        console.print("[bold green]✅ Server Starting...[/bold green]")
        uvicorn.run("viorra.server:app", host="127.0.0.1", port=8000, reload=False)
    else:
        console.print(f"[bold yellow]Unknown command: {args.command}[/bold yellow]")

if __name__ == "__main__":
    main()
