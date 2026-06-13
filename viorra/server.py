"""
VIORRA FASTAPI SERVER
---------------------
This module manages the HTTP web server and API endpoints for the UI frontend.
It uses FastAPI to serve standard REST endpoints (/api/analyze, /api/chat).

CRITICAL ARCHITECTURE NOTE:
All heavy AI workloads (LiteRT inference, FastEmbed computation, FAISS searches)
are dispatched to background threads using `asyncio.to_thread()`. This is absolutely
vital because LiteRT is a synchronous blocking C++ engine. If we ran it in the main
asyncio event loop, the entire web server would hang and stop responding to UI requests.
"""

import os
import asyncio
import io
import json
import uuid
import logging
import zipfile
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Initialize the main FastAPI backend application
app = FastAPI(title="VIORRA Application")

import time
last_active_time = time.time()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(inactivity_monitor())

async def inactivity_monitor():
    global last_active_time
    sleeping = False
    while True:
        await asyncio.sleep(10)
        inactive_seconds = time.time() - last_active_time
        
        # Shutdown completely after 10 minutes (600 seconds)
        if inactive_seconds > 600:
            from rich.console import Console
            Console().print("[bold red]Viorra shut down because of inactivity.[/bold red]")
            os._exit(0)
            
        # Sleep mode (unload from VRAM) after 5 minutes (300 seconds)
        elif inactive_seconds > 300 and not sleeping:
            from viorra.engine import unload_models
            unload_models()
            sleeping = True
            
        # Wake up detection
        elif inactive_seconds < 300 and sleeping:
            sleeping = False

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """
    Handles file uploads from the frontend UI. 
    Parses raw text out of common document formats (.txt, .md, .pdf, .docx).
    """
    filename = file.filename.lower()
    content = await file.read()
    text = ""
    
    try:
        # Standard raw text
        if filename.endswith(".txt") or filename.endswith(".md"):
            text = content.decode("utf-8", errors="ignore")
            
        # PDF Parsing
        elif filename.endswith(".pdf"):
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                    
        # Word Document Parsing
        elif filename.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
                
        else:
            return {"error": "Unsupported file format. Please upload TXT, MD, PDF, or DOCX."}
            
        return {"text": text.strip()}
    except Exception as e:
        return {"error": f"Failed to parse file: {str(e)}"}

class EndpointFilter(logging.Filter):
    """
    Filters out spammy Uvicorn health checks from the console output 
    to keep the terminal pristine for the user.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /api/status") == -1

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

@app.on_event("startup")
def startup_event():
    """
    Triggered when FastAPI boots up.
    We spawn the heavy LiteRT/Gemma model loading in a background daemon thread.
    This prevents the models from blocking the web server port from opening.
    """
    import threading
    from viorra.engine import ensure_models_loaded
    # Pre-load the models in the background so the server can still bind to the port immediately
    threading.Thread(target=ensure_models_loaded, daemon=True).start()


# --- STORAGE UTILITIES ---
def get_storage_dir():
    import platform
    system = platform.system()
    home = os.path.expanduser("~")
    
    if system == "Windows":
        base_path = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
    elif system == "Darwin":
        base_path = os.path.join(home, "Library", "Application Support")
    else:
        base_path = os.environ.get("XDG_CONFIG_HOME", os.path.join(home, ".config"))
        
    storage_path = os.path.join(base_path, "Viorra")
    os.makedirs(os.path.join(storage_path, "Sessions"), exist_ok=True)
    return storage_path

USER_DATA_DIR = get_storage_dir()
CONFIG_PATH = os.path.join(USER_DATA_DIR, "config.json")
# -------------------------

# Get path to static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    global last_active_time
    last_active_time = time.time()
    
    from viorra.engine import analyze_essay
    session_id = datetime.now().strftime("%m.%d.%Y.%H.%M.%S")

    # Run the heavy analysis in a separate thread so it doesn't block the async loop
    result = await asyncio.to_thread(analyze_essay, request.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    result["session_id"] = session_id
    return result


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    essay_text: str
    previous_feedback: str
    chat_history: List[ChatMessage]
    new_message: str
    retrieved_docs: List[Dict[str, Any]] = []

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global last_active_time
    last_active_time = time.time()
    
    from viorra.engine import chat_with_viorra
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
    
    result = await asyncio.to_thread(
        chat_with_viorra,
        request.essay_text,
        request.previous_feedback,
        history_dicts,
        request.new_message,
        request.retrieved_docs
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    # The frontend will call /api/sessions/save to sync the state instead of appending to chat logs
    return result



class SaveSessionRequest(BaseModel):
    session_id: str
    essay_text: str
    data: Dict[str, Any]
    chat_history: List[Dict[str, Any]]

@app.post("/api/sessions/save")
async def api_save_session(req: SaveSessionRequest):
    session_path = os.path.join(USER_DATA_DIR, "Sessions", f"{req.session_id}.json")
    try:
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(req.dict(), f, indent=2)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def api_list_sessions():
    sessions_dir = os.path.join(USER_DATA_DIR, "Sessions")
    if not os.path.exists(sessions_dir):
        return []
        
    files = [f for f in os.listdir(sessions_dir) if f.endswith(".json")]
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(sessions_dir, x)), reverse=True)
    return files

@app.get("/api/sessions/{filename}")
async def api_get_session(filename: str):
    if not filename.endswith(".json"):
        filename += ".json"
    session_path = os.path.join(USER_DATA_DIR, "Sessions", filename)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sessions/{filename}")
async def api_delete_session(filename: str):
    if not filename.endswith(".json"):
        filename += ".json"
    session_path = os.path.join(USER_DATA_DIR, "Sessions", filename)
    if os.path.exists(session_path):
        os.remove(session_path)
        return {"success": True}
    return {"success": False, "error": "Not found"}

BOOT_ID = str(uuid.uuid4())

@app.get("/api/status")
def api_status():
    import viorra.engine
    status_msg = getattr(viorra.engine, "boot_status_message", "Waking up...")
    return {"ready": viorra.engine.is_loaded, "boot_id": BOOT_ID, "status": status_msg}






@app.delete("/api/factory_reset")
async def factory_reset():
    import shutil
    import asyncio
    
    # Wipe user sessions and config
    if os.path.exists(USER_DATA_DIR):
        try:
            shutil.rmtree(USER_DATA_DIR)
        except:
            pass
            
    # Wipe HuggingFace models and datasets cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_cache):
        try:
            shutil.rmtree(hf_cache)
        except:
            pass
            
    # Trigger self-destruct 1 second after returning success to frontend
    async def delayed_shutdown():
        await asyncio.sleep(1)
        os._exit(0)
        
    asyncio.create_task(delayed_shutdown())
    
    return {"success": True}



# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "index.html not found"}
