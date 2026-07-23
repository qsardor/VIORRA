"""
VIORRA FASTAPI SERVER
---------------------
This module manages the HTTP web server and API endpoints for the UI frontend.
It uses FastAPI to serve standard REST endpoints (/api/analyze, /api/chat).

CRITICAL ARCHITECTURE NOTE:
All heavy AI workloads (Llama.cpp inference, FastEmbed computation, TurboVec searches)
are dispatched to background threads using `asyncio.to_thread()`. This is absolutely
vital because Llama.cpp is a synchronous blocking C++ engine. If we ran it in the main
asyncio event loop, the entire web server would hang and stop responding to UI requests.
"""

import os
import asyncio
import io
import json
import uuid
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    try:
        from viorra.telemetry import record_install_event
        record_install_event()
    except Exception:
        pass
    
    import threading
    from viorra.engine import ensure_models_loaded
    
    # 1. Spawn heavy AI loading in a background thread so it doesn't block the async loop
    threading.Thread(target=ensure_models_loaded, daemon=True).start()
    
    # 2. Start the inactivity monitor daemon
    asyncio.create_task(inactivity_monitor())
    
    yield
    
    # --- SHUTDOWN ---
    from viorra.engine import unload_models
    unload_models()

# Initialize the main FastAPI backend application
app = FastAPI(title="VIORRA Application", lifespan=lifespan)

import time
last_active_time = time.time()

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
            
        # Word Document Parsing
        elif filename.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
                
        else:
            return {"error": "Unsupported file format. Please upload TXT, MD, or DOCX."}
            
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

# --- STORAGE UTILITIES ---
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Viorra")
os.makedirs(os.path.join(USER_DATA_DIR, "Sessions"), exist_ok=True)
CONFIG_PATH = os.path.join(USER_DATA_DIR, "config.json")
# -------------------------

# Get path to static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

class AnalyzeRequest(BaseModel):
    text: str
    debug_mode: bool = False
    incognito: bool = False

@app.post("/api/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    global last_active_time
    last_active_time = time.time()
    
    from viorra.engine import analyze_essay
    session_id = datetime.now().strftime("%m.%d.%Y.%H.%M.%S")

    # Run the heavy analysis in a separate thread so it doesn't block the async loop
    result = await asyncio.to_thread(analyze_essay, request.text, request.debug_mode)
        
    result["session_id"] = session_id

    # Record operational telemetry event
    if "error" not in result:
        try:
            from viorra.telemetry import record_analysis_event
            record_analysis_event(session_id, request.text, result, request.incognito)
        except Exception:
            pass

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
    incognito: bool = False

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
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

    # If the engine returned an error dict, surface it cleanly instead of crashing
    if "response" not in result:
        return result
        
    # Compile the chat log for the memory agent
    chat_log = ""
    for msg in history_dicts:
        chat_log += f"{msg['role'].capitalize()}: {msg['content']}\n"
    chat_log += f"User: {request.new_message}\n"
    chat_log += f"Viorra: {result['response']}\n"
    
    # Spawn the silent background extraction thread during idle time (unless incognito)
    if not request.incognito:
        try:
            from viorra.memory_agent import run_memory_agent_async
            background_tasks.add_task(run_memory_agent_async, request.new_message, result['response'])
        except Exception as e:
            import logging
            logging.error(f"[MEMORY AGENT ERROR] {e}")
        
    # The frontend will call /api/sessions/save to sync the state instead of appending to chat logs
    # Frontend handles skipping /api/sessions/save if incognito=true
    return result



class SaveSessionRequest(BaseModel):
    session_id: str
    essay_text: str
    data: Dict[str, Any]
    chat_history: List[Dict[str, Any]]
    incognito: bool = False

@app.post("/api/sessions/save")
async def api_save_session(req: SaveSessionRequest):
    if req.incognito:
        return {"status": "ok", "skipped": True}
        
    import viorra.db as db
    try:
        db.save_session(req.session_id, req.essay_text, req.data, req.chat_history)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def api_list_sessions():
    import viorra.db as db
    try:
        sessions = db.get_all_sessions()
        # Return list of filenames to match frontend expectations (id.json)
        return [f"{s['session_id']}.json" for s in sessions]
    except Exception:
        return []

@app.get("/api/sessions/{filename}")
async def api_get_session(filename: str):
    import viorra.db as db
    session_id = filename.removesuffix(".json") if filename.endswith(".json") else filename
    session = db.get_session(session_id)
    
    if not session:
        # Fallback: check if legacy session JSON file exists on disk
        legacy_filename = filename if filename.endswith(".json") else f"{filename}.json"
        legacy_path = os.path.join(USER_DATA_DIR, "Sessions", legacy_filename)
        if os.path.exists(legacy_path):
            try:
                import json
                with open(legacy_path, "r", encoding="utf-8") as f:
                    disk_data = json.load(f)
                sid = disk_data.get("session_id", session_id)
                text = disk_data.get("essay_text") or disk_data.get("text") or ""
                data = disk_data.get("data") or {}
                hist = disk_data.get("chat_history") or []
                db.save_session(sid, text, data, hist)
                session = db.get_session(sid)
            except Exception:
                pass
                
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session

@app.delete("/api/sessions/{filename}")
async def api_delete_session(filename: str):
    import viorra.db as db
    session_id = filename.removesuffix(".json") if filename.endswith(".json") else filename
    deleted = db.delete_session(session_id)
    
    # Also clean up legacy disk JSON file if present
    legacy_filename = filename if filename.endswith(".json") else f"{filename}.json"
    legacy_path = os.path.join(USER_DATA_DIR, "Sessions", legacy_filename)
    if os.path.exists(legacy_path):
        try:
            os.remove(legacy_path)
            deleted = True
        except Exception:
            pass

    if deleted:
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
    import asyncio
    import viorra.db as db
    
    # Wipe database tables cleanly
    try:
        db.wipe_database()
    except Exception as e:
        import logging
        logging.error(f"[DB WIPE ERROR] {e}")
        
    # Wipe config
    config_path = os.path.join(USER_DATA_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            os.remove(config_path)
        except Exception:
            pass
            
    # [PROTECTION]: We explicitly do NOT wipe USER_DATA_DIR or the HuggingFace cache 
    # to protect the 2.62GB GGUF model and the TurboVec embeddings from being destroyed.
            
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
