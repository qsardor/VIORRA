import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from viorra.engine import analyze_essay, ensure_models_loaded

app = FastAPI(title="VIORRA Local Server")

import logging
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /api/status") == -1

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

@app.on_event("startup")
def startup_event():
    import threading
    from viorra.engine import ensure_models_loaded
    # Pre-load the models in the background so the server can still bind to the port immediately
    threading.Thread(target=ensure_models_loaded, daemon=True).start()

# Get path to static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def api_analyze(request: AnalyzeRequest):
    # Run the heavy analysis in a separate thread so it doesn't block the async loop
    result = await asyncio.to_thread(analyze_essay, request.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/api/status")
def api_status():
    import viorra.engine
    return {"ready": viorra.engine.is_loaded}

import json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except:
            pass
    return {"language": "English"}

class SettingsRequest(BaseModel):
    language: str

@app.get("/api/settings")
def api_get_settings():
    return load_config()

@app.post("/api/settings")
def api_save_settings(req: SettingsRequest):
    with open(CONFIG_PATH, "w") as f:
        json.dump(req.dict(), f)
    return {"status": "ok"}

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "index.html not found"}
