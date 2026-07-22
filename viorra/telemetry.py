import os
import json
import threading
import requests
import viorra.db as db

SUPABASE_URL = "https://hxydvhhfmldifpccyhxj.supabase.co/rest/v1/telemetry"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh4eWR2aGhmbWxkaWZwY2N5aHhqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODQ3MjU1MjUsImV4cCI6MjEwMDMwMTUyNX0.P4FdB736sTrwbHD3dWIfBG_2Dri_-GCmVNUi0ipf_qc"

def _send_supabase_event(payload: dict):
    """
    Sends a non-blocking telemetry payload to Supabase cloud.
    """
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    try:
        requests.post(SUPABASE_URL, headers=headers, json=payload, timeout=4)
    except Exception:
        pass

def record_install_event():
    """
    Fired on first boot of Viorra client to track total downloads / app installs.
    """
    from viorra.server import USER_DATA_DIR
    ping_flag = os.path.join(USER_DATA_DIR, ".installed_ping")
    if not os.path.exists(ping_flag):
        payload = {
            "event_type": "install",
            "session_id": "new_client_install",
            "word_count": 0,
            "infer_time": 0.0,
            "tokens_sec": 0.0,
            "diagnostics_count": 0
        }
        try:
            with open(ping_flag, "w", encoding="utf-8") as f:
                f.write("1")
        except Exception:
            pass
            
        threading.Thread(target=_send_supabase_event, args=(payload,), daemon=True).start()

def record_analysis_event(session_id: str, essay_text: str, result_data: dict, incognito: bool = False):
    """
    Logs an essay analysis event locally and syncs to Supabase cloud telemetry.
    Respects incognito mode toggle.
    """
    if incognito:
        return

    word_count = len(essay_text.split())
    benchmark = result_data.get("benchmark") or {}
    infer_time = benchmark.get("infer_time", 0.0)
    tokens_sec = benchmark.get("tokens_sec", 0.0)
    diagnostics = result_data.get("diagnostics") or []
    diag_count = len(diagnostics)

    # 1. Log to local SQLite DB
    db.log_analytics_event("analyze", session_id, word_count, infer_time, tokens_sec, diag_count)

    # 2. Sync to central Supabase Cloud DB in background thread
    payload = {
        "event_type": "analyze",
        "session_id": session_id,
        "word_count": word_count,
        "infer_time": infer_time,
        "tokens_sec": tokens_sec,
        "diagnostics_count": diag_count
    }
    threading.Thread(target=_send_supabase_event, args=(payload,), daemon=True).start()
