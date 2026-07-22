import sqlite3
import os
import json
from datetime import datetime

USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Viorra")
DB_PATH = os.path.join(USER_DATA_DIR, "viorra.db")

def get_connection():
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def migrate_legacy_sessions(conn):
    """
    Scans USER_DATA_DIR/Sessions/ for old JSON session files from previous Viorra versions 
    and imports them into SQLite DB to prevent any session loss when upgrading versions.
    """
    sessions_dir = os.path.join(USER_DATA_DIR, "Sessions")
    if not os.path.exists(sessions_dir):
        return
        
    cursor = conn.cursor()
    for fname in os.listdir(sessions_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(sessions_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                
                session_id = content.get("session_id") or os.path.splitext(fname)[0]
                essay_text = content.get("essay_text") or content.get("text") or ""
                data = content.get("data") or {}
                chat_history = content.get("chat_history") or []
                updated_at = content.get("updated_at") or datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT OR IGNORE INTO sessions (session_id, updated_at, essay_text, data, chat_history)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, updated_at, essay_text, json.dumps(data), json.dumps(chat_history)))
            except Exception:
                pass
    conn.commit()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            updated_at TIMESTAMP,
            essay_text TEXT,
            data TEXT,
            chat_history TEXT
        )
    ''')
    
    # Create FTS5 virtual table for full-text search
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
            session_id UNINDEXED,
            essay_text,
            chat_history
        )
    ''')
    
    # Create memory table for permanent facts
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT UNIQUE,
            created_at TIMESTAMP
        )
    ''')
    
    # Create analytics table for telemetry and progress tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            session_id TEXT,
            word_count INTEGER,
            infer_time REAL,
            tokens_sec REAL,
            diagnostics_count INTEGER,
            created_at TIMESTAMP
        )
    ''')
    
    conn.commit()

    # Automatically migrate legacy disk sessions from older versions
    migrate_legacy_sessions(conn)

    # Sync unindexed sessions into FTS5 virtual table
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO sessions_fts (session_id, essay_text, chat_history)
            SELECT session_id, essay_text, chat_history FROM sessions
            WHERE session_id NOT IN (SELECT session_id FROM sessions_fts)
        ''')
        conn.commit()
    except Exception:
        pass

    conn.close()

# Initialize tables when this module is imported
init_db()

# --- SESSIONS API ---

def normalize_session_data(raw_data):
    """
    Normalizes session data schema across different Viorra version formats.
    Ensures mentor_summary, diagnostics, retrieved_docs, and benchmark keys are always present.
    """
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except Exception:
            raw_data = {"mentor_summary": raw_data, "diagnostics": []}

    if not isinstance(raw_data, dict):
        raw_data = {}

    mentor_summary = raw_data.get("mentor_summary") or raw_data.get("summary") or raw_data.get("raw_output") or ""
    diagnostics = raw_data.get("diagnostics") or []
    if not isinstance(diagnostics, list):
        diagnostics = []
    
    normalized_diagnostics = []
    for d in diagnostics:
        if isinstance(d, dict):
            quote = d.get("quote") or d.get("text") or ""
            feedback = d.get("feedback") or d.get("comment") or ""
            normalized_diagnostics.append({"quote": quote, "feedback": feedback})

    retrieved_docs = raw_data.get("retrieved_docs") or []
    if not isinstance(retrieved_docs, list):
        retrieved_docs = []

    return {
        "mentor_summary": mentor_summary,
        "diagnostics": normalized_diagnostics,
        "retrieved_docs": retrieved_docs,
        "benchmark": raw_data.get("benchmark")
    }

def normalize_chat_history(raw_history):
    if isinstance(raw_history, str):
        try:
            raw_history = json.loads(raw_history)
        except Exception:
            raw_history = []
    if not isinstance(raw_history, list):
        return []
    
    normalized = []
    for msg in raw_history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            if role in ["assistant", "bot", "viorra"]:
                role = "model"
            content = msg.get("content") or msg.get("text") or ""
            normalized.append({"role": role, "content": content})
    return normalized

def save_session(session_id: str, essay_text: str, data: dict, chat_history: list):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    
    norm_data = normalize_session_data(data)
    norm_history = normalize_chat_history(chat_history)

    cursor.execute('''
        INSERT OR REPLACE INTO sessions (session_id, updated_at, essay_text, data, chat_history)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, now, essay_text, json.dumps(norm_data), json.dumps(norm_history)))
    
    # Sync FTS table
    cursor.execute('DELETE FROM sessions_fts WHERE session_id = ?', (session_id,))
    cursor.execute('''
        INSERT INTO sessions_fts (session_id, essay_text, chat_history)
        VALUES (?, ?, ?)
    ''', (session_id, essay_text, json.dumps(norm_history)))
    
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT session_id, updated_at FROM sessions ORDER BY updated_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [{"session_id": row["session_id"], "updated_at": row["updated_at"]} for row in rows]

def get_session(session_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "session_id": row["session_id"],
            "updated_at": row["updated_at"],
            "essay_text": row["essay_text"] or "",
            "data": normalize_session_data(row["data"]),
            "chat_history": normalize_chat_history(row["chat_history"])
        }
    return None

def delete_session(session_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
    deleted = cursor.rowcount > 0
    cursor.execute('DELETE FROM sessions_fts WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()
    return deleted

def search_sessions(query: str):
    """
    Search past sessions using SQLite FTS5.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.session_id, s.updated_at, snippet(sessions_fts, -1, '<b>', '</b>', '...', 32) as match_snippet
        FROM sessions_fts fts
        JOIN sessions s ON fts.session_id = s.session_id
        WHERE sessions_fts MATCH ?
        ORDER BY rank
        LIMIT 10
    ''', (query,))
    rows = cursor.fetchall()
    conn.close()
    return [{"session_id": row["session_id"], "updated_at": row["updated_at"], "snippet": row["match_snippet"]} for row in rows]

# --- MEMORY API ---

def save_memory_fact(fact: str):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    try:
        cursor.execute('INSERT INTO memory (fact, created_at) VALUES (?, ?)', (fact, now))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False # Already exists
    finally:
        conn.close()
    return success

def read_all_memory():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT fact FROM memory ORDER BY created_at ASC')
    rows = cursor.fetchall()
    conn.close()
    return [row["fact"] for row in rows]

def wipe_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sessions')
    cursor.execute('DELETE FROM sessions_fts')
    cursor.execute('DELETE FROM memory')
    cursor.execute('DELETE FROM analytics')
    conn.commit()
    conn.close()

# --- ANALYTICS & TELEMETRY API ---

def log_analytics_event(event_type: str, session_id: str, word_count: int, infer_time: float, tokens_sec: float, diagnostics_count: int):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    try:
        cursor.execute('''
            INSERT INTO analytics (event_type, session_id, word_count, infer_time, tokens_sec, diagnostics_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (event_type, session_id, word_count, infer_time, tokens_sec, diagnostics_count, now))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

def get_analytics_summary():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as total_events FROM analytics')
    total_events = cursor.fetchone()["total_events"]
    
    cursor.execute('SELECT COUNT(DISTINCT session_id) as total_sessions FROM analytics')
    total_sessions = cursor.fetchone()["total_sessions"]
    
    cursor.execute('SELECT AVG(infer_time) as avg_infer_time, AVG(tokens_sec) as avg_tps, AVG(word_count) as avg_words FROM analytics WHERE event_type = "analyze"')
    row = cursor.fetchone()
    avg_infer_time = round(row["avg_infer_time"] or 0, 2)
    avg_tps = round(row["avg_tps"] or 0, 2)
    avg_words = round(row["avg_words"] or 0, 1)
    
    cursor.execute('SELECT * FROM analytics ORDER BY id DESC LIMIT 50')
    recent_rows = cursor.fetchall()
    conn.close()
    
    recent_events = [
        {
            "id": r["id"],
            "event_type": r["event_type"],
            "session_id": r["session_id"],
            "word_count": r["word_count"],
            "infer_time": r["infer_time"],
            "tokens_sec": r["tokens_sec"],
            "diagnostics_count": r["diagnostics_count"],
            "created_at": r["created_at"]
        } for r in recent_rows
    ]
    
    return {
        "total_analyses": total_events,
        "total_sessions": total_sessions,
        "avg_infer_time_sec": avg_infer_time,
        "avg_tokens_per_sec": avg_tps,
        "avg_word_count": avg_words,
        "recent_events": recent_events
    }
