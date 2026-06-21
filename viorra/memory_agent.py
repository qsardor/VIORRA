import os
import json
import logging
import viorra.engine as ve

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
    return os.path.join(base_path, "Viorra")

MEMORY_FILE = os.path.join(get_storage_dir(), "viorra_memory.json")

def load_memory() -> list:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def manage_memory(add_facts: list[str], remove_facts: list[str]) -> str:
    """
    Manages the permanent memory ledger for the user. Use this to add new facts or remove obsolete/resolved facts.
    
    Args:
        add_facts: A list of new permanent facts to add.
        remove_facts: A list of exact facts to remove (must match existing facts closely).
    """
    memory = load_memory()
    
    # Remove
    for fact_to_remove in remove_facts:
        memory = [m for m in memory if fact_to_remove.lower() not in m.lower()]
        
    # Add
    memory.extend(add_facts)
    
    # Cap at 20 facts
    if len(memory) > 20:
        memory = memory[-20:]
        
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)
        
    return f"Memory updated. Added {len(add_facts)} facts, removed {len(remove_facts)}."

import re
import json_repair

def run_memory_extraction(chat_log: str):
    """
    Called asynchronously by the background thread. Parses chat log and updates memory via direct JSON extraction.
    """
    if not ve.is_loaded or ve.llm_engine is None:
        return

    current_memory = load_memory()
    memory_str = "\n".join([f"- {m}" for m in current_memory]) if current_memory else "Empty"
    
    sys_prompt = f"""You are a background memory manager.
Read the recent chat log. Extract permanent, useful facts about the user's background or writing weaknesses.
If the user fixed a weakness, remove it from the ledger.

CURRENT MEMORY LEDGER:
{memory_str}

RECENT CHAT LOG:
{chat_log}

OUTPUT FORMAT:
Output ONLY raw JSON. Do not write markdown blocks. Do not invent keys.
{{
  "add_facts": ["fact 1 to add", "fact 2 to add"],
  "remove_facts": ["exact fact to remove from ledger"]
}}"""

    import litert_lm
    system_messages = [litert_lm.Message.system(sys_prompt)]
    
    try:
        with ve._inference_lock:
            with ve.llm_engine.create_conversation(messages=system_messages) as conversation:
                response = conversation.send_message("Generate the JSON memory update.")
                output_text = response["content"][0]["text"]
        
        # Parse JSON output
        text_to_parse = re.sub(r'<\|channel\|?>thought.*?<channel\|>', '', output_text, flags=re.DOTALL).strip()
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text_to_parse, re.DOTALL)
        if json_match:
            text_to_parse = json_match.group(1)
            
        data = json_repair.loads(text_to_parse)
        if isinstance(data, dict):
            manage_memory(data.get("add_facts", []), data.get("remove_facts", []))
            
    except Exception as e:
        logging.error(f"Memory extraction failed: {e}")
