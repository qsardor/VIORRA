import os
import json
from smolagents import CodeAgent, Model, ChatMessage, tool
from smolagents.models import MessageRole
import viorra.engine as e

# We define a dedicated file for the user's permanent memory
MEMORY_FILE = os.path.join(
    os.getenv("LOCALAPPDATA", os.path.expanduser("~")), 
    "Viorra", 
    "user_knowledge_graph.json"
)

# --- 1. MACRO-TOOL: KNOWLEDGE GRAPH MANAGER ---
@tool
def manage_user_memory(action: str, knowledge: str = "") -> str:
    """
    Manages the permanent knowledge graph about the user.
    Use this tool to save or retrieve facts about the user's life, essays, goals, or weaknesses.
    
    Args:
        action: Must be either 'save' or 'read'.
        knowledge: The fact to save (e.g., 'User struggles with writing conclusions' or 'User plays squash'). Leave empty if reading.
    """
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    
    # Load current memory
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            try:
                memory_db = json.load(f)
            except:
                memory_db = []
    else:
        memory_db = []

    if action == "save":
        if knowledge and knowledge not in memory_db:
            memory_db.append(knowledge)
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(memory_db, f, indent=4)
            return f"Successfully saved knowledge: '{knowledge}'"
        return "Knowledge already exists or was empty."
        
    elif action == "read":
        if not memory_db:
            return "The user's memory database is currently empty."
        return "Current Knowledge Graph:\n" + "\n".join([f"- {m}" for m in memory_db])
        
    return "Invalid action. Must be 'save' or 'read'."

def load_memory():
    """Helper function to cleanly load memory for the main prompt injection."""
    try:
        # We manually read it here to bypass the LLM reasoning step for speed during chat injection
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                db = json.load(f)
                return db if db else []
        return []
    except Exception:
        return []

# --- 2. CUSTOM IN-PROCESS MODEL FOR SMOLAGENTS ---
class InProcessLlamaModel(Model):
    def __init__(self, **kwargs):
        super().__init__(model_id="viorra-gemma4", **kwargs)
        
    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list = None,
        **kwargs,
    ) -> ChatMessage:
        from smolagents.models import get_clean_message_list
        clean_msgs = get_clean_message_list(
            messages,
            role_conversions={"assistant": "model"}
        )
        
        # Format conversation using standard Gemma 4 format
        raw_prompt = ""
        for m in clean_msgs:
            role = m["role"]
            content = m.get("content", "")
            
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content_str = "\n".join(text_parts)
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content) if content is not None else ""
                
            if role in ["system", "developer"]:
                role = "system"
            elif role == "assistant":
                role = "model"
            raw_prompt += f"<|turn|>{role}\n{content_str.strip()}\n"
        raw_prompt += "<|turn|>model\n"
        
        # Stop sequences
        stops = ["<turn|>", "<|turn|>"]
        if stop_sequences:
            stops.extend(stop_sequences)
            
        # Run local in-process inference sharing same engine
        if not e.llm_engine:
            e.ensure_models_loaded()
            
        response = e.llm_engine.create_completion(
            prompt=raw_prompt,
            max_tokens=512,
            temperature=0.0,
            stop=stops
        )
        output_text = response["choices"][0]["text"].strip()
        
        # Strip thinking tag residues to ensure clean Python code outputs
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()
        if "<channel|>" in output_text:
            output_text = output_text.split("<channel|>")[-1].strip()
            
        return ChatMessage(role=MessageRole.ASSISTANT, content=output_text)

# --- 3. THE SMOLAGENTS ORCHESTRATOR ---
def run_memory_agent_async(user_message: str, viorra_response: str):
    """
    Runs completely in the background so the user's chat speed is never delayed.
    It reads the latest exchange and decides if there are any critical facts about the user 
    that should be permanently saved into the Knowledge Graph.
    """
    # Initialize the custom local model
    model = InProcessLlamaModel()
    
    # Instantiate the CodeAgent with our Memory Macro-Tool
    agent = CodeAgent(
        tools=[manage_user_memory],
        model=model,
        additional_authorized_imports=["json", "os"]
    )
    
    # The Prompt for the Background Agent
    agent_task = f"""
    You are Viorra's background Memory Manager.
    Your job is to read the latest conversation and permanently save any important facts about the user.
    
    User just said: "{user_message}"
    Viorra replied: "{viorra_response}"
    
    If the user revealed a personal fact, a specific goal, or a recurring writing weakness, use the `manage_user_memory` tool with action='save' to store it.
    If there is nothing profoundly important to remember, do not save anything.
    Write the code to execute this logic.
    """
    
    try:
        agent.run(agent_task)
    except Exception as e:
        import logging
        logging.error(f"[SMOLAGENTS ERROR] Failed to run memory extraction: {e}")
