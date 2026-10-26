import json
import os
import uuid
from datetime import datetime

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")


def _ensure_history_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def get_all_sessions() -> list[dict]:
    """Retrieve all saved meeting sessions, sorted by date (newest first)."""
    _ensure_history_file()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            sessions = json.load(f)
            return sorted(sessions, key=lambda s: s.get("created_at", ""), reverse=True)
    except Exception as e:
        print(f"Error reading history file: {e}")
        return []


def get_session(session_id: str) -> dict | None:
    """Retrieve a specific session by its unique ID."""
    sessions = get_all_sessions()
    for s in sessions:
        if s.get("id") == session_id:
            return s
    return None


def save_session(session_data: dict) -> str:
    """Save or update a session in history.json and return its ID."""
    _ensure_history_file()
    sessions = get_all_sessions()

    session_id = session_data.get("id") or str(uuid.uuid4())[:8]
    created_at = session_data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "id": session_id,
        "created_at": created_at,
        "title": session_data.get("title", "Untitled Meeting"),
        "source": session_data.get("source", ""),
        "language": session_data.get("language", "english"),
        "summary_mode": session_data.get("summary_mode", "Comprehensive Meeting Minutes"),
        "summary": session_data.get("summary", ""),
        "action_items": session_data.get("action_items", ""),
        "key_decisions": session_data.get("key_decisions", ""),
        "open_questions": session_data.get("open_questions", ""),
        "transcript": session_data.get("transcript", ""),
        "timestamped_transcript": session_data.get("timestamped_transcript", ""),
        "chapters": session_data.get("chapters", ""),
        "chat_history": session_data.get("chat_history", []),
    }

    # Replace if exists, else prepend
    existing_idx = next((i for i, s in enumerate(sessions) if s.get("id") == session_id), -1)
    if existing_idx >= 0:
        sessions[existing_idx] = entry
    else:
        sessions.insert(0, entry)

    # Keep maximum 50 most recent sessions to avoid bloated JSON
    sessions = sessions[:50]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)

    return session_id


def delete_session(session_id: str) -> bool:
    """Delete a specific session by ID."""
    _ensure_history_file()
    sessions = get_all_sessions()
    new_sessions = [s for s in sessions if s.get("id") != session_id]
    if len(new_sessions) < len(sessions):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(new_sessions, f, indent=2, ensure_ascii=False)
        return True
    return False


def clear_all_history() -> bool:
    """Clear all past meeting sessions."""
    _ensure_history_file()
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
    return True


def search_sessions(query: str) -> list[dict]:
    """Search sessions by title, summary, or transcript."""
    query = query.lower().strip()
    if not query:
        return get_all_sessions()
    sessions = get_all_sessions()
    return [
        s for s in sessions
        if query in s.get("title", "").lower()
        or query in s.get("summary", "").lower()
        or query in s.get("action_items", "").lower()
        or query in s.get("source", "").lower()
    ]
