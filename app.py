import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Synchronize Streamlit Cloud secrets into os.environ
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k != "YOUTUBE_COOKIES" and len(v) < 30000:
                os.environ[k] = v
except Exception:
    pass

from utils.audio_processor import process_input, sanitize_youtube_url, DOWNLOAD_DIR
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title, generate_chapters
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from core.history_manager import (
    get_all_sessions,
    get_session,
    save_session,
    delete_session,
    search_sessions,
)
from core.export_manager import (
    generate_markdown_report,
    generate_plain_text_report,
    generate_action_items_csv,
    generate_notion_slack_tasks,
)
from core.instant_transcriber import fetch_instant_transcript, has_youtube_captions

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }
.badge-amber  { background: rgba(245,158,11,0.15); color: var(--warning);    border: 1px solid rgba(245,158,11,0.3); }

/* ── Export Toolbar ── */
.export-panel {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.export-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Inputs & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}

.stDownloadButton > button {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.2s !important;
}

.stDownloadButton > button:hover {
    border-color: var(--accent-2) !important;
    color: var(--accent-2) !important;
    box-shadow: 0 0 10px rgba(6,182,212,0.25) !important;
}

/* ── Progress Panel & Console ── */
.progress-panel {
    background: var(--surface);
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 25px rgba(124, 58, 237, 0.18);
    position: relative;
    overflow: hidden;
}

.progress-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.progress-step-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.progress-pct-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent-2);
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.35);
    border-radius: 8px;
    padding: 0.25rem 0.85rem;
    box-shadow: 0 0 12px rgba(6, 182, 212, 0.2);
}

.terminal-box {
    background: #09090f;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-top: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.7;
    max-height: 240px;
    overflow-y: auto;
}

.terminal-title-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.terminal-dots {
    display: inline-flex;
    gap: 6px;
    margin-right: 8px;
}

.t-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}
.t-red { background: #ef4444; }
.t-yellow { background: #f59e0b; }
.t-green { background: #10b981; }

.terminal-line {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}

.terminal-time {
    color: var(--accent-glow);
    flex-shrink: 0;
}

.terminal-msg {
    color: var(--text-muted);
    word-break: break-word;
}

.terminal-msg-latest {
    color: var(--accent-2) !important;
    font-weight: 600;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 0.9rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.35rem 0;
    border: 1px solid var(--border);
    font-size: 0.78rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 340px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

.chapters-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    font-size: 0.84rem;
    line-height: 1.8;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c3aed 0%, #9f67ff 50%, #06b6d4 100%) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: var(--surface-2) !important;
    border-radius: 4px !important;
    border: 1px solid var(--border) !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "current_session_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Input Section ──
    st.markdown('<span class="badge badge-purple">Input Source</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    uploaded_file = st.file_uploader(
        "Or Upload Audio / Video File",
        type=["mp4", "m4a", "wav", "mp3", "webm", "ogg"],
        help="Upload an audio or video file directly (ideal for cloud deployments when YouTube blocks cloud IPs)."
    )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    # ── Processing Modes ──
    st.markdown('<span class="badge badge-purple" style="margin-top:0.75rem">⚡ Processing Pipeline</span>', unsafe_allow_html=True)
    pipeline_mode = st.radio(
        "Choose Pipeline Speed",
        [
            "✨ Auto (Instant if captions exist, else Deep Audio)",
            "🚀 Instant Mode (2-5s, YouTube Captions)",
            "🎙️ Deep Audio Mode (Whisper AI, High Accuracy)",
        ],
        index=0,
        help="Instant Mode extracts existing YouTube captions in 2-4 seconds with zero audio download. Deep Audio Mode downloads audio and runs local Whisper transcription."
    )

    # ── Summary Presets ──
    st.markdown('<span class="badge badge-cyan" style="margin-top:0.75rem">Summary Preset / Audience</span>', unsafe_allow_html=True)
    summary_mode = st.selectbox(
        "Select Summary Tone",
        [
            "📋 Executive Brief",
            "⚙️ Technical Deep-Dive",
            "🎓 Study & Lecture Notes",
            "✉️ Email-Ready Follow-up",
            "📝 Comprehensive Minutes",
        ],
        index=0,
        help="Tailor the summary style for executives, engineering teams, students, or team emails."
    )

    run_btn = st.button("⚡  Analyse Meeting", use_container_width=True)

    # ── Saved Meetings Library ──
    st.markdown("---")
    st.markdown('<span class="badge badge-green">Saved Meetings Library</span>', unsafe_allow_html=True)
    
    search_q = st.text_input("🔍 Search Past Meetings", placeholder="Filter by keyword...", key="history_search")
    past_sessions = search_sessions(search_q) if search_q else get_all_sessions()

    if past_sessions:
        session_options = {s["id"]: f"[{s.get('created_at', '')[:10]}] {s.get('title', 'Untitled')[:28]}" for s in past_sessions}
        selected_session_id = st.selectbox(
            "Select from History",
            options=list(session_options.keys()),
            format_func=lambda x: session_options.get(x, x),
            label_visibility="collapsed",
        )

        hist_col1, hist_col2 = st.columns(2, gap="small")
        with hist_col1:
            if st.button("📂 Load", use_container_width=True):
                sess = get_session(selected_session_id)
                if sess:
                    st.session_state.result = {
                        "id": sess["id"],
                        "title": sess["title"],
                        "created_at": sess.get("created_at"),
                        "source": sess.get("source", ""),
                        "summary_mode": sess.get("summary_mode", "Comprehensive Minutes"),
                        "language": sess.get("language", "english"),
                        "summary": sess["summary"],
                        "chapters": sess.get("chapters", ""),
                        "action_items": sess["action_items"],
                        "key_decisions": sess["key_decisions"],
                        "open_questions": sess["open_questions"],
                        "transcript": sess["transcript"],
                        "timestamped_transcript": sess.get("timestamped_transcript", sess["transcript"]),
                        "rag_chain": build_rag_chain(
                            transcript=sess["transcript"],
                            summary=sess.get("summary", ""),
                            chapters=sess.get("chapters", ""),
                            key_decisions=sess.get("key_decisions", []),
                            action_items=sess.get("action_items", []),
                        ),
                    }
                    st.session_state.chat_history = sess.get("chat_history", [])
                    st.session_state.current_session_id = sess["id"]
                    st.session_state.pipeline_done = True
                    st.rerun()

        with hist_col2:
            if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                delete_session(selected_session_id)
                if st.session_state.current_session_id == selected_session_id:
                    st.session_state.result = None
                    st.session_state.current_session_id = None
                    st.session_state.pipeline_done = False
                st.rerun()

        if st.session_state.result:
            if st.button("➕ New Analysis", use_container_width=True):
                st.session_state.result = None
                st.session_state.current_session_id = None
                st.session_state.pipeline_done = False
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.markdown('<div style="color:var(--text-muted);font-size:0.75rem">No past meetings saved yet.</div>', unsafe_allow_html=True)

    # ── Pipeline Status in Sidebar ──
    sidebar_status_placeholder = st.empty()

    def render_sidebar_steps():
        with sidebar_status_placeholder.container():
            st.markdown("---")
            badge_class = "badge-green" if st.session_state.pipeline_done else "badge-purple"
            badge_label = "Pipeline Status" if st.session_state.pipeline_done else "Live Pipeline Status"
            st.markdown(f'<span class="badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)
            for step, icon, label in [
                ("audio",      "🔊", "Audio Processing"),
                ("transcript", "📝", "Transcription"),
                ("title",      "🏷️", "Title Generation"),
                ("chapters",   "📑", "Timeline Chapters"),
                ("summary",    "📋", "Summarisation"),
                ("extract",    "🔍", "Extraction"),
                ("rag",        "🧠", "RAG Engine"),
            ]:
                render_step_bar(label, step, icon)

    if st.session_state.pipeline_done:
        render_sidebar_steps()

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chapterise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    target_source = None
    if uploaded_file is not None:
        save_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        target_source = save_path
    elif source and source.strip():
        target_source = sanitize_youtube_url(source.strip())

    if not target_source:
        st.error("Please enter a YouTube URL or upload an audio/video file.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {
            "audio": "pending",
            "transcript": "pending",
            "title": "pending",
            "chapters": "pending",
            "summary": "pending",
            "extract": "pending",
            "rag": "pending",
        }

        progress_placeholder = st.empty()
        logs = []
        start_time = time.time()

        def get_elapsed_str():
            el = int(time.time() - start_time)
            return f"{el // 60:02d}:{el % 60:02d}"

        def update_progress(pct: int, step_title: str, msg: str):
            elapsed = get_elapsed_str()
            logs.append((elapsed, msg))

            lines_html = ""
            for idx, (t, m) in enumerate(logs):
                is_latest = (idx == len(logs) - 1)
                cls = "terminal-msg terminal-msg-latest" if is_latest else "terminal-msg"
                lines_html += f'<div class="terminal-line"><span class="terminal-time">[{t}]</span><span class="{cls}">&gt; {m}</span></div>'

            with progress_placeholder.container():
                st.markdown(f"""
                <div class="progress-panel">
                    <div class="progress-header-row">
                        <div class="progress-step-title">
                            <div class="status-dot dot-active"></div>
                            <span>{step_title}</span>
                        </div>
                        <div class="progress-pct-badge">{pct}%</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.progress(pct / 100.0, text=f"{msg} ({pct}%)")

                st.markdown(f"""
                <div class="terminal-box">
                    <div class="terminal-title-bar">
                        <div>
                            <span class="terminal-dots">
                                <span class="t-dot t-red"></span>
                                <span class="t-dot t-yellow"></span>
                                <span class="t-dot t-green"></span>
                            </span>
                            <span>Live Progress Console</span>
                        </div>
                        <span>Elapsed: {elapsed}</span>
                    </div>
                    {lines_html}
                </div>""", unsafe_allow_html=True)

            render_sidebar_steps()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            render_sidebar_steps()

        try:
            display_name = os.path.basename(target_source) if os.path.exists(target_source) else target_source
            transcript = None
            timestamped_transcript = None
            used_instant_mode = False

            is_youtube = ("youtube.com" in target_source or "youtu.be" in target_source)
            should_try_instant = is_youtube and ("Instant" in pipeline_mode or "Auto" in pipeline_mode)

            # ── 1. Try Instant Mode (YouTube Captions) ──
            if should_try_instant:
                update_step("audio", "active")
                update_progress(5, "Instant Mode", "Querying YouTube for existing captions/subtitles...")
                raw_cap, ts_cap = fetch_instant_transcript(target_source)
                if raw_cap and len(raw_cap.split()) >= 10:
                    transcript = raw_cap
                    timestamped_transcript = ts_cap
                    used_instant_mode = True
                    update_step("audio", "done")
                    update_step("transcript", "done")
                    word_count = len(transcript.split())
                    update_progress(60, "Instant Captions Acquired", f"🚀 Fetched {word_count} words in 2 seconds! Audio download skipped.")
                else:
                    if "Instant Mode" in pipeline_mode and "Auto" not in pipeline_mode:
                        update_progress(8, "Instant Mode Warning", "No captions found on YouTube. Falling back to Deep Audio Mode (Whisper AI)...")
                    else:
                        update_progress(8, "Auto Mode", "No pre-existing captions. Proceeding with Deep Audio Mode (Whisper AI)...")

            # ── 2. Deep Audio Mode (Whisper / Sarvam) if captions not used ──
            if not transcript:
                update_step("audio", "active")
                update_progress(10, "Audio Acquisition", f"Target: {display_name}")

                def audio_cb(p, m):
                    update_progress(p, "Audio Processing", m)

                chunks = process_input(target_source, status_callback=audio_cb)
                update_step("audio", "done")
                update_progress(20, "Audio Ready", f"Audio segmented into {len(chunks)} chunk(s).")

                update_step("transcript", "active")
                engine_name = "Sarvam AI (Hinglish)" if language.lower() == "hinglish" else "Whisper"
                update_progress(22, "Transcription", f"Initializing {engine_name}...")

                def trans_cb(p, m):
                    update_progress(p, "Transcription", m)

                transcript, timestamped_transcript = transcribe_all(
                    chunks, language, status_callback=trans_cb, return_timestamps=True
                )
                update_step("transcript", "done")
                word_count = len(transcript.split())
                update_progress(60, "Transcription Complete", f"Transcribed {word_count} words successfully with timestamps.")

            # ── 3. Title Generation ──
            update_step("title", "active")
            update_progress(62, "Title Generation", "Analyzing context to generate meeting title...")
            title = generate_title(transcript)
            update_step("title", "done")
            update_progress(66, "Title Generated", f"Title: {title}")

            # ── 4. Chapters & Timeline ──
            update_step("chapters", "active")
            update_progress(68, "Timeline Chapters", "Segmenting discussion into chronological topic chapters...")
            
            def chap_cb(p, m):
                update_progress(p, "Timeline Chapters", m)

            chapters = generate_chapters(timestamped_transcript or transcript, status_callback=chap_cb)
            update_step("chapters", "done")
            update_progress(72, "Chapters Ready", "Video chapters and timeline landmarks mapped.")

            # ── 5. Summarisation ──
            update_step("summary", "active")
            update_progress(74, "Summarisation", f"Synthesizing {summary_mode} with Mistral AI...")

            def summ_cb(p, m):
                update_progress(p, "Summarisation", m)

            summary = summarize(transcript, mode=summary_mode, status_callback=summ_cb)
            update_step("summary", "done")
            update_progress(84, "Summarisation Complete", f"{summary_mode} generated successfully.")

            # ── 6. Information Extraction ──
            update_step("extract", "active")
            update_progress(86, "Information Extraction", "Extracting action items with tasks, owners & deadlines...")
            action_items = extract_action_items(transcript)
            update_progress(89, "Information Extraction", "Extracting key decisions made...")
            decisions = extract_key_decisions(transcript)
            update_progress(92, "Information Extraction", "Extracting unresolved questions...")
            questions = extract_questions(transcript)
            update_step("extract", "done")
            update_progress(94, "Extraction Complete", "Action items, decisions & questions extracted.")

            # ── 7. RAG Engine ──
            update_step("rag", "active")
            update_progress(96, "RAG Engine", "Building semantic embeddings & grounded retriever...")
            rag_chain = build_rag_chain(
                transcript=transcript,
                summary=summary,
                chapters=chapters,
                key_decisions=decisions,
                action_items=action_items,
            )
            update_step("rag", "done")
            update_progress(100, "Analysis Complete", "All pipeline artifacts generated! Rendering results...")

            proc_label = "🚀 Instant Mode (YouTube Captions)" if used_instant_mode else "🎙️ Deep Audio Mode (Whisper AI)"
            result_data = {
                "title": title,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": target_source,
                "summary_mode": summary_mode,
                "processing_mode": proc_label,
                "language": language,
                "transcript": transcript,
                "timestamped_transcript": timestamped_transcript,
                "chapters": chapters,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }

            # Save to history manager
            session_id = save_session(result_data)
            result_data["id"] = session_id

            st.session_state.result = result_data
            st.session_state.current_session_id = session_id
            st.session_state.pipeline_done = True
            time.sleep(1.0)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio", "transcript", "title", "chapters", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            render_sidebar_steps()
            progress_placeholder.error(f"❌ Error during pipeline: {e}")

# ── Results Presentation ────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    created_date = r.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    summary_badge = r.get("summary_mode", "Comprehensive Minutes")
    proc_badge = r.get("processing_mode", "🎙️ Deep Audio Mode (Whisper AI)")
    slug = "".join(c if c.isalnum() else "_" for c in r.get("title", "meeting")[:25]).strip("_")

    # ── Title Banner ──
    st.markdown(f"""
    <div class="card">
        <div class="card-title">
            <span>📌 {created_date} · {r.get('language', 'English').upper()}</span>
            <div>
                <span class="badge badge-green" style="margin-right:0.35rem">{proc_badge}</span>
                <span class="badge badge-purple">{summary_badge}</span>
            </div>
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.35rem">
            Source: <code>{r.get('source', 'Unknown')}</code>
        </div>
    </div>""", unsafe_allow_html=True)


    # ── Productivity & Export Toolbar ──
    md_content = generate_markdown_report(r)
    txt_content = generate_plain_text_report(r)
    csv_content = generate_action_items_csv(r.get("action_items", ""))
    tasks_notion = generate_notion_slack_tasks(r.get("action_items", ""))

    st.markdown("""
    <div class="export-panel">
        <div class="export-title">
            <span>🚀 Meeting Export Center:</span>
        </div>
    </div>""", unsafe_allow_html=True)

    exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1], gap="small")
    with exp_c1:
        st.download_button(
            "📥 Download Minutes (.md)",
            data=md_content,
            file_name=f"{slug}_minutes.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with exp_c2:
        st.download_button(
            "📄 Download Plain Text (.txt)",
            data=txt_content,
            file_name=f"{slug}_notes.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with exp_c3:
        st.download_button(
            "📊 Export Action Items (.csv)",
            data=csv_content,
            file_name=f"{slug}_action_items.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("📋 Copy Formatted Checklist for Notion, Slack or Jira"):
        st.code(tasks_notion, language="markdown")

    # ── Embedded Audio / Video Player ──
    source_val = r.get("source", "")
    if source_val:
        with st.expander("🎬 Watch / Listen to Recording", expanded=True):
            if "youtube.com" in source_val or "youtu.be" in source_val:
                st.video(source_val)
            elif os.path.exists(source_val):
                ext = os.path.splitext(source_val)[1].lower()
                if ext in [".mp4", ".webm", ".mov", ".mkv"]:
                    st.video(source_val)
                else:
                    st.audio(source_val)

    # ── Summary & Chapters Grid ──
    col_sum, col_chap = st.columns([3, 2], gap="medium")

    with col_sum:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">
                <span>📋 Executive Summary</span>
                <span class="badge badge-cyan">{summary_badge}</span>
            </div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col_chap:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">
                <span>📑 Timeline & Chapters</span>
                <span class="badge badge-purple">Milestones</span>
            </div>
            <div class="chapters-box">{r.get('chapters', 'No chapters generated.')}</div>
        </div>""", unsafe_allow_html=True)

    # ── Action Center Grid (Action Items | Decisions | Questions) ──
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">
                <span>✅ Action Items</span>
                <span class="badge badge-green">Tasks</span>
            </div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">
                <span>🔑 Key Decisions</span>
                <span class="badge badge-cyan">Decisions</span>
            </div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">
                <span>❓ Open Questions</span>
                <span class="badge badge-amber">Follow-ups</span>
            </div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Full Transcript with Tabs ──
    with st.expander("📝 Full Meeting Transcript", expanded=False):
        tab_ts, tab_raw = st.tabs(["⏱️ Timestamped Transcript", "📄 Continuous Raw Text"])
        with tab_ts:
            st.markdown(
                f'<div class="transcript-box">{r.get("timestamped_transcript", r["transcript"])}</div>',
                unsafe_allow_html=True,
            )
        with tab_raw:
            st.markdown(
                f'<div class="transcript-box">{r["transcript"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── RAG Chat ──
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.25rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about this meeting! ChromaDB vector embeddings will ground the response.</div>
        </div>""", unsafe_allow_html=True)

    # Chat input
    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Your question", placeholder="What were the main decisions or deadlines?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Analyzing meeting transcript…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Persist updated chat in session history
        if st.session_state.current_session_id:
            r["chat_history"] = st.session_state.chat_history
            save_session(r)

        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            if st.session_state.current_session_id:
                r["chat_history"] = []
                save_session(r)
            st.rerun()

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:440px;line-height:1.7">
            Paste a YouTube URL or upload a meeting recording in the sidebar. Select your summary preset and language, then hit <strong>Analyse Meeting</strong>.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">⏱️ Timestamps</span>
            <span class="badge badge-cyan">📑 Chapters</span>
            <span class="badge badge-green">🚀 One-Click Export</span>
            <span class="badge badge-amber">📁 Session Library</span>
        </div>
    </div>""", unsafe_allow_html=True)