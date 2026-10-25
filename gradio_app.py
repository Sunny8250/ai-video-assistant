import os
import sys
import shutil

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure FFmpeg is available in PATH
if not shutil.which("ffmpeg"):
    winget_ffmpeg = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin")
    if os.path.exists(winget_ffmpeg):
        os.environ["PATH"] = winget_ffmpeg + os.pathsep + os.environ.get("PATH", "")

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from utils.audio_processor import process_input, sanitize_youtube_url
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


CUSTOM_CSS = """
/* ── Container Aesthetics ── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.hero-box {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 18px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 650px;
    margin: 0 auto;
}

.section-card {
    background: #111118;
    border: 1px solid #2a2a3a;
    border-radius: 14px;
    padding: 1.25rem;
    margin-top: 1rem;
}
"""


def analyze_video(youtube_url, uploaded_file, language, progress=gr.Progress(track_tqdm=True)):
    source = None
    if youtube_url and youtube_url.strip():
        source = sanitize_youtube_url(youtube_url.strip())
    elif uploaded_file is not None:
        source = uploaded_file

    if not source:
        raise gr.Error("Please provide either a YouTube URL or upload an audio/video file.")

    progress(0.05, desc="[1/5] Ingesting audio...")
    chunks = process_input(source)

    lang_choice = "hinglish" if "Hinglish" in language else "english"
    engine_name = "Sarvam AI (Hinglish)" if lang_choice == "hinglish" else "Whisper"
    progress(0.25, desc=f"[2/5] Transcribing via {engine_name}...")
    transcript = transcribe_all(chunks, language=lang_choice)

    progress(0.65, desc="[3/5] Generating meeting title & executive summary...")
    title = generate_title(transcript)
    summary = summarize(transcript)

    progress(0.85, desc="[4/5] Extracting action items, decisions & questions...")
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    progress(0.95, desc="[5/5] Building RAG vector index for interactive chat...")
    rag_chain = build_rag_chain(transcript)

    progress(1.0, desc="Complete!")

    title_display = f"### 📌 {title}"
    chat_init = [
        {"role": "assistant", "content": f"Hi! I've indexed **\"{title}\"**. Ask me any question about this meeting or video!"}
    ]

    return (
        title_display,
        summary,
        action_items,
        decisions,
        questions,
        transcript,
        rag_chain,
        chat_init,
        gr.update(visible=True)
    )


def chat_with_meeting(user_message, history, rag_chain_state):
    if not rag_chain_state:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ Please analyse a video first before chatting."})
        return history, ""

    if not user_message or not user_message.strip():
        return history, ""

    history.append({"role": "user", "content": user_message})
    try:
        answer = ask_question(rag_chain_state, user_message)
    except Exception as e:
        answer = f"⚠️ Error answering question: {str(e)}"

    history.append({"role": "assistant", "content": answer})
    return history, ""


with gr.Blocks(title="AI Video Assistant") as demo:
    rag_chain_state = gr.State(None)

    gr.HTML("""
    <div class="hero-box">
        <div class="hero-title">🎬 AI Video Assistant</div>
        <div class="hero-subtitle">
            Instant AI transcription, executive summaries, action item extraction, and intelligent Q&A with your videos and meetings.
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 📥 1. Select Input Source")
                yt_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1
                )
                gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>— OR UPLOAD LOCAL FILE —</div>")
                file_input = gr.File(
                    label="Upload Audio / Video (MP4, MP3, WAV, M4A, etc.)",
                    file_types=["audio", "video"],
                    type="filepath"
                )

                language_select = gr.Radio(
                    choices=["English (Whisper GPU)", "Hinglish (Sarvam AI)"],
                    value="English (Whisper GPU)",
                    label="Transcription Language & Engine"
                )

                analyze_btn = gr.Button("🚀 Analyse Video", variant="primary", size="lg")

        with gr.Column(scale=6):
            title_banner = gr.Markdown("### 📌 Session Results will appear here", visible=True)

            with gr.Tabs(visible=False) as results_tabs:
                with gr.Tab("📋 Summary"):
                    summary_out = gr.Markdown()
                with gr.Tab("✅ Action Items"):
                    actions_out = gr.Markdown()
                with gr.Tab("🔑 Key Decisions"):
                    decisions_out = gr.Markdown()
                with gr.Tab("❓ Open Questions"):
                    questions_out = gr.Markdown()
                with gr.Tab("📝 Transcript"):
                    transcript_out = gr.Textbox(label="Full Transcript", lines=14)
                with gr.Tab("💬 Chat with Video"):
                    chatbot = gr.Chatbot(
                        label="Meeting Q&A Bot",
                        height=420
                    )
                    with gr.Row():
                        chat_msg = gr.Textbox(
                            placeholder="Ask a question about this meeting (e.g., What was decided about budget?)...",
                            label="Your Question",
                            scale=8,
                            lines=1
                        )
                        send_btn = gr.Button("Ask", variant="primary", scale=2)

    # Event Wiring
    analyze_btn.click(
        fn=analyze_video,
        inputs=[yt_input, file_input, language_select],
        outputs=[
            title_banner,
            summary_out,
            actions_out,
            decisions_out,
            questions_out,
            transcript_out,
            rag_chain_state,
            chatbot,
            results_tabs
        ]
    )

    send_btn.click(
        fn=chat_with_meeting,
        inputs=[chat_msg, chatbot, rag_chain_state],
        outputs=[chatbot, chat_msg]
    )

    chat_msg.submit(
        fn=chat_with_meeting,
        inputs=[chat_msg, chatbot, rag_chain_state],
        outputs=[chatbot, chat_msg]
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo"),
        css=CUSTOM_CSS
    )
