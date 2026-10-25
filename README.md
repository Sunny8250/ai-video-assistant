# 🎬 AI Video Assistant

An AI-powered meeting and video assistant that downloads, transcribes, summarizes, extracts action items & key decisions, and lets you chat with your meetings using LangChain RAG.

---

## ⚡ Features

- **YouTube & Local Video/Audio Ingestion**: Automatically downloads and extracts audio using `yt-dlp` and `FFmpeg`.
- **Speech-to-Text**:
  - **English**: High-accuracy local transcription using OpenAI Whisper with **NVIDIA CUDA GPU acceleration**.
  - **Hinglish**: Cloud translation and transcription via Sarvam AI API.
- **AI Meeting Summaries & Extraction**:
  - Executive summary and meeting title generation using Mistral AI.
  - Action items with assignees and deadlines.
  - Key decisions and open questions.
- **RAG Chatbot**: Chat interactively with the meeting transcript using Chroma vector database and Mistral LLM.
- **Streamlit Web UI & CLI**: Run through the modern dark-themed browser interface or command line.

---

## 🚀 Quick Start

### 1. Configure API Keys
Copy `.env.example` to `.env` (already created for you):
```env
# Required for summarization, action items, and RAG chat
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional: tiny, base, small, medium, large (defaults to small)
WHISPER_MODEL=small

# Optional: Only needed for Hinglish translation
SARVAM_API_KEY=your_sarvam_api_key_here
```
> Get a free Mistral API key at [console.mistral.ai](https://console.mistral.ai/).

### 2. Run the Streamlit Web Application
Double-click `run_app.bat` or run in terminal:
```powershell
.\.venv\Scripts\streamlit run app.py
```

### 3. Run via CLI
```powershell
.\.venv\Scripts\python main.py
```
