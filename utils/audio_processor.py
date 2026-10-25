import os
import shutil
import sys
import tempfile

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

# Ensure Node.js is available for JS challenge solving
if not shutil.which("node"):
    default_node_dir = r"C:\Program Files\nodejs"
    if os.path.exists(default_node_dir):
        os.environ["PATH"] = default_node_dir + os.pathsep + os.environ.get("PATH", "")

import urllib.parse

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Path to YouTube cookies file (Netscape format exported from browser extension)
# Override in .env with: YOUTUBE_COOKIES_FILE=cookies.txt
_DEFAULT_COOKIES_FILE = "cookies.txt"


def sanitize_youtube_url(url: str) -> str:
    """Extract clean single-video URL, stripping playlist & tracking parameters."""
    try:
        parsed = urllib.parse.urlparse(url)
        if "youtube.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return f"https://www.youtube.com/watch?v={qs['v'][0]}"
        elif "youtu.be" in parsed.netloc:
            video_id = parsed.path.strip("/")
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"
    except Exception:
        pass
    return url


def _build_ydl_opts(status_callback=None) -> dict:
    """Build yt-dlp options for YouTube audio download.

    Uses Node.js/Deno runtime for JS challenge solving if available,
    plus YOUTUBE_COOKIES secret or cookies.txt if present for authentication.
    """
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": False,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb", "web"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    elif shutil.which("deno"):
        opts["js_runtimes"] = {"deno": {}}

    if status_callback:
        def ydl_progress_hook(d):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes") or 0
                if total > 0:
                    pct = int(5 + (downloaded / total) * 10)  # 5% to 15%
                    down_mb = downloaded / (1024 * 1024)
                    tot_mb = total / (1024 * 1024)
                    status_callback(pct, f"Downloading audio: {down_mb:.1f}MB / {tot_mb:.1f}MB ({pct}%)")
            elif d.get("status") == "finished":
                status_callback(15, "Download complete. Extracting WAV audio...")

        opts["progress_hooks"] = [ydl_progress_hook]

    # Check for raw cookies in Streamlit secrets or environment
    raw_cookies = ""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "YOUTUBE_COOKIES" in st.secrets:
                raw_cookies = str(st.secrets["YOUTUBE_COOKIES"]).strip()
            elif "youtube_cookies" in st.secrets:
                raw_cookies = str(st.secrets["youtube_cookies"]).strip()
    except Exception:
        pass

    if not raw_cookies:
        raw_cookies = os.getenv("YOUTUBE_COOKIES", "").strip()

    temp_cookie_path = None
    if raw_cookies:
        temp_fd, temp_cookie_path = tempfile.mkstemp(suffix=".txt", prefix="ytdlp_env_cookies_")
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write(raw_cookies)
        opts["cookiefile"] = temp_cookie_path
        print("  [Cookies] Loaded cookies from secrets/environment")
    else:
        cookies_file = os.getenv("YOUTUBE_COOKIES_FILE", "").strip() or _DEFAULT_COOKIES_FILE
        if os.path.exists(cookies_file):
            temp_fd, temp_cookie_path = tempfile.mkstemp(suffix=".txt", prefix="ytdlp_cookies_")
            os.close(temp_fd)
            shutil.copyfile(cookies_file, temp_cookie_path)
            opts["cookiefile"] = temp_cookie_path
            print(f"  [Cookies] Using protected copy of: {cookies_file}")

    return opts, temp_cookie_path


def download_youtube_audio(url: str, status_callback=None) -> str:
    clean_url = sanitize_youtube_url(url)
    ydl_opts, temp_cookie_path = _build_ydl_opts(status_callback=status_callback)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=True)
            raw_filename = ydl.prepare_filename(info)
            base_path = os.path.splitext(raw_filename)[0]
            wav_path = base_path + ".wav"
            
            # Verify the file exists or locate it in DOWNLOAD_DIR
            if not os.path.exists(wav_path):
                title = info.get("title", "")
                for f in os.listdir(DOWNLOAD_DIR):
                    if f.endswith(".wav") and (title[:15] in f if title else False):
                        wav_path = os.path.join(DOWNLOAD_DIR, f)
                        break
        return wav_path
    except Exception as e:
        err_str = str(e)
        if any(keyword in err_str for keyword in ["403", "Forbidden", "confirm you're not a bot", "bot"]):
            raise RuntimeError(
                "YouTube has blocked audio download on the cloud server (HTTP 403: Forbidden / Bot Check).\n\n"
                "💡 To solve this:\n"
                "1. Upload your video or audio file directly using the 'Upload Audio / Video File' button in the sidebar (recommended).\n"
                "2. Or export your YouTube cookies and paste them into Streamlit Cloud App Settings -> Secrets as YOUTUBE_COOKIES."
            ) from e
        raise e
    finally:
        if temp_cookie_path and os.path.exists(temp_cookie_path):
            try:
                os.remove(temp_cookie_path)
            except Exception:
                pass



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str, status_callback=None) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        msg = "Detected YouTube URL. Downloading audio..."
        print(msg)
        if status_callback:
            status_callback(5, msg)
        wav_path = download_youtube_audio(source, status_callback=status_callback)
    else:
        msg = "Detected local file. Converting to 16kHz WAV..."
        print(msg)
        if status_callback:
            status_callback(5, msg)
        wav_path = convert_to_wav(source)

    chunk_msg = "Chunking audio..."
    print(chunk_msg)
    if status_callback:
        status_callback(15, chunk_msg)
    chunks = chunk_audio(wav_path)
    ready_msg = f"Audio ready — {len(chunks)} chunk(s) created."
    print(ready_msg)
    if status_callback:
        status_callback(20, ready_msg)
    return chunks


