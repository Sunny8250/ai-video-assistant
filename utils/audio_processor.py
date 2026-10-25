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

    Uses Node.js + yt-dlp-ejs runtime for JS challenge solving,
    plus cookies.txt if present for authentication.
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
        "js_runtimes": {"node": {}},
        "quiet": True,
        "no_warnings": False,
    }

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

    # Add cookies file for YouTube auth if present
    temp_cookie_path = None
    cookies_file = os.getenv("YOUTUBE_COOKIES_FILE", "").strip() or _DEFAULT_COOKIES_FILE
    if os.path.exists(cookies_file):
        # yt-dlp by default modifies and overwrites the cookiefile on exit.
        # If YouTube returns a bot-challenge or login-required response, yt-dlp
        # overwrites the user's authenticated cookies with stripped session cookies!
        # To prevent cookies.txt from ever being corrupted, we use a temporary copy.
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


