import contextlib
import io
import os
import re
import shutil
import sys
import wave
import requests
import whisper
from pydub import AudioSegment

# Ensure FFmpeg is available in PATH
if not shutil.which("ffmpeg"):
    winget_ffmpeg = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin")
    if os.path.exists(winget_ffmpeg):
        os.environ["PATH"] = winget_ffmpeg + os.pathsep + os.environ.get("PATH", "")

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():

    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 


def _get_wav_duration_seconds(path: str) -> float:
    """Read WAV file duration in seconds without loading entire audio into RAM."""
    try:
        with contextlib.closing(wave.open(path, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    try:
        size = os.path.getsize(path)
        return max(size / 32000.0, 1.0)
    except Exception:
        return 600.0


class _WhisperStdoutInterceptor:
    """
    Intercepts Whisper's verbose stdout stream:
    `[MM:SS.mmm --> MM:SS.mmm] <text>`
    and notifies status_callback in real-time as each ~30s segment finishes decoding.
    """
    def __init__(self, original_stdout, chunk_idx: int, total_chunks: int, chunk_duration: float, status_callback=None):
        self.original_stdout = original_stdout
        self.chunk_idx = chunk_idx
        self.total_chunks = max(total_chunks, 1)
        self.chunk_duration = max(chunk_duration, 1.0)
        self.status_callback = status_callback
        self.buffer = ""
        self._pattern = re.compile(r"\[(\d+):(\d+)(?:\.(\d+))?\s*-->\s*(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)")
        self._last_sec = -1

    def write(self, s):
        if self.original_stdout:
            self.original_stdout.write(s)
            try:
                self.original_stdout.flush()
            except Exception:
                pass

        self.buffer += s
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            m = self._pattern.search(line)
            if m and self.status_callback:
                try:
                    end_m = int(m.group(4))
                    end_s = int(m.group(5))
                    seg_end_sec = end_m * 60 + end_s

                    # Update UI if moved forward by at least 3 seconds of audio
                    if seg_end_sec - self._last_sec >= 3 or self._last_sec == -1:
                        self._last_sec = seg_end_sec
                        text_snippet = m.group(7).strip()
                        chunk_ratio = min(seg_end_sec / self.chunk_duration, 1.0)

                        # Map progress to 25% - 59% overall range
                        chunk_span = 34.0 / self.total_chunks
                        base_pct = 25.0 + (self.chunk_idx * chunk_span)
                        pct = int(min(base_pct + (chunk_ratio * chunk_span), 59.0))

                        time_str = f"{end_m:02d}:{end_s:02d}"
                        short_text = (text_snippet[:35] + "…") if len(text_snippet) > 35 else text_snippet
                        msg = f"Transcribing chunk {self.chunk_idx + 1}/{self.total_chunks} [{time_str}] \"{short_text}\""
                        self.status_callback(pct, msg)
                except Exception:
                    pass
        return len(s)

    def flush(self):
        if self.original_stdout:
            try:
                self.original_stdout.flush()
            except Exception:
                pass


try:
    import spaces
    @spaces.GPU(duration=120)
    def _run_whisper_inference(model, chunk_path):
        return model.transcribe(chunk_path, task="transcribe", verbose=True)
except Exception:
    def _run_whisper_inference(model, chunk_path):
        return model.transcribe(chunk_path, task="transcribe", verbose=True)


def _format_time(seconds: float) -> str:
    """Format seconds into [MM:SS] or [HH:MM:SS]."""
    total_sec = max(int(seconds), 0)
    m = total_sec // 60
    s = total_sec % 60
    h = m // 60
    if h > 0:
        m = m % 60
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def transcribe_chunk_whisper(chunk_path: str, chunk_idx: int = 0, total_chunks: int = 1, status_callback=None, offset_sec: float = 0.0) -> tuple[str, str]:
    model = load_model()  
    duration = _get_wav_duration_seconds(chunk_path)

    old_stdout = sys.stdout
    interceptor = _WhisperStdoutInterceptor(old_stdout, chunk_idx, total_chunks, duration, status_callback)
    sys.stdout = interceptor
    try:
        result = _run_whisper_inference(model, chunk_path)
    finally:
        sys.stdout = old_stdout
        if interceptor.buffer.strip():
            interceptor.write("\n")

    raw_text = result.get("text", "").strip()
    segments = result.get("segments", [])
    
    timestamped_lines = []
    if segments:
        for seg in segments:
            seg_start = offset_sec + float(seg.get("start", 0.0))
            ts_str = _format_time(seg_start)
            seg_txt = seg.get("text", "").strip()
            if seg_txt:
                timestamped_lines.append(f"{ts_str} {seg_txt}")
    
    timestamped_text = "\n".join(timestamped_lines) if timestamped_lines else raw_text
    return raw_text, timestamped_text  


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str, chunk_idx: int = 0, total_chunks: int = 1, status_callback=None, offset_sec: float = 0.0) -> tuple[str, str]:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    timestamped_lines = []
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    chunk_span = 34.0 / max(total_chunks, 1)
    base_pct = 25.0 + (chunk_idx * chunk_span)

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            if status_callback:
                piece_pct = int(base_pct + ((i / max(total_pieces, 1)) * chunk_span))
                status_callback(piece_pct, f"Sarvam AI piece {i + 1}/{total_pieces} (chunk {chunk_idx + 1}/{total_chunks})...")
            piece_text = _send_to_sarvam(piece_path).strip()
            if piece_text:
                full_text += piece_text + " "
                seg_time = offset_sec + (i * SARVAM_PIECE_SECONDS)
                timestamped_lines.append(f"{_format_time(seg_time)} {piece_text}")
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    timestamped_text = "\n".join(timestamped_lines) if timestamped_lines else full_text.strip()
    return full_text.strip(), timestamped_text


def transcribe_chunk(chunk_path: str, language: str = "english", chunk_idx: int = 0, total_chunks: int = 1, status_callback=None, offset_sec: float = 0.0) -> tuple[str, str]:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    Returns (raw_text, timestamped_text).
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path, chunk_idx=chunk_idx, total_chunks=total_chunks, status_callback=status_callback, offset_sec=offset_sec)
    return transcribe_chunk_whisper(chunk_path, chunk_idx=chunk_idx, total_chunks=total_chunks, status_callback=status_callback, offset_sec=offset_sec)


def transcribe_all(chunks: list, language: str = "english", status_callback=None, return_timestamps: bool = False):
    """Transcribe all chunks and optionally return both raw transcript and timestamped transcript."""
    full_transcript = "" 
    full_timestamped = []

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")
    if status_callback:
        status_callback(22, f"Preparing {engine} for transcription...")

    total_chunks = len(chunks)
    current_offset = 0.0

    for i, chunk in enumerate(chunks):  
        pct = int(25 + ((i) / max(total_chunks, 1)) * 34)
        msg = f"Transcribing chunk {i + 1}/{total_chunks} with {engine}..."
        print(msg)
        if status_callback:
            status_callback(pct, msg)

        raw_chunk, ts_chunk = transcribe_chunk(
            chunk,
            language=language,
            chunk_idx=i,
            total_chunks=total_chunks,
            status_callback=status_callback,
            offset_sec=current_offset,
        )  

        full_transcript += raw_chunk + " "
        if ts_chunk:
            full_timestamped.append(ts_chunk)

        # Advance timeline offset by chunk duration
        chunk_dur = _get_wav_duration_seconds(chunk)
        current_offset += chunk_dur

    raw_result = full_transcript.strip()
    timestamped_result = "\n".join(full_timestamped).strip() if full_timestamped else raw_result

    word_count = len(raw_result.split())
    done_msg = f"Transcription complete — {word_count} words transcribed."
    print(done_msg)
    if status_callback:
        status_callback(60, done_msg)

    if return_timestamps:
        return raw_result, timestamped_result
    return raw_result  
