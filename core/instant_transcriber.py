import re
import urllib.parse
import urllib.request
import json
from youtube_transcript_api import YouTubeTranscriptApi
from core.transcriber import _format_time


def extract_video_id(url_or_id: str) -> str | None:
    """Extract YouTube video ID from various YouTube URL formats or bare IDs."""
    if not url_or_id:
        return None

    cleaned = url_or_id.strip()

    # If it's already a clean 11-char ID
    if re.match(r"^[a-zA-Z0-9_-]{11}$", cleaned):
        return cleaned

    try:
        parsed = urllib.parse.urlparse(cleaned)
        if "youtube.com" in parsed.netloc:
            # e.g. /watch?v=JHm7BkRp3T8
            qs = urllib.parse.parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
            # e.g. /embed/JHm7BkRp3T8 or /shorts/JHm7BkRp3T8
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ["embed", "shorts", "v"]:
                return parts[1]
        elif "youtu.be" in parsed.netloc:
            # e.g. https://youtu.be/JHm7BkRp3T8
            parts = [p for p in parsed.path.split("/") if p]
            if parts:
                return parts[0]
    except Exception:
        pass

    # Regex fallback for embedded IDs
    m = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})(?:\?|&|\/|$)", cleaned)
    if m:
        return m.group(1)

    return None


def _fetch_via_youtube_transcript_api(video_id: str, preferred_languages: list[str]) -> tuple[str | None, str | None]:
    """Primary Tier: Fetch captions via youtube_transcript_api (fastest, < 1 second)."""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # 1. Try manually uploaded or preferred languages
        transcript_obj = None
        try:
            transcript_obj = transcript_list.find_transcript(preferred_languages)
        except Exception:
            # 2. Try generated transcripts
            try:
                transcript_obj = transcript_list.find_generated_transcript(preferred_languages)
            except Exception:
                # 3. Fallback to any first available transcript
                for t in transcript_list:
                    transcript_obj = t
                    break

        if not transcript_obj:
            return None, None

        snippets = transcript_obj.fetch()
        if not snippets:
            return None, None

        raw_parts = []
        timestamped_lines = []

        for item in snippets:
            txt = getattr(item, "text", "") or ""
            txt = txt.strip()
            if not txt:
                continue

            raw_parts.append(txt)
            start_sec = getattr(item, "start", 0.0)
            ts_str = _format_time(start_sec)
            timestamped_lines.append(f"{ts_str} {txt}")

        raw_text = " ".join(raw_parts).strip()
        timestamped_text = "\n".join(timestamped_lines).strip()

        return raw_text, timestamped_text
    except Exception as e:
        print(f"[Instant Mode] youtube_transcript_api note: {e}")
        return None, None


def _fetch_via_ytdlp_subtitles(url_or_id: str, preferred_languages: list[str]) -> tuple[str | None, str | None]:
    """Secondary Tier: Fetch captions using yt-dlp subtitle metadata without downloading media."""
    try:
        import yt_dlp
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "source_address": "0.0.0.0",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_url = url_or_id if url_or_id.startswith("http") else f"https://www.youtube.com/watch?v={url_or_id}"
            info = ydl.extract_info(video_url, download=False)
            if not info:
                return None, None

            subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}
            all_subs = {**auto_subs, **subs}
            if not all_subs:
                return None, None

            # Choose best language
            target_track = None
            for lang in preferred_languages:
                if lang in all_subs:
                    target_track = all_subs[lang]
                    break
            if not target_track:
                # Take first available
                target_track = next(iter(all_subs.values()))

            if not target_track:
                return None, None

            # Prefer json3 format for structured timestamps
            fmt = next((f for f in target_track if f.get("ext") == "json3"), target_track[0])
            sub_url = fmt.get("url")
            if not sub_url:
                return None, None

            req = urllib.request.Request(sub_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="replace")

            if fmt.get("ext") == "json3":
                data = json.loads(content)
                events = data.get("events", [])
                raw_parts = []
                timestamped_lines = []
                for ev in events:
                    if "segs" not in ev:
                        continue
                    text = "".join([s.get("utf8", "") for s in ev["segs"]]).strip()
                    if not text:
                        continue
                    start_sec = ev.get("tStartMs", 0) / 1000.0
                    raw_parts.append(text)
                    timestamped_lines.append(f"{_format_time(start_sec)} {text}")
                return " ".join(raw_parts).strip(), "\n".join(timestamped_lines).strip()
            else:
                # Simple VTT or text regex
                clean_lines = []
                for line in content.splitlines():
                    line = line.strip()
                    if not line or "-->" in line or line.startswith("WEBVTT") or line.isdigit():
                        continue
                    clean_lines.append(line)
                txt = " ".join(clean_lines).strip()
                return txt, txt

    except Exception as e:
        print(f"[Instant Mode] yt-dlp subtitle fallback note: {e}")
        return None, None


def has_youtube_captions(url_or_id: str) -> bool:
    """Quickly check if the YouTube video has subtitles/captions available."""
    video_id = extract_video_id(url_or_id)
    if not video_id:
        return False
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        for _ in transcript_list:
            return True
        return False
    except Exception:
        return False


def fetch_instant_transcript(url_or_id: str, preferred_languages: list[str] = None) -> tuple[str | None, str | None]:
    """
    Dual-Tier Instant Mode Subtitle Extraction:
    Tier 1: youtube_transcript_api (2-4 seconds, zero media download, direct YouTube caption endpoint)
    Tier 2: yt-dlp subtitle metadata fallback (if tier 1 encounters an edge case)

    Returns (raw_transcript, timestamped_transcript).
    If no subtitles exist, returns (None, None).
    """
    video_id = extract_video_id(url_or_id)
    if not video_id:
        return None, None

    if preferred_languages is None:
        preferred_languages = ["en", "en-US", "en-GB", "hi", "hi-orig", "en-orig"]

    # ── Tier 1: youtube_transcript_api ──
    raw_text, ts_text = _fetch_via_youtube_transcript_api(video_id, preferred_languages)
    if raw_text and len(raw_text.split()) >= 10:
        return raw_text, ts_text

    # ── Tier 2: yt-dlp subtitle extraction fallback ──
    raw_text, ts_text = _fetch_via_ytdlp_subtitles(url_or_id, preferred_languages)
    if raw_text and len(raw_text.split()) >= 10:
        return raw_text, ts_text

    return None, None
