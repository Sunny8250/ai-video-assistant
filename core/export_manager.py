import csv
import io
import re
from datetime import datetime


def generate_markdown_report(data: dict) -> str:
    """Generate a comprehensive, beautifully formatted Markdown meeting minutes report."""
    title = data.get("title", "Meeting Report")
    created_at = data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source = data.get("source", "N/A")
    summary_mode = data.get("summary_mode", "Comprehensive Minutes")
    language = data.get("language", "English").capitalize()

    summary = data.get("summary", "No summary available.")
    chapters = data.get("chapters", "")
    action_items = data.get("action_items", "No action items extracted.")
    decisions = data.get("key_decisions", "No key decisions recorded.")
    questions = data.get("open_questions", "No open questions recorded.")
    transcript = data.get("timestamped_transcript") or data.get("transcript", "No transcript available.")

    md = f"""# 📝 Meeting Minutes: {title}

**📅 Date:** {created_at}  
**🎯 Summary Preset:** {summary_mode}  
**🌐 Language:** {language}  
**🔗 Source / Recording:** {source}  

---

## 📋 Executive Summary
{summary}

"""
    if chapters and chapters.strip():
        md += f"""---

## 📑 Timeline & Chapters
{chapters}

"""

    md += f"""---

## ✅ Action Items & Tasks
{action_items}

---

## 🔑 Key Decisions Made
{decisions}

---

## ❓ Open Questions & Follow-ups
{questions}

---

## 📝 Full Transcript (with Timestamps)
```text
{transcript}
```

---
*Report generated automatically by AI Video Assistant.*
"""
    return md.strip()


def generate_plain_text_report(data: dict) -> str:
    """Generate plain text meeting minutes for easy emailing or documentation."""
    title = data.get("title", "Meeting Report")
    created_at = data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = data.get("summary", "")
    action_items = data.get("action_items", "")
    decisions = data.get("key_decisions", "")
    questions = data.get("open_questions", "")

    txt = f"MEETING MINUTES: {title.upper()}\n"
    txt += f"Date: {created_at}\n"
    txt += f"Source: {data.get('source', 'N/A')}\n"
    txt += "=" * 60 + "\n\n"

    txt += "SUMMARY:\n"
    txt += summary + "\n\n"
    txt += "-" * 60 + "\n\n"

    txt += "ACTION ITEMS:\n"
    txt += action_items + "\n\n"
    txt += "-" * 60 + "\n\n"

    txt += "KEY DECISIONS:\n"
    txt += decisions + "\n\n"
    txt += "-" * 60 + "\n\n"

    txt += "OPEN QUESTIONS:\n"
    txt += questions + "\n\n"
    return txt


def generate_action_items_csv(action_items_text: str) -> str:
    """Parse action items text into a structured CSV string (Task, Owner, Deadline, Status)."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Task Description", "Owner / Assignee", "Deadline", "Status"])

    # Regex or line-by-line parsing
    lines = action_items_text.splitlines()
    item_count = 0
    current_task = ""
    current_owner = "Unassigned"
    current_deadline = "Not specified"

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue

        # Check for numbered item start: e.g. "1. Task description" or "- Task:"
        num_match = re.match(r"^(\d+)[\.\)]\s*(.*)", line_s)
        if num_match:
            # Save previous if any
            if current_task:
                item_count += 1
                writer.writerow([item_count, current_task, current_owner, current_deadline, "Pending"])
            current_task = num_match.group(2).strip()
            current_owner = "Unassigned"
            current_deadline = "Not specified"
        elif "owner" in line_s.lower() or "responsible" in line_s.lower():
            parts = line_s.split(":", 1)
            if len(parts) > 1:
                current_owner = parts[1].strip()
        elif "deadline" in line_s.lower() or "due" in line_s.lower():
            parts = line_s.split(":", 1)
            if len(parts) > 1:
                current_deadline = parts[1].strip()
        elif "task" in line_s.lower() and ":" in line_s:
            parts = line_s.split(":", 1)
            if len(parts) > 1 and not current_task:
                current_task = parts[1].strip()
        else:
            if not current_task:
                current_task = line_s.lstrip("-*• ")
            else:
                current_task += " " + line_s

    if current_task:
        item_count += 1
        writer.writerow([item_count, current_task, current_owner, current_deadline, "Pending"])

    # If parsing found nothing structured, add raw text
    if item_count == 0 and action_items_text.strip():
        writer.writerow([1, action_items_text.strip().replace("\n", " "), "Unassigned", "Not specified", "Pending"])

    return output.getvalue()


def generate_notion_slack_tasks(action_items_text: str) -> str:
    """Format action items as markdown task checkboxes ready for Notion, Slack, or GitHub."""
    lines = action_items_text.splitlines()
    formatted = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Replace numbers or bullets with "- [ ] "
        cleaned = re.sub(r"^(\d+[\.\)]|[-*•])\s*", "", s)
        formatted.append(f"- [ ] {cleaned}")
    return "\n".join(formatted) if formatted else "- [ ] " + action_items_text.strip()
