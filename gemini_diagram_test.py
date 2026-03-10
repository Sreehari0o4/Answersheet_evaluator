"""Small helper script to test Gemini's ability to spot diagrams
question‑wise on a scanned answer sheet.

Usage (from project root, venv activated):

    py gemini_diagram_test.py PATH/TO/SHEET.pdf --questions 1-6

Environment:
    - Set GEMINI_API_KEY or GOOGLE_API_KEY in .env or environment.

This script does NOT touch your database; it only prints JSON to stdout
so you can see what Gemini thinks for each question.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable

from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai


def _get_api_key() -> str:
    # Load .env from project root so GEMINI_API_KEY/GOOGLE_API_KEY are available
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
    if os.path.exists(env_path):
        load_dotenv(env_path)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY must be set.")
    return api_key


def parse_question_spec(spec: str) -> list[int]:
    """Parse a question spec like "1-6" or "1,3,4,6" into a list of ints."""

    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            try:
                out.add(int(part))
            except ValueError:
                continue
    return sorted(out)


def build_diagram_prompt(question_nos: Iterable[int]) -> str:
    lines: list[str] = []
    lines.append(
        "You are checking a scanned exam answer sheet. "
        "For each question number listed below, look at the attached image/PDF "
        "and decide whether the student's answer for that question includes any "
        "meaningful diagram, ray diagram, graph, labelled figure, circuit "
        "diagram, or sketch. Use only what you SEE on the sheet, not "
        "assumptions from the question text."
    )
    lines.append(
        "Return ONLY JSON with this structure: {\"questions\":[{"
        "\"question_no\":int,\"has_diagram\":bool,\"reason\":str}]}. "
        "The reason should briefly explain what you saw (e.g. 'ray diagram "
        "with labelled axes') or why you think there is no diagram."
    )
    lines.append("")

    for q_no in question_nos:
        lines.append(f"Question {q_no}:")
        lines.append("(Check the region of the sheet where answer to this question is written.)")
        lines.append("")

    return "\n".join(lines)


def run_diagram_check(sheet_path: str, question_nos: list[int], model_name: str = "gemini-2.5-flash") -> dict[str, Any]:
    api_key = _get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = build_diagram_prompt(question_nos)

    ext = os.path.splitext(sheet_path)[1].lower()
    if ext == ".pdf":
        file_obj = genai.upload_file(path=sheet_path)
        gen_inputs: list[Any] = [prompt, file_obj]
    else:
        img = Image.open(sheet_path)
        gen_inputs = [prompt, img]

    resp = model.generate_content(
        gen_inputs,
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    raw = resp.text or "{}"
    try:
        data = json.loads(str(raw))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Gemini returned invalid JSON: {exc}\nRaw: {raw!r}")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Gemini diagram detection per question.")
    parser.add_argument("sheet", help="Path to scanned answer sheet (PDF or image).")
    parser.add_argument(
        "--questions",
        default="1-6",
        help="Question numbers to check, e.g. '1-6' or '1,3,4,6'. Default: 1-6.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name to use (default: gemini-2.5-flash).",
    )

    args = parser.parse_args()

    sheet_path = os.path.abspath(args.sheet)
    if not os.path.exists(sheet_path):
        raise SystemExit(f"Sheet file not found: {sheet_path}")

    q_nos = parse_question_spec(args.questions)
    if not q_nos:
        raise SystemExit("No valid question numbers parsed from --questions.")

    result = run_diagram_check(sheet_path, q_nos, model_name=args.model)

    # Pretty-print so it is easy to inspect.
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
