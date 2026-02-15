"""Gemini helpers for OCR and evaluation.

This replaces the previous Scripily integration and also provides
an optional Gemini-based evaluator for per-question scoring.

Usage (OCR only):
    from gemini_ocr_client import extract_text, GeminiConfigError
    text = extract_text("/absolute/path/to/image.jpg")

Usage (evaluation):
    from gemini_ocr_client import evaluate_answers_with_gemini
    result = evaluate_answers_with_gemini(per_question_items)

Configuration (loaded via .env):
    - GEMINI_API_KEY or GOOGLE_API_KEY must be set.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from PIL import Image
import google.generativeai as genai


class GeminiConfigError(RuntimeError):
    """Configuration or environment issue for Gemini OCR."""


def _get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GeminiConfigError(
            "GEMINI_API_KEY or GOOGLE_API_KEY must be set in the environment/.env",
        )
    return api_key


def extract_text(image_path: str, *, model_name: str = "gemini-2.5-flash") -> str:
    """Run OCR on a local image file using Gemini.

    Returns plain extracted text. Any evaluation or scoring logic
    should be applied by the calling code.
    """

    api_key = _get_api_key()

    # Configure SDK once per call (cheap enough for our usage pattern).
    genai.configure(api_key=api_key)

    # 2.5 Flash gives good accuracy for extraction at a reasonable cost.
    model = genai.GenerativeModel(model_name)

    img = Image.open(image_path)

    prompt = (
        "You are an OCR engine for exam answer sheets. "
        "Extract ALL readable handwritten and printed text from this image. "
        "Keep question numbers and line breaks where possible. "
        "Return ONLY the extracted text, with no explanations, comments, or labels."
    )

    response = model.generate_content(
        [prompt, img],
        generation_config={
            "temperature": 0.1,
        },
    )

    text: Any = response.text or ""
    return str(text).strip()


def evaluate_answers_with_gemini(
    per_question_items: Iterable[dict[str, Any]],
    *,
    model_name: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """Call Gemini to score answers per question.

    ``per_question_items`` is an iterable of dicts with keys:

        - question_no (int)
        - question_text (str)
        - model_answer (str)
        - max_marks (float | None)
        - student_answer (str)

    Returns a JSON-like dict of the form::

        {
          "questions": [
            {"question_no": int, "score": float, "feedback": str}
          ],
          "total_score": float
        }
    """

    api_key = _get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    items = list(per_question_items)

    lines: list[str] = []
    lines.append(
        "You are an experienced exam evaluator. For each question you are given "
        "the question text, a model answer, the maximum marks, and the student's "
        "answer. Grade strictly against the model answer."
    )
    lines.append(
        "For each question, output a JSON object with: question_no (int), "
        "score (float between 0 and max_marks, inclusive), and feedback (short "
        "explanation). Use the full range [0, max_marks] where appropriate: "
        "answers that are mostly correct should receive a score close to "
        "max_marks, not a tiny decimal."
    )
    lines.append(
        "Finally, also include total_score as the sum of per-question scores. "
        "Return ONLY JSON with this structure: "
        "{\"questions\":[{\"question_no\":int,\"score\":float,\"feedback\":str}],"
        " \"total_score\": float}."
    )
    lines.append("")

    for item in items:
        q_no = item.get("question_no")
        q_text = item.get("question_text") or ""
        m_ans = item.get("model_answer") or ""
        max_marks = item.get("max_marks")
        s_ans = item.get("student_answer") or ""

        marks_part = f"Max marks: {max_marks}" if max_marks is not None else "Max marks: use a 0-1 scale"
        lines.append(f"Question {q_no}:")
        lines.append(q_text)
        lines.append(f"Model answer: {m_ans}")
        lines.append(marks_part)
        if s_ans.strip():
            lines.append(f"Student answer: {s_ans}")
        else:
            lines.append("Student answer: [UNANSWERED]")
        lines.append("")

    prompt = "\n".join(lines)

    response = model.generate_content(
        [prompt],
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    import json

    text: Any = response.text or "{}"
    return json.loads(str(text))
