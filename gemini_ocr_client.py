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
import json
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
    """Run OCR on a local image or scanned-PDF file using Gemini.

    - For image formats (JPG/PNG, etc.), passes a PIL image.
    - For PDFs, uploads the file to Gemini and lets the model handle
      the scanned pages (including images) directly.

    Returns plain extracted text. Any evaluation or scoring logic
    should be applied by the calling code.
    """

    api_key = _get_api_key()

    # Configure SDK once per call (cheap enough for our usage pattern).
    genai.configure(api_key=api_key)

    # 2.5 Flash gives good accuracy for extraction at a reasonable cost.
    model = genai.GenerativeModel(model_name)

    ext = os.path.splitext(image_path)[1].lower()

    prompt = (
        "You are an OCR engine for exam answer sheets. "
        "Extract ALL readable handwritten and printed text from this document. "
        "Keep question numbers and line breaks where possible. "
        "Return ONLY the extracted text, with no explanations, comments, or labels."
    )

    if ext == ".pdf":
        # Upload the scanned PDF so Gemini can process its pages
        # (including embedded images/handwriting).
        file_obj = genai.upload_file(path=image_path)
        response = model.generate_content(
            [prompt, file_obj],
            generation_config={
                "temperature": 0.1,
            },
        )
    else:
        img = Image.open(image_path)
        response = model.generate_content(
            [prompt, img],
            generation_config={
                "temperature": 0.1,
            },
        )

    text: Any = response.text or ""
    return str(text).strip()


def extract_students_from_image(
    image_path: str,
    *,
    model_name: str = "gemini-2.5-flash",
) -> list[dict[str, str]]:
    """Use Gemini Vision to extract students from a handwritten list image.

    The image is expected to contain one student per row with a name and
    a roll number (for example "SREERAG - 20422088"). Separators can be
    dashes, colons, or just spaces; Gemini will infer the structure.

    Returns a list of dicts with keys ``name`` and ``roll_no``. Any
    parsing/validation beyond that should be handled by the caller.
    """

    api_key = _get_api_key()
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)

    img = Image.open(image_path)

    prompt = (
        "You are reading a photo of a handwritten student list. "
        "Each entry has a student name and a roll number. "
        "Extract all clearly readable entries and return ONLY JSON with "
        "this structure: {\"students\":[{\"name\":str,\"roll_no\":str}]}. "
        "Do not include any explanations or extra keys. "
        "Trim whitespace; keep roll_no exactly as written (digits only)."
    )

    response = model.generate_content(
        [prompt, img],
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    raw: Any = response.text or "{}"
    try:
        data = json.loads(str(raw))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Gemini student extraction returned invalid JSON: {exc}")

    students = data.get("students") or []
    out: list[dict[str, str]] = []
    for item in students:
        name = str(item.get("name") or "").strip()
        roll_no = str(item.get("roll_no") or "").strip()
        if not name or not roll_no:
            continue
        out.append({"name": name, "roll_no": roll_no})

    return out


def evaluate_answers_with_gemini(
    per_question_items: Iterable[dict[str, Any]],
    *,
    model_name: str = "gemini-2.5-flash",
    sheet_image_path: str | None = None,
) -> dict[str, Any]:
    """Call Gemini to score answers per question.

    ``per_question_items`` is an iterable of dicts with keys:

        - question_no (int)
        - question_text (str)
        - model_answer (str, optional)
        - max_marks (float | None)
        - student_answer (str)

    Behaviour:

    - If at least one item provides a non-empty ``model_answer``,
        Gemini is instructed to grade *strictly against the model
        answer*.
    - If all ``model_answer`` fields are empty/omitted, Gemini is
        instructed to grade using only the question text and max marks,
        based on typical expectations for that subject.

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

    # Decide whether we are grading strictly against provided model
    # answers or more generically using only question text.
    has_any_model_answer = any(
        bool((item.get("model_answer") or "").strip()) for item in items
    )

    lines: list[str] = []
    if has_any_model_answer:
        lines.append(
            "You are an experienced exam evaluator. For each question you are given "
            "the question text, a model answer, the maximum marks, and the student's "
            "answer. Grade strictly against the model answer. "
            "Student answers may be written as bullet points, in table form, or "
            "partly as diagrams; treat those as normal content and grade their "
            "meaning, not their format."
        )
    else:
        lines.append(
            "You are an experienced exam evaluator. For each question you are given "
            "the question text, the maximum marks, and the student's answer. "
            "There is no explicit model answer; grade based on what a well-" \
            "prepared student should write for that question, focusing on "
            "conceptual correctness, completeness, and clarity. "
            "Student answers may include bullet lists, tables, or diagrams; "
            "treat these as valid content, not as missing answers."
        )

    # Step-wise marking rules for numerical/mathematical questions.
    lines.append(
        "When a question is mathematical or numerical, apply strict university-"
        "style step-wise marking: award marks for correct formulas, substitutions, "
        "intermediate steps, and the final result; do not give full marks for only "
        "the final answer if steps or formulas are missing; deduct marks when "
        "calculations or reasoning steps are skipped; give partial marks when the "
        "method is correct but there are minor arithmetic mistakes; and if the "
        "final answer is correct but steps are incomplete, deduct 1–2 marks as "
        "appropriate."
    )

    # In general, mark strictly rather than generously: vague, off-topic,
    # or incomplete answers should receive low scores, and full marks should
    # be reserved only for answers that clearly meet all key points expected
    # for the given max_marks.
    lines.append(
        "Be strict, not lenient: only award full marks when the student's answer "
        "is complete and clearly correct; give low or zero marks when important "
        "steps, justifications, or key concepts are missing."
    )

    lines.append(
        "For each question, output a JSON object with: question_no (int), "
        "score (float between 0 and max_marks, inclusive), and feedback (short "
        "explanation). Use the full range [0, max_marks] where appropriate: "
        "answers that are mostly correct should receive a score close to "
        "max_marks, not a tiny decimal. Do not say that a question is "
        "unanswered if there is any non-trivial student text; in that case, "
        "assign at least some partial marks if any relevant points are present."
    )
    lines.append(
        "Finally, also include total_score as the sum of per-question scores. "
        "Return ONLY JSON with this structure: "
        "{\"questions\":[{\"question_no\":int,\"score\":float,\"feedback\":str}],"
        " \"total_score\": float}."
    )

    if sheet_image_path:
        lines.append(
            "You also have the full scanned answer sheet attached as an image or "
            "PDF. When grading each question, read the student's answer directly "
            "from the sheet, including any ray diagrams, graphs, labelled figures, "
            "or other visual elements. If the OCR text above misses details that "
            "are clearly shown in the diagrams or handwriting, use the sheet image "
            "as the source of truth."
        )
    lines.append("")

    for item in items:
        q_no = item.get("question_no")
        q_text = item.get("question_text") or ""
        m_ans = (item.get("model_answer") or "").strip()
        max_marks = item.get("max_marks")
        s_ans = item.get("student_answer") or ""

        marks_part = f"Max marks: {max_marks}" if max_marks is not None else "Max marks: use a 0-1 scale"
        lines.append(f"Question {q_no}:")
        lines.append(q_text)
        if m_ans:
            lines.append(f"Model answer: {m_ans}")
        lines.append(marks_part)
        if s_ans.strip():
            lines.append(f"Student answer: {s_ans}")
        else:
            lines.append("Student answer: [UNANSWERED]")
        lines.append("")

    prompt = "\n".join(lines)

    # If a sheet image/PDF is provided, call Gemini in multimodal mode so
    # diagrams and handwritten content on the sheet can be considered during
    # grading. Otherwise, fall back to text-only grading.
    if sheet_image_path:
        ext = os.path.splitext(sheet_image_path)[1].lower()
        if ext == ".pdf":
            file_obj = genai.upload_file(path=sheet_image_path)
            gen_inputs: list[Any] = [prompt, file_obj]
        else:
            img = Image.open(sheet_image_path)
            gen_inputs = [prompt, img]
    else:
        gen_inputs = [prompt]

    response = model.generate_content(
        gen_inputs,
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    import json

    text: Any = response.text or "{}"
    return json.loads(str(text))
