"""Quick manual test for OpenAI GPT-4o OCR + evaluation.

Usage (after setting OPENAI_API_KEY in .env):

    py openai_ocr_test.py --sheet-id 1

This will:
- Look up the AnswerSheet with the given sheet_id
- Load the corresponding image file from the uploads folder
- Load the exam questions + model answers
- Call GPT-4o with the image and rubric
- Print extracted answers and tentative scores per question

This script does NOT write anything back to the DB; it is just for
experimentation so you can see how GPT-4o behaves on your real data.
"""

from __future__ import annotations

import argparse
import base64
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from gradix import create_app
from gradix.extensions import db
from gradix.models import AnswerSheet, ExamQuestion


# Load environment variables from .env at project root
load_dotenv()


@dataclass
class QuestionRubric:
    question_no: int
    question_text: str
    model_answer: str
    max_marks: float | None


def _encode_image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    # Infer a basic mime type from extension; good enough for testing.
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif ext == "png":
        mime = "image/png"
    else:
        mime = "application/octet-stream"
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_prompt(rubric: list[QuestionRubric]) -> str:
    lines: list[str] = []
    lines.append(
        "You are an examiner. The student answer sheet image will be provided. "
        "First, transcribe the student's answers, then evaluate them strictly "
        "against the given model answers and marks."
    )
    lines.append("")
    lines.append("Rubric (questions and model answers):")
    for q in rubric:
        marks_part = f" ({q.max_marks} marks)" if q.max_marks is not None else ""
        lines.append(f"Q{q.question_no}{marks_part}: {q.question_text}")
        lines.append(f"Model answer: {q.model_answer}")
        lines.append("")
    lines.append(
        "Return JSON with this structure only: "
        "{\"questions\":[{\"question_no\":int,\"extracted_answer\":str,"
        "\"score\":float,\"feedback\":str}], \"total_score\": float}."
    )
    lines.append(
        "If you cannot read an answer, set score=0 and feedback='Not legible'."
    )

    return "\n".join(lines)


def call_openai_vision(image_path: str, prompt: str) -> dict[str, Any]:
    # Lazy import so this file doesn't break for users without openai installed.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The 'openai' package is not installed. Run 'pip install openai'."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")

    client = OpenAI(api_key=api_key)

    data_url = _encode_image_to_data_url(image_path)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You read exam answer sheet images and grade them.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    content = completion.choices[0].message.content or "{}"
    import json

    return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenAI OCR + eval for a sheet")
    parser.add_argument("--sheet-id", type=int, required=True)
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        sheet = AnswerSheet.query.get(args.sheet_id)
        if sheet is None:
            raise SystemExit(f"No AnswerSheet with sheet_id={args.sheet_id}")

        upload_folder = app.config["UPLOAD_FOLDER"]
        filename = os.path.basename(sheet.file_path)
        image_path = os.path.join(upload_folder, filename)
        if not os.path.exists(image_path):
            raise SystemExit(f"File not found: {image_path}")

        # Build rubric from ExamQuestion rows
        questions = (
            ExamQuestion.query.filter_by(exam_id=sheet.exam_id)
            .order_by(ExamQuestion.question_no.asc())
            .all()
        )
        if not questions:
            raise SystemExit("No questions/rubric defined for this exam.")

        rubric = [
            QuestionRubric(
                question_no=q.question_no,
                question_text=q.question_text,
                model_answer=q.answer_text,
                max_marks=q.marks,
            )
            for q in questions
        ]

        prompt = build_prompt(rubric)

        print(f"Calling OpenAI on {image_path} ...")
        result = call_openai_vision(image_path, prompt)

        print("\n--- Raw JSON result ---\n")
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
