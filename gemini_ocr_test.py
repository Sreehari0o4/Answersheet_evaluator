"""Quick manual test for Google Gemini OCR + evaluation.

Usage (after setting GEMINI_API_KEY in .env):

    py gemini_ocr_test.py --sheet-id 1

This will:
- Look up the AnswerSheet with the given sheet_id
- Load the corresponding image file from the uploads folder
- Load the exam questions + model answers
- Call Gemini (via google-generativeai) with the image and rubric
- Print extracted answers and tentative scores per question

This script does NOT write anything back to the DB; it is just for
experimentation so you can see how Gemini behaves on your real data.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from gradix import create_app
from gradix.extensions import db  # noqa: F401  # ensure app models are loaded
from gradix.models import AnswerSheet, ExamQuestion


# Load environment variables from .env at project root
load_dotenv()


@dataclass
class QuestionRubric:
    question_no: int
    question_text: str
    model_answer: str
    max_marks: float | None


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
        '{"questions":[{"question_no":int,"extracted_answer":str,'
        '"score":float,"feedback":str}], "total_score": float}.'
    )
    lines.append(
        "If you cannot read an answer, set score=0 and feedback='Not legible'."
    )

    return "\n".join(lines)


def call_gemini_vision(image_path: str, prompt: str) -> dict[str, Any]:
    """Call Google Gemini 1.5 Flash with an image and prompt.

    Expects GEMINI_API_KEY or GOOGLE_API_KEY in the environment/.env.
    """

    try:
        import google.generativeai as genai
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The 'google-generativeai' and 'Pillow' packages are required. "
            "Run 'pip install -U google-generativeai pillow'."
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in environment/.env"
        )

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    img = Image.open(image_path)

    response = model.generate_content(
        [prompt, img],
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    )

    text = response.text or "{}"
    import json

    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Google Gemini OCR + eval for a sheet"
    )
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

        print(f"Calling Gemini on {image_path} ...")
        result = call_gemini_vision(image_path, prompt)

        print("\n--- Raw JSON result ---\n")
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
