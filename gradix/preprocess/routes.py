"""Text preprocessing and helpers.

This module also provides utilities to split OCR output into
numbered question/answer segments, which are used for per-question
evaluation and review in the web UI.
"""

import re
from http import HTTPStatus

from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import ExtractedText, UserRole
from ..rbac import role_required


preprocess_bp = Blueprint("preprocess", __name__, url_prefix="/")


def preprocess_text(raw_text: str) -> str:
    """Mock text preprocessing function.

    NOTE: Phase I implementation only lowercases text. Real
    normalization (lemmatization, stopwords, etc.) should replace
    this stub in a later phase.
    """
    return raw_text.lower()


def split_numbered_answers(text: str):
    """Split OCR text into (question_no, answer_text) segments.

    Assumes answers are numbered in the OCR output, e.g.::

        1. First answer text...
        2) Second answer text...

    If no numbering is detected, the whole text is treated as a
    single answer for question 1.
    """

    if not text or not text.strip():
        return []

    pattern = re.compile(r"(?m)^\s*(\d+)[\).\s]+")
    matches = list(pattern.finditer(text))

    if not matches:
        return [(1, text.strip())]

    segments = []
    for i, match in enumerate(matches):
        q_no = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        answer = text[start:end].strip()
        if answer:
            segments.append((q_no, answer))

    return segments


@preprocess_bp.post("preprocess/<int:sheet_id>")
@jwt_required()
@role_required({UserRole.TEACHER})
def preprocess(sheet_id: int):
    extracted = ExtractedText.query.filter_by(sheet_id=sheet_id).first()
    if extracted is None:
        return (
            jsonify({"message": "No extracted text found for this sheet. Run OCR first."}),
            HTTPStatus.BAD_REQUEST,
        )

    extracted.cleaned_text = preprocess_text(extracted.raw_text)
    db.session.commit()

    return (
        jsonify(
            {
                "text_id": extracted.text_id,
                "sheet_id": extracted.sheet_id,
                "raw_text": extracted.raw_text,
                "cleaned_text": extracted.cleaned_text,
                "extraction_confidence": extracted.extraction_confidence,
            }
        ),
        HTTPStatus.OK,
    )
