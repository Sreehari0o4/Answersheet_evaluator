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
