import os
from http import HTTPStatus

from flask import Blueprint, current_app, jsonify
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import AnswerSheet, ExtractedText, UserRole
from ..rbac import role_required


ocr_bp = Blueprint("ocr", __name__, url_prefix="/ocr")


def run_ocr(file_path: str) -> tuple[str, float]:
    """Mocked OCR function.

    NOTE: Phase I implementation is a stub. It does not inspect the
    file at ``file_path``; it always returns the same dummy text and
    a fixed confidence score. Replace this with real OCR in Phase II.
    """
    _ = file_path  # unused in mock
    return "This is dummy extracted text", 0.92


@ocr_bp.post("/run/<int:sheet_id>")
@jwt_required()
@role_required({UserRole.TEACHER})
def ocr_run(sheet_id: int):
    sheet = AnswerSheet.query.get(sheet_id)
    if sheet is None:
        return (
            jsonify({"message": "AnswerSheet not found."}),
            HTTPStatus.NOT_FOUND,
        )

    # Build absolute file path; mock doesn't actually need it but we keep it logical
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    filename = os.path.basename(sheet.file_path)
    abs_path = os.path.join(upload_folder, filename)

    raw_text, confidence = run_ocr(abs_path)
    cleaned_text = raw_text.strip().lower()

    # Upsert ExtractedText for this sheet
    extracted = ExtractedText.query.filter_by(sheet_id=sheet.sheet_id).first()
    if extracted is None:
        extracted = ExtractedText(
            sheet_id=sheet.sheet_id,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extraction_confidence=confidence,
        )
        db.session.add(extracted)
    else:
        extracted.raw_text = raw_text
        extracted.cleaned_text = cleaned_text
        extracted.extraction_confidence = confidence

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
        HTTPStatus.CREATED,
    )
