from http import HTTPStatus
from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import (
    AnswerSheet,
    AnswerSheetStatus,
    Evaluation,
    UserRole,
)
from ..rbac import role_required


review_bp = Blueprint("review", __name__, url_prefix="/review")


@review_bp.get("/<int:sheet_id>")
@jwt_required()
@role_required({UserRole.TEACHER})
def get_review(sheet_id: int):
    sheet = AnswerSheet.query.get(sheet_id)
    if sheet is None:
        return (
            jsonify({"message": "AnswerSheet not found."}),
            HTTPStatus.NOT_FOUND,
        )

    if sheet.status not in {AnswerSheetStatus.GRADED, AnswerSheetStatus.REVIEWED}:
        return (
            jsonify({"message": "Only graded or reviewed answer sheets can be viewed for review."}),
            HTTPStatus.BAD_REQUEST,
        )

    extracted = sheet.extracted_text
    if extracted is None or extracted.evaluation is None:
        return (
            jsonify({"message": "No evaluation found. Run evaluation first."}),
            HTTPStatus.BAD_REQUEST,
        )

    evaluation = extracted.evaluation

    return (
        jsonify(
            {
                "sheet": {
                    "sheet_id": sheet.sheet_id,
                    "student_id": sheet.student_id,
                    "exam_id": sheet.exam_id,
                    "file_path": sheet.file_path,
                    "status": sheet.status.value,
                    "upload_date": sheet.upload_date.isoformat(),
                },
                "extracted_text": {
                    "text_id": extracted.text_id,
                    "raw_text": extracted.raw_text,
                    "cleaned_text": extracted.cleaned_text,
                    "extraction_confidence": extracted.extraction_confidence,
                },
                "evaluation": {
                    "eval_id": evaluation.eval_id,
                    "model_answer_ref": evaluation.model_answer_ref,
                    "score": evaluation.score,
                    "feedback": evaluation.feedback,
                    "evaluated_on": evaluation.evaluated_on.isoformat(),
                },
            }
        ),
        HTTPStatus.OK,
    )


@review_bp.post("/<int:sheet_id>/override")
@jwt_required()
@role_required({UserRole.TEACHER})
def override_review(sheet_id: int):
    sheet = AnswerSheet.query.get(sheet_id)
    if sheet is None:
        return (
            jsonify({"message": "AnswerSheet not found."}),
            HTTPStatus.NOT_FOUND,
        )

    if sheet.status != AnswerSheetStatus.GRADED:
        return (
            jsonify({"message": "Only graded answer sheets can be overridden."}),
            HTTPStatus.BAD_REQUEST,
        )

    extracted = sheet.extracted_text
    if extracted is None or extracted.evaluation is None:
        return (
            jsonify({"message": "No evaluation found to override."}),
            HTTPStatus.BAD_REQUEST,
        )

    evaluation = extracted.evaluation

    data = request.get_json(silent=True) or {}
    score = data.get("score")
    feedback = data.get("feedback")

    if score is None:
        return (
            jsonify({"message": "'score' is required to override."}),
            HTTPStatus.BAD_REQUEST,
        )

    try:
        score_val = float(score)
    except (TypeError, ValueError):
        return (
            jsonify({"message": "'score' must be a number."}),
            HTTPStatus.BAD_REQUEST,
        )

    evaluation.score = score_val
    if feedback is not None:
        evaluation.feedback = feedback
    evaluation.evaluated_on = datetime.utcnow()

    sheet.status = AnswerSheetStatus.REVIEWED

    db.session.commit()

    return (
        jsonify(
            {
                "sheet_id": sheet.sheet_id,
                "sheet_status": sheet.status.value,
                "eval_id": evaluation.eval_id,
                "score": evaluation.score,
                "feedback": evaluation.feedback,
                "evaluated_on": evaluation.evaluated_on.isoformat(),
            }
        ),
        HTTPStatus.OK,
    )
