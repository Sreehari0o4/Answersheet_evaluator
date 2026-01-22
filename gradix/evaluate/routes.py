from http import HTTPStatus
from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import (
    AnswerSheet,
    AnswerSheetStatus,
    Evaluation,
    ExtractedText,
    UserRole,
)
from ..rbac import role_required


evaluate_bp = Blueprint("evaluate", __name__, url_prefix="/evaluate")


def semantic_score(student_text: str, model_answer: str) -> float:
    """Mock semantic similarity function.

    NOTE: Phase I implementation ignores ``student_text`` and
    ``model_answer`` and returns a constant score. Replace with
    real embedding-based similarity in a later phase.
    """
    _ = (student_text, model_answer)
    return 0.78


@evaluate_bp.post("/<int:sheet_id>")
@jwt_required()
@role_required({UserRole.TEACHER})
def evaluate_sheet(sheet_id: int):
    sheet = AnswerSheet.query.get(sheet_id)
    if sheet is None:
        return (
            jsonify({"message": "AnswerSheet not found."}),
            HTTPStatus.NOT_FOUND,
        )

    if sheet.status != AnswerSheetStatus.PENDING:
        return (
            jsonify({"message": "Only Pending answer sheets can be evaluated."}),
            HTTPStatus.BAD_REQUEST,
        )

    extracted = sheet.extracted_text
    if extracted is None:
        return (
            jsonify({"message": "No extracted text found. Run OCR and preprocessing first."}),
            HTTPStatus.BAD_REQUEST,
        )

    data = request.get_json(silent=True) or {}
    model_answer = data.get("model_answer")
    if not model_answer:
        return (
            jsonify({"message": "'model_answer' is required in request body."}),
            HTTPStatus.BAD_REQUEST,
        )

    # Mock component scores (all values are dummy for Phase I)
    semantic = semantic_score(extracted.cleaned_text, model_answer)
    keyword_score = 0.80
    grammar_score = 0.90

    # Simple weighted average (equal weights for mock)
    final_score = round((semantic + keyword_score + grammar_score) / 3.0, 2)

    feedback = (
        f"Semantic: {semantic:.2f}, Keyword: {keyword_score:.2f}, "
        f"Grammar: {grammar_score:.2f}. Final score: {final_score:.2f}."
    )

    evaluation = Evaluation.query.filter_by(text_id=extracted.text_id).first()
    if evaluation is None:
        evaluation = Evaluation(
            text_id=extracted.text_id,
            model_answer_ref=model_answer,
            score=final_score,
            feedback=feedback,
            evaluated_on=datetime.utcnow(),
        )
        db.session.add(evaluation)
    else:
        evaluation.model_answer_ref = model_answer
        evaluation.score = final_score
        evaluation.feedback = feedback
        evaluation.evaluated_on = datetime.utcnow()

    sheet.status = AnswerSheetStatus.GRADED

    db.session.commit()

    return (
        jsonify(
            {
                "eval_id": evaluation.eval_id,
                "text_id": evaluation.text_id,
                "model_answer_ref": evaluation.model_answer_ref,
                "score": evaluation.score,
                "feedback": evaluation.feedback,
                "evaluated_on": evaluation.evaluated_on.isoformat(),
                "sheet_status": sheet.status.value,
            }
        ),
        HTTPStatus.CREATED,
    )
