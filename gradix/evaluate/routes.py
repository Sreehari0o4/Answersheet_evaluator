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
    QuestionEvaluation,
    UserRole,
)
from ..preprocess.routes import split_numbered_answers
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


def evaluate_text_by_questions(cleaned_text: str, model_answer: str) -> tuple[float, str, list[dict]]:
    """Compute a mock evaluation using per-question segments.

    Returns a tuple of (final_score, feedback_text, per_question_details).
    Each item in per_question_details is a dict with keys:
    ``question_no``, ``answer_text``, and ``semantic``.
    """

    segments = split_numbered_answers(cleaned_text)

    question_details: list[dict] = []
    if segments:
        semantic_scores = []
        for q_no, ans_text in segments:
            sem = semantic_score(ans_text, model_answer)
            semantic_scores.append(sem)
            question_details.append(
                {
                    "question_no": q_no,
                    "answer_text": ans_text,
                    "score": sem,
                }
            )
        semantic_overall = sum(semantic_scores) / len(semantic_scores)
    else:
        sem = semantic_score(cleaned_text, model_answer)
        semantic_overall = sem
        question_details.append(
            {
                "question_no": 1,
                "answer_text": cleaned_text,
                "score": sem,
            }
        )

    # Mock component scores (same as before but driven by semantic_overall)
    keyword_score = 0.80
    grammar_score = 0.90

    final_score = round((semantic_overall + keyword_score + grammar_score) / 3.0, 2)

    per_q_lines = [
        f"Q{qd['question_no']}: {qd['score']:.2f}" for qd in question_details
    ]
    per_q_text = "; ".join(per_q_lines)

    feedback = (
        f"Per-question semantic: {per_q_text}. "
        f"Overall semantic: {semantic_overall:.2f}, Keyword: {keyword_score:.2f}, "
        f"Grammar: {grammar_score:.2f}. Final score: {final_score:.2f}."
    )

    return final_score, feedback, question_details


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

    final_score, feedback, per_q = evaluate_text_by_questions(
        extracted.cleaned_text,
        model_answer,
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
        db.session.flush()
    else:
        evaluation.model_answer_ref = model_answer
        evaluation.score = final_score
        evaluation.feedback = feedback
        evaluation.evaluated_on = datetime.utcnow()
        db.session.flush()

    # Upsert per-question evaluations linked to this evaluation
    existing_q = {
        (qe.question_no): qe for qe in evaluation.question_scores
    }
    for item in per_q:
        q_no = int(item["question_no"])
        q_score = float(item["score"])
        qe = existing_q.get(q_no)
        if qe is None:
            qe = QuestionEvaluation(
                eval_id=evaluation.eval_id,
                question_no=q_no,
                score=q_score,
            )
            db.session.add(qe)
        else:
            qe.score = q_score

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
