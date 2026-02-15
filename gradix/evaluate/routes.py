from http import HTTPStatus
from datetime import datetime
import logging
import re

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
logger = logging.getLogger(__name__)


def semantic_score(student_text: str, model_answer: str) -> float:
    """Mock semantic similarity function.

    NOTE: Phase I implementation ignores ``student_text`` and
    ``model_answer`` and returns a constant score. Replace with
    real embedding-based similarity in a later phase.
    """
    _ = (student_text, model_answer)
    return 0.78


def evaluate_text_by_questions(text: str, model_answer: str) -> tuple[float, str, list[dict]]:
    """Compute a mock evaluation using per-question segments.

    Returns a tuple of (final_score, feedback_text, per_question_details).
    Each item in per_question_details is a dict with keys:
    ``question_no``, ``answer_text``, and ``semantic``.
    """

    segments = split_numbered_answers(text)

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
        sem = semantic_score(text, model_answer)
        semantic_overall = sem
        question_details.append(
            {
                "question_no": 1,
                "answer_text": text,
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

    # Allow evaluation on newly uploaded and already graded sheets so that
    # scores can be recomputed if the evaluation logic or rubric changes.
    if sheet.status not in {AnswerSheetStatus.PENDING, AnswerSheetStatus.GRADED}:
        return (
            jsonify({"message": "Only Pending or Graded answer sheets can be evaluated."}),
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

    # First, split the *raw* OCR text into numbered answers so
    # question numbers are not lost by later grammar correction.
    segments = split_numbered_answers(extracted.raw_text)

    # Prefer Gemini-based evaluation when exam questions/rubric are defined.
    sheet_exam = sheet.exam
    exam_questions = sorted(
        getattr(sheet_exam, "questions", []),
        key=lambda q: q.question_no,
    )

    final_score: float
    feedback: str
    per_q: list[dict]

    if exam_questions:
        answers_by_q = {q_no: ans for q_no, ans in segments}

        try:  # pragma: no cover - external API
            from gemini_ocr_client import (
                GeminiConfigError,
                evaluate_answers_with_gemini,
            )

            payload_items = []
            for eq in exam_questions:
                payload_items.append(
                    {
                        "question_no": eq.question_no,
                        "question_text": eq.question_text,
                        "model_answer": eq.answer_text,
                        "max_marks": float(eq.marks) if eq.marks is not None else None,
                        "student_answer": answers_by_q.get(eq.question_no, ""),
                    }
                )

            gemini_result = evaluate_answers_with_gemini(payload_items)
            questions_out = gemini_result.get("questions", []) or []
            total_score = gemini_result.get("total_score")

            per_q = []
            for item in questions_out:
                # Robustly parse question number from Gemini output
                q_no_raw = item.get("question_no")
                q_no = None
                try:
                    if isinstance(q_no_raw, (int, float)):
                        q_no = int(q_no_raw)
                    else:
                        s = str(q_no_raw)
                        m = re.search(r"\\d+", s)
                        if m:
                            q_no = int(m.group(0))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    q_no = None
                if q_no is None:
                    continue
                try:
                    q_score = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    q_score = 0.0
                feedback_text = (item.get("feedback") or "").strip()
                ans_text = next(
                    (x["student_answer"] for x in payload_items if x["question_no"] == q_no),
                    "",
                )
                per_q.append(
                    {
                        "question_no": q_no,
                        "answer_text": ans_text,
                        "score": q_score,
                        "feedback": feedback_text,
                    }
                )

            if total_score is None:
                total_score = sum(p["score"] for p in per_q) if per_q else 0.0

            final_score = round(float(total_score), 2)
            per_q_lines = [f"Q{p['question_no']}: {p['score']:.2f}" for p in per_q]
            feedback = "LLM (Gemini) evaluation. " + "; ".join(per_q_lines)
        except GeminiConfigError as exc:
            logger.warning("Gemini evaluation misconfigured, falling back to mock scoring: %s", exc)
            final_score, feedback, per_q = evaluate_text_by_questions(
                extracted.raw_text,
                model_answer,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini evaluation failed, falling back to mock scoring: %s", exc)
            final_score, feedback, per_q = evaluate_text_by_questions(
                extracted.raw_text,
                model_answer,
            )
    else:
        # No structured exam questions; fall back to simple heuristic scoring.
        final_score, feedback, per_q = evaluate_text_by_questions(
            extracted.raw_text,
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

    # Upsert per-question evaluations linked to this evaluation,
    # aligned to the exam's defined questions when available.
    existing_q = {qe.question_no: qe for qe in evaluation.question_scores}

    per_q_by_no: dict[int, dict] = {}
    for item in per_q:
        try:
            q_no_int = int(item["question_no"])
        except (TypeError, ValueError):
            continue
        per_q_by_no[q_no_int] = item

    if exam_questions:
        # Use exam questions as the source of truth; mark
        # missing answers as unanswered.
        for eq in exam_questions:
            item = per_q_by_no.get(eq.question_no)
            if item is not None:
                try:
                    q_score = float(item["score"])
                except (TypeError, ValueError):
                    q_score = 0.0
                feedback_text = None
            else:
                q_score = 0.0
                feedback_text = "Unanswered"

            qe = existing_q.get(eq.question_no)
            if qe is None:
                qe = QuestionEvaluation(
                    eval_id=evaluation.eval_id,
                    question_no=eq.question_no,
                    score=q_score,
                    feedback=feedback_text,
                )
                db.session.add(qe)
            else:
                qe.score = q_score
                qe.feedback = feedback_text
    else:
        # Legacy behaviour: rely solely on numbered segments
        for item in per_q:
            try:
                q_no = int(item["question_no"])
                q_score = float(item["score"])
            except (TypeError, ValueError):
                continue
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
