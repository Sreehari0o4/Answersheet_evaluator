from http import HTTPStatus

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import Exam, ExamQuestion, Report, UserRole
from ..rbac import role_required


exam_bp = Blueprint("exam", __name__, url_prefix="/exam")


def _delete_exam_with_children(exam: Exam) -> None:
    """Delete an exam and all related dependent records.

    This removes:
    - Exam questions
    - Answer sheets (and their extracted text, evaluations, and question scores)
    - Reports linked to this exam
    """

    # Delete answer sheets and their nested objects
    for sheet in list(exam.answer_sheets):
        extracted = sheet.extracted_text
        if extracted is not None:
            evaluation = extracted.evaluation
            if evaluation is not None:
                # QuestionEvaluation rows are deleted via cascade from Evaluation
                db.session.delete(evaluation)
            db.session.delete(extracted)

        db.session.delete(sheet)

    # Delete reports linked to this exam
    Report.query.filter_by(exam_id=exam.exam_id).delete(synchronize_session=False)

    # Delete questions explicitly (in addition to backref cascade, for clarity)
    for q in list(exam.questions):
        db.session.delete(q)

    # Finally delete the exam itself
    db.session.delete(exam)


@exam_bp.post("/create")
@jwt_required()
@role_required({UserRole.ADMIN, UserRole.TEACHER})
def create_exam():
    data = request.get_json(silent=True) or {}
    subject = data.get("subject")
    max_marks = data.get("max_marks")
    rubric_details = data.get("rubric_details")
    questions_data = data.get("questions") or []

    if not subject or max_marks is None:
        return (
            jsonify({"message": "'subject' and 'max_marks' are required."}),
            HTTPStatus.BAD_REQUEST,
        )

    try:
        max_marks_int = int(max_marks)
    except (TypeError, ValueError):
        return (
            jsonify({"message": "'max_marks' must be an integer."}),
            HTTPStatus.BAD_REQUEST,
        )

    exam = Exam(subject=subject, max_marks=max_marks_int)
    db.session.add(exam)
    db.session.flush()

    rubric_parts: list[str] = []

    # Optional per-question details provided via API
    if isinstance(questions_data, list) and questions_data:
        for idx, item in enumerate(questions_data, start=1):
            q_no_raw = item.get("question_no")
            try:
                q_no = int(q_no_raw) if q_no_raw is not None else idx
            except (TypeError, ValueError):
                q_no = idx

            question_text = (item.get("question_text") or "").strip()
            answer_text = (item.get("answer_text") or "").strip()
            marks_raw = item.get("marks")

            marks_val = None
            if marks_raw is not None:
                try:
                    marks_val = float(marks_raw)
                except (TypeError, ValueError):
                    marks_val = None

            if not question_text or not answer_text:
                continue

            eq = ExamQuestion(
                exam_id=exam.exam_id,
                question_no=q_no,
                question_text=question_text,
                answer_text=answer_text,
                marks=marks_val,
            )
            db.session.add(eq)
            rubric_parts.append(
                (
                    f"Q{q_no}"
                    + (f" ({marks_val} marks)" if marks_val is not None else "")
                    + f". {question_text}\nAnswer: {answer_text}"
                )
            )

    # If we built a rubric from questions, prefer that; else fall back
    if rubric_parts:
        exam.rubric_details = "\n\n".join(rubric_parts)
    elif rubric_details is not None:
        exam.rubric_details = rubric_details

    db.session.commit()

    return (
        jsonify(
            {
                "exam_id": exam.exam_id,
                "subject": exam.subject,
                "max_marks": exam.max_marks,
                "rubric_details": exam.rubric_details,
            }
        ),
        HTTPStatus.CREATED,
    )


@exam_bp.get("/list")
@jwt_required()
def list_exams():
    exams = Exam.query.order_by(Exam.exam_id.desc()).all()
    return (
        jsonify(
            [
                {
                    "exam_id": e.exam_id,
                    "subject": e.subject,
                    "max_marks": e.max_marks,
                    "rubric_details": e.rubric_details,
                }
                for e in exams
            ]
        ),
        HTTPStatus.OK,
    )


@exam_bp.get("/<int:exam_id>")
@jwt_required()
def get_exam(exam_id: int):
    exam = Exam.query.get_or_404(exam_id)
    questions = (
        ExamQuestion.query.filter_by(exam_id=exam.exam_id)
        .order_by(ExamQuestion.question_no.asc())
        .all()
    )

    return (
        jsonify(
            {
                "exam_id": exam.exam_id,
                "subject": exam.subject,
                "max_marks": exam.max_marks,
                "rubric_details": exam.rubric_details,
                "questions": [
                    {
                        "id": q.id,
                        "question_no": q.question_no,
                        "question_text": q.question_text,
                        "answer_text": q.answer_text,
                        "marks": q.marks,
                    }
                    for q in questions
                ],
            }
        ),
        HTTPStatus.OK,
    )


@exam_bp.delete("/<int:exam_id>")
@jwt_required()
@role_required({UserRole.ADMIN, UserRole.TEACHER})
def delete_exam(exam_id: int):
    exam = Exam.query.get_or_404(exam_id)
    _delete_exam_with_children(exam)
    db.session.commit()
    return ("", HTTPStatus.NO_CONTENT)
