from http import HTTPStatus

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import Exam, UserRole
from ..rbac import role_required


exam_bp = Blueprint("exam", __name__, url_prefix="/exam")


@exam_bp.post("/create")
@jwt_required()
@role_required({UserRole.ADMIN, UserRole.TEACHER})
def create_exam():
    data = request.get_json(silent=True) or {}
    subject = data.get("subject")
    max_marks = data.get("max_marks")
    rubric_details = data.get("rubric_details")

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

    exam = Exam(subject=subject, max_marks=max_marks_int, rubric_details=rubric_details)
    db.session.add(exam)
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
