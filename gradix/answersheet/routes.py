import os
from http import HTTPStatus
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import AnswerSheet, AnswerSheetStatus, Exam, Student, UserRole
from ..rbac import role_required


answersheet_bp = Blueprint("answersheet", __name__, url_prefix="/answersheet")


ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@answersheet_bp.post("/upload")
@jwt_required()
@role_required({UserRole.TEACHER})
def upload_answersheet():
    student_id = request.form.get("student_id")
    exam_id = request.form.get("exam_id")
    file = request.files.get("file")

    if not student_id or not exam_id or file is None:
        return (
            jsonify({"message": "'student_id', 'exam_id' and file are required."}),
            HTTPStatus.BAD_REQUEST,
        )

    # Validate numeric IDs
    try:
        student_id_int = int(student_id)
        exam_id_int = int(exam_id)
    except ValueError:
        return (
            jsonify({"message": "'student_id' and 'exam_id' must be integers."}),
            HTTPStatus.BAD_REQUEST,
        )

    # Validate foreign keys
    student = Student.query.get(student_id_int)
    if student is None:
        return (
            jsonify({"message": "Invalid student_id."}),
            HTTPStatus.BAD_REQUEST,
        )

    exam = Exam.query.get(exam_id_int)
    if exam is None:
        return (
            jsonify({"message": "Invalid exam_id."}),
            HTTPStatus.BAD_REQUEST,
        )

    # Validate file type
    filename = file.filename or ""
    if filename == "" or not _allowed_file(filename):
        return (
            jsonify({"message": "Invalid file type. Allowed: PDF, JPG, PNG."}),
            HTTPStatus.BAD_REQUEST,
        )

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)

    ext = filename.rsplit(".", 1)[1].lower()
    safe_name = f"{student_id_int}_{exam_id_int}_{uuid4().hex}.{ext}"
    full_path = os.path.join(upload_folder, safe_name)

    file.save(full_path)

    # Store relative path (uploads/<filename>) for portability
    relative_path = os.path.join("uploads", safe_name)

    sheet = AnswerSheet(
        student_id=student_id_int,
        exam_id=exam_id_int,
        file_path=relative_path,
        status=AnswerSheetStatus.PENDING,
    )
    db.session.add(sheet)
    db.session.commit()

    return (
        jsonify(
            {
                "sheet_id": sheet.sheet_id,
                "student_id": sheet.student_id,
                "exam_id": sheet.exam_id,
                "upload_date": sheet.upload_date.isoformat(),
                "file_path": sheet.file_path,
                "status": sheet.status.value,
            }
        ),
        HTTPStatus.CREATED,
    )
