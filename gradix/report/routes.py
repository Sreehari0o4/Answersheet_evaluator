from http import HTTPStatus
from datetime import datetime

from flask import Blueprint, jsonify
from flask_jwt_extended import get_jwt, jwt_required

from ..extensions import db
from ..models import AnswerSheet, AnswerSheetStatus, Evaluation, Report, UserRole


report_bp = Blueprint("report", __name__, url_prefix="/report")


@report_bp.get("/<int:student_id>/<int:exam_id>")
@jwt_required()
def get_report(student_id: int, exam_id: int):
    claims = get_jwt()
    role = claims.get("role")

    # Access control: Teacher/Admin can see all; Student can see only own (if linked)
    if role == UserRole.STUDENT.value:
        claim_student_id = claims.get("student_id")
        try:
            claim_student_id_int = int(claim_student_id)
        except (TypeError, ValueError):
            return (
                jsonify({"message": "Forbidden: invalid student identity in token."}),
                HTTPStatus.FORBIDDEN,
            )
        if claim_student_id_int != student_id:
            return (
                jsonify({"message": "Forbidden: students can only view their own reports."}),
                HTTPStatus.FORBIDDEN,
            )
    elif role not in {UserRole.TEACHER.value, UserRole.ADMIN.value}:
        return (
            jsonify({"message": "Forbidden: insufficient permissions."}),
            HTTPStatus.FORBIDDEN,
        )

    # If a report already exists, return it
    report = Report.query.filter_by(student_id=student_id, exam_id=exam_id).first()
    if report is None:
        # Generate only if there is at least one reviewed answer sheet
        sheets = (
            AnswerSheet.query.filter_by(
                student_id=student_id,
                exam_id=exam_id,
                status=AnswerSheetStatus.REVIEWED,
            )
            .order_by(AnswerSheet.sheet_id.asc())
            .all()
        )

        if not sheets:
            return (
                jsonify({"message": "Report not available. No reviewed answer sheets found."}),
                HTTPStatus.BAD_REQUEST,
            )

        # Aggregate evaluation scores across reviewed sheets
        scores = []
        remarks_parts = []
        for sheet in sheets:
            extracted = sheet.extracted_text
            if not extracted or not extracted.evaluation:
                continue
            eval_obj: Evaluation = extracted.evaluation
            scores.append(eval_obj.score)
            if eval_obj.feedback:
                remarks_parts.append(
                    f"Sheet {sheet.sheet_id}: {eval_obj.feedback}"
                )

        if not scores:
            return (
                jsonify({"message": "No evaluation scores found for reviewed sheets."}),
                HTTPStatus.BAD_REQUEST,
            )

        total_score = round(sum(scores) / len(scores), 2)
        remarks = " \n".join(remarks_parts) if remarks_parts else None

        report = Report(
            student_id=student_id,
            exam_id=exam_id,
            total_score=total_score,
            remarks=remarks,
            generated_on=datetime.utcnow(),
        )
        db.session.add(report)
        db.session.commit()

    return (
        jsonify(
            {
                "report_id": report.report_id,
                "student_id": report.student_id,
                "exam_id": report.exam_id,
                "total_score": report.total_score,
                "remarks": report.remarks,
                "generated_on": report.generated_on.isoformat(),
            }
        ),
        HTTPStatus.OK,
    )
