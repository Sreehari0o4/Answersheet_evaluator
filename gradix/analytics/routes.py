from http import HTTPStatus

from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import AnswerSheet, Report, UserRole
from ..rbac import role_required


analytics_bp = Blueprint("analytics", __name__, url_prefix="/analytics")


@analytics_bp.get("/exam/<int:exam_id>")
@jwt_required()
@role_required({UserRole.TEACHER, UserRole.ADMIN})
def exam_analytics(exam_id: int):
    # Total submissions (all answer sheets for this exam)
    total_submissions = AnswerSheet.query.filter_by(exam_id=exam_id).count()

    # Use reports as the source for scores (one per student per exam)
    reports = Report.query.filter_by(exam_id=exam_id).all()
    if not reports:
        return (
            jsonify(
                {
                    "exam_id": exam_id,
                    "class_average": None,
                    "total_submissions": total_submissions,
                    "score_distribution": {},
                }
            ),
            HTTPStatus.OK,
        )

    scores = [r.total_score for r in reports]
    class_average = round(sum(scores) / len(scores), 2)

    # Simple score distribution buckets (dummy but useful)
    buckets = {
        "0-20": 0,
        "20-40": 0,
        "40-60": 0,
        "60-80": 0,
        "80-100": 0,
        ">100": 0,
    }
    for s in scores:
        if s < 20:
            buckets["0-20"] += 1
        elif s < 40:
            buckets["20-40"] += 1
        elif s < 60:
            buckets["40-60"] += 1
        elif s < 80:
            buckets["60-80"] += 1
        elif s <= 100:
            buckets["80-100"] += 1
        else:
            buckets[">100"] += 1

    return (
        jsonify(
            {
                "exam_id": exam_id,
                "class_average": class_average,
                "total_submissions": total_submissions,
                "score_distribution": buckets,
            }
        ),
        HTTPStatus.OK,
    )
