from http import HTTPStatus

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import Student, UserRole
from ..rbac import role_required


student_bp = Blueprint("student", __name__, url_prefix="/student")


@student_bp.post("/create")
@jwt_required()
@role_required({UserRole.ADMIN, UserRole.TEACHER})
def create_student():
    data = request.get_json(silent=True) or {}

    name = data.get("name")
    roll_no = data.get("roll_no")
    course = data.get("course")
    semester = data.get("semester")

    if not all([name, roll_no, course, semester]):
        return (
            jsonify({"message": "'name', 'roll_no', 'course', and 'semester' are required."}),
            HTTPStatus.BAD_REQUEST,
        )

    existing = Student.query.filter_by(roll_no=roll_no).first()
    if existing is not None:
        return (
            jsonify({"message": "Student with this roll number already exists."}),
            HTTPStatus.CONFLICT,
        )

    student = Student(
        name=name,
        roll_no=roll_no,
        course=course,
        semester=semester,
    )
    db.session.add(student)
    db.session.commit()

    return (
        jsonify(
            {
                "student_id": student.student_id,
                "name": student.name,
                "roll_no": student.roll_no,
                "course": student.course,
                "semester": student.semester,
            }
        ),
        HTTPStatus.CREATED,
    )


@student_bp.get("/list")
@jwt_required()
@role_required({UserRole.ADMIN, UserRole.TEACHER})
def list_students():
    students = Student.query.order_by(Student.student_id.desc()).all()
    return (
        jsonify(
            [
                {
                    "student_id": s.student_id,
                    "name": s.name,
                    "roll_no": s.roll_no,
                    "course": s.course,
                    "semester": s.semester,
                }
                for s in students
            ]
        ),
        HTTPStatus.OK,
    )
