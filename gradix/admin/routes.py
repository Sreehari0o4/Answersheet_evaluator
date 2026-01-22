from http import HTTPStatus

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
import bcrypt

from ..extensions import db
from ..models import User, UserRole
from ..rbac import role_required


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.post("/create-teacher")
@jwt_required()
@role_required({UserRole.ADMIN})
def create_teacher():
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return (
            jsonify({"message": "Name, email and password are required."}),
            HTTPStatus.BAD_REQUEST,
        )

    existing = User.query.filter_by(email=email).first()
    if existing is not None:
        return (
            jsonify({"message": "User with this email already exists."}),
            HTTPStatus.CONFLICT,
        )

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(
        "utf-8"
    )

    teacher = User(
        name=name,
        email=email,
        password_hash=password_hash,
        role=UserRole.TEACHER,
    )
    db.session.add(teacher)
    db.session.commit()

    return (
        jsonify(
            {
                "message": "Teacher account created successfully.",
                "teacher": {
                    "user_id": teacher.user_id,
                    "name": teacher.name,
                    "email": teacher.email,
                    "role": teacher.role.value,
                },
            }
        ),
        HTTPStatus.CREATED,
    )
