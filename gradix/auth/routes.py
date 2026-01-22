from http import HTTPStatus

from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token
import bcrypt

from ..extensions import db
from ..models import User, UserRole


auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return (
            jsonify({"message": "Email and password are required."}),
            HTTPStatus.BAD_REQUEST,
        )

    user = User.query.filter_by(email=email).first()
    if user is None:
        return (
            jsonify({"message": "Invalid credentials."}),
            HTTPStatus.UNAUTHORIZED,
        )

    if not bcrypt.checkpw(
        password.encode("utf-8"), user.password_hash.encode("utf-8")
    ):
        return (
            jsonify({"message": "Invalid credentials."}),
            HTTPStatus.UNAUTHORIZED,
        )

    additional_claims = {"role": user.role.value}
    # Identity ("sub" claim) must be a string for newer Flask-JWT-Extended versions
    access_token = create_access_token(identity=str(user.user_id), additional_claims=additional_claims)

    return (
        jsonify(
            {
                "access_token": access_token,
                "user": {
                    "user_id": user.user_id,
                    "name": user.name,
                    "email": user.email,
                    "role": user.role.value,
                },
            }
        ),
        HTTPStatus.OK,
    )
