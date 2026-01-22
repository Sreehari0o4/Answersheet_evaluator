from functools import wraps
from http import HTTPStatus
from typing import Iterable

from flask import jsonify
from flask_jwt_extended import get_jwt

from .models import UserRole


def role_required(allowed_roles: Iterable[UserRole]):
    allowed_values = {r.value if isinstance(r, UserRole) else str(r) for r in allowed_roles}

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            claims = get_jwt()
            role = claims.get("role")
            if role not in allowed_values:
                return (
                    jsonify({"message": "Forbidden: insufficient permissions."}),
                    HTTPStatus.FORBIDDEN,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator
