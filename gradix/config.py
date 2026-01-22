import os


basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("GRADIX_SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "GRADIX_DATABASE_URI",
        "sqlite:///gradix.db",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get("GRADIX_JWT_SECRET_KEY", "dev-jwt-secret")

    # File uploads
    UPLOAD_FOLDER = os.environ.get(
        "GRADIX_UPLOAD_FOLDER",
        os.path.abspath(os.path.join(basedir, "..", "uploads")),
    )
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
