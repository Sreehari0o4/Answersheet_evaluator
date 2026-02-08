from flask import Flask
import os

from dotenv import load_dotenv

from .config import Config
from .extensions import db, jwt
from .auth.routes import auth_bp
from .admin.routes import admin_bp
from .exam.routes import exam_bp
from .student.routes import student_bp
from .answersheet.routes import answersheet_bp
from .ocr.routes import ocr_bp
from .preprocess.routes import preprocess_bp
from .evaluate.routes import evaluate_bp
from .review.routes import review_bp
from .report.routes import report_bp
from .analytics.routes import analytics_bp
from .web import web_bp


def create_app() -> Flask:
    # Load environment variables from a local .env file (if present).
    # This lets you keep secrets such as API keys out of the code.
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(env_path)

    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    jwt.init_app(app)

    app.register_blueprint(web_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(exam_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(answersheet_bp)
    app.register_blueprint(ocr_bp)
    app.register_blueprint(preprocess_bp)
    app.register_blueprint(evaluate_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(analytics_bp)

    with app.app_context():
        db.create_all()

    return app
