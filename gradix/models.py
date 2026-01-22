from enum import Enum
from datetime import datetime

from .extensions import db


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    TEACHER = "TEACHER"
    STUDENT = "STUDENT"


class User(db.Model):
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum(UserRole), nullable=False, default=UserRole.STUDENT)


class Student(db.Model):
    __tablename__ = "students"

    student_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    roll_no = db.Column(db.String(100), unique=True, nullable=False, index=True)
    course = db.Column(db.String(255), nullable=False)
    semester = db.Column(db.String(50), nullable=False)


class Exam(db.Model):
    __tablename__ = "exams"

    exam_id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(255), nullable=False)
    max_marks = db.Column(db.Integer, nullable=False)
    rubric_details = db.Column(db.Text, nullable=True)


class AnswerSheetStatus(str, Enum):
    PENDING = "Pending"
    GRADED = "Graded"
    REVIEWED = "Reviewed"


class AnswerSheet(db.Model):
    __tablename__ = "answer_sheets"

    sheet_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(
        db.Integer, db.ForeignKey("students.student_id"), nullable=False
    )
    exam_id = db.Column(db.Integer, db.ForeignKey("exams.exam_id"), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    file_path = db.Column(db.String(500), nullable=False)
    status = db.Column(
        db.Enum(AnswerSheetStatus), nullable=False, default=AnswerSheetStatus.PENDING
    )

    student = db.relationship("Student", backref="answer_sheets")
    exam = db.relationship("Exam", backref="answer_sheets")


class ExtractedText(db.Model):
    __tablename__ = "extracted_text"

    text_id = db.Column(db.Integer, primary_key=True)
    sheet_id = db.Column(
        db.Integer, db.ForeignKey("answer_sheets.sheet_id"), nullable=False, unique=True
    )
    raw_text = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text, nullable=False)
    extraction_confidence = db.Column(db.Float, nullable=False)

    sheet = db.relationship(
        "AnswerSheet", backref=db.backref("extracted_text", uselist=False)
    )


class Evaluation(db.Model):
    __tablename__ = "evaluations"

    eval_id = db.Column(db.Integer, primary_key=True)
    text_id = db.Column(
        db.Integer, db.ForeignKey("extracted_text.text_id"), nullable=False, unique=True
    )
    model_answer_ref = db.Column(db.Text, nullable=False)
    score = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=True)
    evaluated_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    extracted_text = db.relationship(
        "ExtractedText",
        backref=db.backref("evaluation", uselist=False),
    )


class Report(db.Model):
    __tablename__ = "reports"

    report_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(
        db.Integer, db.ForeignKey("students.student_id"), nullable=False
    )
    exam_id = db.Column(db.Integer, db.ForeignKey("exams.exam_id"), nullable=False)
    total_score = db.Column(db.Float, nullable=False)
    remarks = db.Column(db.Text, nullable=True)
    generated_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    student = db.relationship("Student", backref="reports")
    exam = db.relationship("Exam", backref="reports")
