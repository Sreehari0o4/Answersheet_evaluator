import os
from datetime import datetime
from functools import wraps

import bcrypt
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
    jsonify,
)

from .extensions import db
from .models import (
    AnswerSheet,
    AnswerSheetStatus,
    Evaluation,
    Exam,
    ExtractedText,
    Report,
    Student,
    User,
    UserRole,
)
from .ocr.routes import run_ocr
from .preprocess.routes import preprocess_text
from .evaluate.routes import semantic_score
from .answersheet.routes import _allowed_file


web_bp = Blueprint("web", __name__)


def login_required_view(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("web.login_page"))
        return fn(*args, **kwargs)

    return wrapper


def role_required_view(allowed_roles):
    role_values = {
        r.value if isinstance(r, UserRole) else str(r) for r in allowed_roles
    }

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = session.get("user")
            if not user or user.get("role") not in role_values:
                flash("You do not have permission to access this page.", "error")
                return redirect(url_for("web.login_page"))
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@web_bp.route("/")
def index():
    if "user" in session:
        role = session["user"].get("role")
        if role == UserRole.TEACHER.value:
            return redirect(url_for("web.teacher_dashboard"))
        if role == UserRole.STUDENT.value:
            return redirect(url_for("web.student_report"))
        if role == UserRole.ADMIN.value:
            return redirect(url_for("web.admin_home"))
    return redirect(url_for("web.login_page"))


@web_bp.route("/login", methods=["GET", "POST"])
def login_page():
    # If already logged in, redirect based on role instead of re-showing login
    if request.method == "GET" and "user" in session:
        role = session["user"].get("role")
        if role == UserRole.TEACHER.value:
            return redirect(url_for("web.teacher_dashboard"))
        if role == UserRole.STUDENT.value:
            return redirect(url_for("web.student_report"))
        if role == UserRole.ADMIN.value:
            return redirect(url_for("web.admin_home"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if user is None:
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        if not bcrypt.checkpw(
            password.encode("utf-8"), user.password_hash.encode("utf-8")
        ):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session.clear()
        session["user"] = {
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "role": user.role.value,
        }

        if user.role == UserRole.TEACHER:
            return redirect(url_for("web.teacher_dashboard"))
        if user.role == UserRole.STUDENT:
            return redirect(url_for("web.student_report"))
        return redirect(url_for("web.admin_home"))

    return render_template("login.html")


@web_bp.route("/student/login", methods=["POST"])
def student_login():
    """Simple student login using name + roll number.

    Phase I convenience: no password, matches directly against the
    Student table and sets a session user with role STUDENT.
    """

    name = (request.form.get("student_name") or "").strip()
    roll_no = (request.form.get("roll_no") or "").strip()

    if not name or not roll_no:
        flash("Name and roll number are required.", "error")
        return redirect(url_for("web.login_page"))

    student = Student.query.filter_by(name=name, roll_no=roll_no).first()
    if student is None:
        flash("No matching student found for given name and roll number.", "error")
        return redirect(url_for("web.login_page"))

    session.clear()
    session["user"] = {
        "user_id": None,
        "student_id": student.student_id,
        "name": student.name,
        "email": None,
        "role": UserRole.STUDENT.value,
    }

    return redirect(url_for("web.student_report"))


@web_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("web.login_page"))


@web_bp.route("/admin")
@login_required_view
@role_required_view({UserRole.ADMIN})
def admin_home():
    return "Admin area (minimal frontend placeholder)."


@web_bp.route("/teacher/dashboard")
@login_required_view
@role_required_view({UserRole.TEACHER})
def teacher_dashboard():
    teacher = session.get("user")
    exams = Exam.query.order_by(Exam.exam_id.asc()).all()
    sheets = AnswerSheet.query.order_by(AnswerSheet.upload_date.desc()).all()
    return render_template(
        "teacher_dashboard.html",
        teacher=teacher,
        exams=exams,
        sheets=sheets,
        AnswerSheetStatus=AnswerSheetStatus,
    )


@web_bp.route("/teacher/exam/create", methods=["GET", "POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def create_exam_page():
    if request.method == "POST":
        subject = (request.form.get("subject") or "").strip()
        max_marks = request.form.get("max_marks")
        rubric_details = request.form.get("rubric_details") or None

        errors = []
        if not subject:
            errors.append("Subject is required.")

        try:
            max_marks_int = int(max_marks)
            if max_marks_int <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Max marks must be a positive integer.")
            max_marks_int = None

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("create_exam.html")

        exam = Exam(subject=subject, max_marks=max_marks_int, rubric_details=rubric_details)
        db.session.add(exam)
        db.session.commit()

        flash("Exam created successfully.", "success")
        return redirect(url_for("web.teacher_dashboard"))

    return render_template("create_exam.html")


@web_bp.route("/teacher/students", methods=["GET", "POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def manage_students():
    """List students and allow teachers/admins to add new ones."""

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        roll_no = (request.form.get("roll_no") or "").strip()
        course = (request.form.get("course") or "").strip()
        semester = (request.form.get("semester") or "").strip()

        errors = []
        if not name:
            errors.append("Name is required.")
        if not roll_no:
            errors.append("Roll number is required.")
        if not course:
            errors.append("Course is required.")
        if not semester:
            errors.append("Semester is required.")

        if Student.query.filter_by(roll_no=roll_no).first() is not None:
            errors.append("A student with this roll number already exists.")

        if errors:
            for e in errors:
                flash(e, "error")
        else:
            student = Student(
                name=name,
                roll_no=roll_no,
                course=course,
                semester=semester,
            )
            db.session.add(student)
            db.session.commit()
            flash("Student added successfully.", "success")

    students = Student.query.order_by(Student.student_id.asc()).all()
    return render_template("students_manage.html", students=students)


@web_bp.route("/teacher/evaluated-students")
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def evaluated_students():
    """Exam-wise evaluated students list with current marks.

    - Shows a row of buttons for each exam that has graded/reviewed
      answer sheets.
    - When an exam is selected (via ?exam_id=...), lists students for
      that exam with their latest mark and links to review/report.
    """

    # Exams that have at least one graded or reviewed sheet
    exams_with_sheets = (
        db.session.query(Exam)
        .join(AnswerSheet, AnswerSheet.exam_id == Exam.exam_id)
        .filter(AnswerSheet.status.in_([AnswerSheetStatus.GRADED, AnswerSheetStatus.REVIEWED]))
        .distinct()
        .order_by(Exam.exam_id.asc())
        .all()
    )

    selected_exam_id_raw = request.args.get("exam_id")
    selected_exam = None
    student_rows = []

    if selected_exam_id_raw:
        try:
            selected_exam_id = int(selected_exam_id_raw)
        except (TypeError, ValueError):
            selected_exam_id = None

        if selected_exam_id is not None:
            selected_exam = Exam.query.get(selected_exam_id)

            if selected_exam is not None:
                from .models import ExtractedText  # local import to avoid cycles

                rows = (
                    db.session.query(Student, AnswerSheet, Evaluation)
                    .join(AnswerSheet, AnswerSheet.student_id == Student.student_id)
                    .join(ExtractedText, ExtractedText.sheet_id == AnswerSheet.sheet_id)
                    .join(Evaluation, Evaluation.text_id == ExtractedText.text_id)
                    .filter(
                        AnswerSheet.exam_id == selected_exam_id,
                        AnswerSheet.status.in_([AnswerSheetStatus.GRADED, AnswerSheetStatus.REVIEWED]),
                    )
                    .order_by(Student.name.asc(), AnswerSheet.upload_date.desc())
                    .all()
                )

                latest_by_student = {}
                for student, sheet, evaluation in rows:
                    # First row per student is the latest due to ordering
                    if student.student_id not in latest_by_student:
                        latest_by_student[student.student_id] = (student, sheet, evaluation)

                for student_id, (student, sheet, evaluation) in latest_by_student.items():
                    student_rows.append(
                        {
                            "student": student,
                            "sheet": sheet,
                            "evaluation": evaluation,
                        }
                    )

                # Consistent ordering by student name
                student_rows.sort(key=lambda r: r["student"].name.lower())

    return render_template(
        "evaluated_students.html",
        exams=exams_with_sheets,
        selected_exam=selected_exam,
        student_rows=student_rows,
    )


@web_bp.route("/upload", methods=["GET", "POST"])
@login_required_view
@role_required_view({UserRole.TEACHER})
def upload_page():
    if request.method == "POST":
        student_name = (request.form.get("student_name") or "").strip()
        roll_no = (request.form.get("roll_no") or "").strip()
        exam_id = request.form.get("exam_id")
        file = request.files.get("file")

        errors = []
        if not student_name:
            errors.append("Student name is required.")
        if not roll_no:
            errors.append("Roll number is required.")
        if not exam_id or file is None:
            errors.append("Exam and file are required.")

        exam = None
        try:
            exam_id_int = int(exam_id) if exam_id is not None else None
            exam = Exam.query.get(exam_id_int) if exam_id_int is not None else None
        except (TypeError, ValueError):
            errors.append("Exam must be valid.")
            exam_id_int = None

        if exam is None:
            errors.append("Selected exam not found.")

        filename = file.filename or ""
        if not filename or not _allowed_file(filename):
            errors.append("Invalid file type. Allowed: PDF, JPG, PNG.")

        if errors:
            for e in errors:
                flash(e, "error")
            exams = Exam.query.order_by(Exam.exam_id.asc()).all()
            return render_template("upload.html", exams=exams)

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)

        ext = filename.rsplit(".", 1)[1].lower()
        from uuid import uuid4

        # Find or create the student using name and roll number only.
        student = Student.query.filter_by(roll_no=roll_no).first()
        if student is None:
            # Course and semester are required in the model, so we store
            # placeholder values since the guide requires only name + roll no.
            student = Student(
                name=student_name,
                roll_no=roll_no,
                course="UNKNOWN",
                semester="UNKNOWN",
            )
            db.session.add(student)
            db.session.flush()

        student_id_int = student.student_id

        safe_name = f"{student_id_int}_{exam_id_int}_{uuid4().hex}.{ext}"
        full_path = os.path.join(upload_folder, safe_name)
        file.save(full_path)

        relative_path = os.path.join("uploads", safe_name)

        sheet = AnswerSheet(
            student_id=student_id_int,
            exam_id=exam_id_int,
            file_path=relative_path,
            status=AnswerSheetStatus.PENDING,
        )
        db.session.add(sheet)
        db.session.flush()  # get sheet_id

        # Run mocked OCR and preprocessing, then evaluation
        raw_text, confidence = run_ocr(full_path)
        cleaned_text = preprocess_text(raw_text)

        extracted = ExtractedText(
            sheet_id=sheet.sheet_id,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extraction_confidence=confidence,
        )
        db.session.add(extracted)
        db.session.flush()

        model_answer = exam.rubric_details or "Reference answer not provided."
        semantic = semantic_score(cleaned_text, model_answer)
        keyword_score = 0.80
        grammar_score = 0.90
        final_score = round((semantic + keyword_score + grammar_score) / 3.0, 2)
        feedback = (
            f"Semantic: {semantic:.2f}, Keyword: {keyword_score:.2f}, "
            f"Grammar: {grammar_score:.2f}. Final score: {final_score:.2f}."
        )

        evaluation = Evaluation(
            text_id=extracted.text_id,
            model_answer_ref=model_answer,
            score=final_score,
            feedback=feedback,
            evaluated_on=datetime.utcnow(),
        )
        db.session.add(evaluation)

        sheet.status = AnswerSheetStatus.GRADED
        db.session.commit()

        flash("Answer sheet uploaded and auto-evaluated (mock).", "success")
        return redirect(url_for("web.teacher_dashboard"))

    exams = Exam.query.order_by(Exam.exam_id.asc()).all()
    return render_template("upload.html", exams=exams)


@web_bp.route("/review/<int:sheet_id>", methods=["GET", "POST"])
@login_required_view
@role_required_view({UserRole.TEACHER})
def review_page(sheet_id: int):
    sheet = AnswerSheet.query.get_or_404(sheet_id)

    if sheet.status not in {AnswerSheetStatus.GRADED, AnswerSheetStatus.REVIEWED}:
        flash("Only graded or reviewed answer sheets can be reviewed.", "error")
        return redirect(url_for("web.teacher_dashboard"))

    extracted = sheet.extracted_text
    evaluation = extracted.evaluation if extracted else None

    if evaluation is None:
        flash("No evaluation found for this answer sheet.", "error")
        return redirect(url_for("web.teacher_dashboard"))

    if request.method == "POST":
        if sheet.status != AnswerSheetStatus.GRADED:
            flash("Only graded answer sheets can be overridden.", "error")
            return redirect(url_for("web.review_page", sheet_id=sheet_id))

        score = request.form.get("score")
        feedback = request.form.get("feedback")

        try:
            score_val = float(score)
        except (TypeError, ValueError):
            flash("Score must be a number.", "error")
            return render_template(
                "review.html",
                sheet=sheet,
                extracted=extracted,
                evaluation=evaluation,
                file_url=url_for(
                    "web.uploaded_file", filename=os.path.basename(sheet.file_path)
                ),
            )

        evaluation.score = score_val
        if feedback:
            evaluation.feedback = feedback
        evaluation.evaluated_on = datetime.utcnow()
        sheet.status = AnswerSheetStatus.REVIEWED
        db.session.commit()

        flash("Score overridden and sheet marked as Reviewed.", "success")
        return redirect(url_for("web.teacher_dashboard"))

    file_url = url_for("web.uploaded_file", filename=os.path.basename(sheet.file_path))
    return render_template(
        "review.html",
        sheet=sheet,
        extracted=extracted,
        evaluation=evaluation,
        file_url=file_url,
    )


@web_bp.route("/uploads/<path:filename>")
@login_required_view
def uploaded_file(filename: str):
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    return send_from_directory(upload_folder, filename)


@web_bp.route("/student/report")
@login_required_view
@role_required_view({UserRole.STUDENT})
def student_report():
    user = session.get("user")
    student_id = user.get("student_id")
    student = None
    if student_id is not None:
        try:
            student = Student.query.get(int(student_id))
        except (TypeError, ValueError):
            student = None

    # Fallback for older sessions: match by name if no student_id stored
    if student is None:
        student = Student.query.filter_by(name=user.get("name")).first()
    if student is None:
        flash(
            "No student record linked to this user. Ask admin/teacher to create one.",
            "error",
        )
        return render_template("student_report.html", student=None, reports=[])

    reports = Report.query.filter_by(student_id=student.student_id).all()
    return render_template("student_report.html", student=student, reports=reports)


@web_bp.route("/teacher/report/<int:student_id>/<int:exam_id>")
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def teacher_report(student_id: int, exam_id: int):
    """Return or generate a report for a given student/exam for teacher view.

    This mirrors the logic of the JWT-protected /report API but uses the
    server-side session instead of a JWT, so teachers can click from the
    dashboard without providing an Authorization header.
    """

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
            flash(
                "Report not available. No reviewed answer sheets found.",
                "error",
            )
            return redirect(url_for("web.teacher_dashboard"))

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
                remarks_parts.append(f"Sheet {sheet.sheet_id}: {eval_obj.feedback}")

        if not scores:
            flash("No evaluation scores found for reviewed sheets.", "error")
            return redirect(url_for("web.teacher_dashboard"))

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

    return jsonify(
        {
            "report_id": report.report_id,
            "student_id": report.student_id,
            "exam_id": report.exam_id,
            "total_score": report.total_score,
            "remarks": report.remarks,
            "generated_on": report.generated_on.isoformat(),
        }
    )
