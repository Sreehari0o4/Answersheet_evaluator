import os
import re
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
    ExamQuestion,
    ExtractedText,
    Report,
    Student,
    User,
    UserRole,
    QuestionStudentComment,
)
from .ocr.routes import run_ocr
from .preprocess.routes import preprocess_text, split_numbered_answers
from .evaluate.routes import evaluate_text_by_questions
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

    # Pending student comments (unresolved) across all sheets
    pending_comments = (
        db.session.query(QuestionStudentComment)
        .join(AnswerSheet, QuestionStudentComment.sheet_id == AnswerSheet.sheet_id)
        .filter(QuestionStudentComment.resolved.is_(False))
        .order_by(QuestionStudentComment.created_at.desc())
        .all()
    )
    return render_template(
        "teacher_dashboard.html",
        teacher=teacher,
        exams=exams,
        sheets=sheets,
        pending_comments=pending_comments,
        AnswerSheetStatus=AnswerSheetStatus,
    )


@web_bp.route("/teacher/exam/create", methods=["GET", "POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def create_exam_page():
    if request.method == "POST":
        subject = (request.form.get("subject") or "").strip()
        max_marks = request.form.get("max_marks")
        total_questions = request.form.get("total_questions")

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

        # Validate total number of questions
        try:
            total_questions_int = int(total_questions) if total_questions is not None else 0
            if total_questions_int <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Total number of questions must be a positive integer.")
            total_questions_int = 0

        questions: list[dict] = []
        if total_questions_int > 0:
            for i in range(1, total_questions_int + 1):
                q_no_raw = request.form.get(f"question_no_{i}") or str(i)
                try:
                    q_no = int(q_no_raw)
                except (TypeError, ValueError):
                    q_no = i

                q_text = (request.form.get(f"question_text_{i}") or "").strip()
                marks_raw = request.form.get(f"marks_{i}")

                try:
                    marks_val = float(marks_raw) if marks_raw is not None else None
                    if marks_val is not None and marks_val <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"Question {i}: marks must be a positive number.")
                    marks_val = None

                if not q_text:
                    errors.append(
                        f"Question {i}: question text is required."
                    )
                    continue

                questions.append(
                    {
                        "question_no": q_no,
                        "question_text": q_text,
                        "marks": marks_val,
                    }
                )

        if total_questions_int > 0 and not questions:
            errors.append("At least one question is required.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("create_exam.html")

        exam = Exam(subject=subject, max_marks=max_marks_int)
        db.session.add(exam)
        db.session.flush()

        rubric_parts: list[str] = []
        for item in questions:
            eq = ExamQuestion(
                exam_id=exam.exam_id,
                question_no=item["question_no"],
                question_text=item["question_text"],
                # Model answers are now optional; store an empty
                # string instead of NULL so existing databases where
                # answer_text is NOT NULL continue to work.
                answer_text="",
                marks=item["marks"],
            )
            db.session.add(eq)
            rubric_parts.append(
                (
                    f"Q{item['question_no']}"
                    + (f" ({item['marks']} marks)" if item["marks"] is not None else "")
                    + f". {item['question_text']}"
                )
            )

        if rubric_parts:
            exam.rubric_details = "\n\n".join(rubric_parts)

        db.session.commit()

        flash("Exam created successfully.", "success")
        return redirect(url_for("web.teacher_dashboard"))

    return render_template("create_exam.html")


@web_bp.route("/teacher/exam/extract-questions", methods=["POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def extract_questions_from_paper():
    """Extract numbered questions and marks from an uploaded exam paper image.

    This is used by the Create Exam page's "Update Questions" button.
    The route returns JSON of the form::

        {"questions": [{"question_no": int, "question_text": str, "marks": float|null}, ...]}
    """

    file = request.files.get("question_paper")
    if file is None or not file.filename:
        return jsonify({"message": "No file uploaded."}), 400

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)

    # Store temporarily under the same uploads root; teachers are not
    # expected to access this file directly.
    from uuid import uuid4

    ext = (file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else "png")
    safe_name = f"question_paper_{uuid4().hex}.{ext}"
    full_path = os.path.join(upload_folder, safe_name)
    file.save(full_path)

    try:
        raw_text = ""
        confidence = 0.0

        use_gemini = os.environ.get("USE_GEMINI", "").lower() in {"1", "true", "yes"}
        if use_gemini:
            # Import Gemini helpers outside the inner try/except so
            # the exception class is defined when referenced.
            try:  # pragma: no cover - external API
                from gemini_ocr_client import (
                    GeminiConfigError,
                    extract_text as gemini_extract,
                )
            except Exception as exc:  # noqa: BLE001
                current_app.logger.warning(
                    "Gemini import failed for question paper OCR, falling back to local OCR: %s",
                    exc,
                )
            else:
                try:  # pragma: no cover - external API
                    current_app.logger.info(
                        "Attempting Gemini OCR for question paper: %s", full_path
                    )
                    g_text = gemini_extract(full_path)
                    if g_text:
                        raw_text = g_text
                        confidence = 0.98
                    else:
                        raise RuntimeError("Empty text from Gemini OCR")
                except GeminiConfigError as exc:
                    current_app.logger.warning(
                        "Gemini misconfigured for question paper OCR, falling back to local OCR: %s",
                        exc,
                    )
                except Exception as exc:  # noqa: BLE001
                    current_app.logger.exception(
                        "Gemini OCR failed for question paper, falling back to local OCR: %s",
                        exc,
                    )

        if not raw_text:
            raw_text, confidence = run_ocr(full_path)

        # Re-use the same splitting logic we use for numbered answers,
        # which expects patterns like "1.", "2)", etc.
        segments = split_numbered_answers(raw_text)

        import re

        questions_out: list[dict] = []
        for q_no, q_text in segments:
            original_text = (q_text or "").strip()
            if not original_text:
                continue

            text = original_text

            # Try to extract marks from common patterns such as
            # "(5 marks)", plain "(3)" at the end of the question,
            # "[5]", "5M", etc., taking the first plausible
            # numeric value.
            marks_val = None
            patterns = [
                r"\((\d+(?:\.\d+)?)\s*marks?\)",
                r"\((\d+(?:\.\d+)?)\)\s*$",
                r"\[(\d+(?:\.\d+)?)\]",
                r"(\d+(?:\.\d+)?)\s*[Mm]arks?",
                r"(\d+(?:\.\d+)?)\s*[Mm]",
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    try:
                        marks_val = float(m.group(1))
                        # Strip the matched mark fragment (e.g. "(3)"
                        # or "(3 marks)" or "3 marks") from the
                        # question text so the stored question is
                        # clean.
                        text = text[: m.start()].rstrip()
                        break
                    except (TypeError, ValueError):
                        marks_val = None

            # Extra heuristic for layouts where only a bare number
            # (e.g. "3") appears in the right-hand "Marks" column,
            # with no explicit "marks" text. We look for a trailing
            # small integer at the end of the last non-empty line.
            if marks_val is None:
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if lines:
                    last_line = lines[-1]
                    m2 = re.search(r"(\d{1,3})\s*$", last_line)
                    if m2:
                        try:
                            candidate = int(m2.group(1))
                        except (TypeError, ValueError):
                            candidate = None
                        # Treat modest integers as marks, not part of
                        # the question text (e.g. 1-50).
                        if candidate is not None and 0 < candidate <= 50:
                            marks_val = float(candidate)
                            # Remove the trailing number from the
                            # question text so the UI shows a clean
                            # question.
                            text = re.sub(r"\b" + re.escape(m2.group(1)) + r"\s*$", "", text).rstrip()

            questions_out.append(
                {
                    "question_no": int(q_no),
                    "question_text": text,
                    "marks": marks_val,
                }
            )

        return jsonify({"questions": questions_out, "ocr_confidence": confidence})
    finally:
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
        except OSError:
            current_app.logger.info("Could not remove temporary question paper file: %s", full_path)


@web_bp.route("/teacher/exam/<int:exam_id>")
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def view_exam(exam_id: int):
    exam = Exam.query.get_or_404(exam_id)
    questions = (
        ExamQuestion.query.filter_by(exam_id=exam.exam_id)
        .order_by(ExamQuestion.question_no.asc())
        .all()
    )
    return render_template("exam_detail.html", exam=exam, questions=questions)


@web_bp.route("/teacher/exam/<int:exam_id>/delete", methods=["POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def delete_exam_view(exam_id: int):
    from .exam.routes import _delete_exam_with_children

    exam = Exam.query.get_or_404(exam_id)

    # Remove exam and all related data (questions, sheets, reports)
    _delete_exam_with_children(exam)
    db.session.commit()

    flash("Exam and related data deleted successfully.", "success")
    return redirect(url_for("web.teacher_dashboard"))


@web_bp.route("/teacher/students", methods=["GET", "POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def manage_students():
    """List students and allow teachers/admins to add new ones.

    The page also supports simple filtering by course and semester so
    teachers can quickly find students by department/section.
    """

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

        return redirect(url_for("web.manage_students"))

    # Build filter options
    course_filter = (request.args.get("course") or "").strip()
    semester_filter = (request.args.get("semester") or "").strip()

    query = Student.query
    if course_filter:
        query = query.filter(Student.course == course_filter)
    if semester_filter:
        query = query.filter(Student.semester == semester_filter)

    students = query.order_by(Student.student_id.asc()).all()

    courses = [row[0] for row in db.session.query(Student.course).distinct().order_by(Student.course.asc()).all()]
    semesters = [row[0] for row in db.session.query(Student.semester).distinct().order_by(Student.semester.asc()).all()]

    return render_template(
        "students_manage.html",
        students=students,
        courses=courses,
        semesters=semesters,
        selected_course=course_filter,
        selected_semester=semester_filter,
    )


@web_bp.route("/teacher/students/<int:student_id>/delete", methods=["POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def delete_student(student_id: int):
    """Remove a student and their related data (answer sheets, reports)."""

    student = Student.query.get_or_404(student_id)

    # Delete answer sheets and their nested evaluation data
    for sheet in list(student.answer_sheets):
        extracted = sheet.extracted_text
        if extracted is not None:
            evaluation = extracted.evaluation
            if evaluation is not None:
                db.session.delete(evaluation)
            db.session.delete(extracted)

        db.session.delete(sheet)

    # Delete reports linked to this student
    Report.query.filter_by(student_id=student.student_id).delete(synchronize_session=False)

    db.session.delete(student)
    db.session.commit()

    flash("Student and related data removed.", "success")
    return redirect(url_for("web.manage_students"))


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
        action = request.form.get("action") or "extract"

        # If the user clicked Evaluate after extracting text, we will
        # receive an existing sheet_id and only need to run evaluation.
        if action == "evaluate" and request.form.get("sheet_id"):
            try:
                sheet_id_int = int(request.form.get("sheet_id"))
            except (TypeError, ValueError):
                flash("Invalid sheet id for evaluation.", "error")
                exams = Exam.query.order_by(Exam.exam_id.asc()).all()
                return render_template("upload.html", exams=exams)

            sheet = AnswerSheet.query.get(sheet_id_int)
            if sheet is None or sheet.extracted_text is None:
                flash("No extracted text found for this sheet. Please extract again.", "error")
                exams = Exam.query.order_by(Exam.exam_id.asc()).all()
                return render_template("upload.html", exams=exams)

            exam = sheet.exam
            extracted = sheet.extracted_text

            # Use numbered segments from the raw OCR text so that
            # question numbers aren't lost by grammar correction.
            segments = split_numbered_answers(extracted.raw_text)

            # If the exam has structured questions (with marks), prefer a
            # Gemini-based evaluation using the full rubric.
            exam_questions = sorted(
                getattr(exam, "questions", []),
                key=lambda q: q.question_no,
            )

            # Build a textual representation of the rubric for storage in
            # Evaluation.model_answer_ref. Prefer any precomputed
            # rubric text; otherwise, store a simple listing of
            # questions (without requiring model answers).
            if exam.rubric_details:
                model_answer_text = exam.rubric_details
            elif exam_questions:
                parts = []
                for q in exam_questions:
                    base = f"Q{q.question_no}. {q.question_text}"
                    parts.append(base)
                model_answer_text = "\n\n".join(parts)
            else:
                model_answer_text = "Reference answer not provided."

            final_score: float
            feedback: str
            per_q: list[dict]

            if exam_questions:
                answers_by_q = {q_no: ans for q_no, ans in segments}

                try:  # pragma: no cover - depends on external API
                    from gemini_ocr_client import (
                        GeminiConfigError,
                        evaluate_answers_with_gemini,
                    )

                    payload_items = []
                    marks_by_q: dict[int, float | None] = {}
                    for eq in exam_questions:
                        max_marks_val = float(eq.marks) if eq.marks is not None else None
                        marks_by_q[eq.question_no] = max_marks_val
                        payload_items.append(
                            {
                                "question_no": eq.question_no,
                                "question_text": eq.question_text,
                                # Model answers may be None/empty; the
                                # Gemini helper will fall back to
                                # grading using only the question text
                                # and max marks in that case.
                                "model_answer": eq.answer_text,
                                "max_marks": max_marks_val,
                                "student_answer": answers_by_q.get(eq.question_no, ""),
                            }
                        )

                    gemini_result = evaluate_answers_with_gemini(payload_items)
                    questions_out = gemini_result.get("questions", []) or []

                    per_q = []
                    # Align Gemini output with payload order so each
                    # exam question gets a score, even if Gemini's
                    # question_no field is inconsistent or missing.
                    for idx, payload in enumerate(payload_items):
                        q_no = payload["question_no"]
                        out_item = questions_out[idx] if idx < len(questions_out) else {}
                        try:
                            raw_score = float(out_item.get("score", 0.0))
                        except (TypeError, ValueError):
                            raw_score = 0.0

                        max_marks = marks_by_q.get(q_no)
                        # If Gemini returns a 0-1 style score while we
                        # have a higher max mark, scale it up so that a
                        # score like 0.7 with max_marks=5 becomes 3.5.
                        if (
                            max_marks is not None
                            and max_marks > 1.0
                            and 0.0 <= raw_score <= 1.0
                        ):
                            q_score = raw_score * max_marks
                        else:
                            q_score = raw_score

                        ans_text = payload["student_answer"] or ""
                        # Do not store Gemini-generated comments; feedback
                        # will be provided only by teachers during review.
                        per_q.append(
                            {
                                "question_no": q_no,
                                "answer_text": ans_text,
                                "score": q_score,
                                "feedback": "",
                            }
                        )

                    # Always compute total as the sum of per-question
                    # scores after any scaling.
                    total_score = sum(p["score"] for p in per_q) if per_q else 0.0

                    final_score = round(float(total_score), 2)
                    # Keep overall feedback minimal; detailed comments are
                    # added later by the teacher.
                    feedback = "Auto-evaluated using Gemini based on exam rubric."
                except GeminiConfigError as exc:
                    current_app.logger.warning(
                        "Gemini evaluation misconfigured, falling back to heuristic scoring: %s",
                        exc,
                    )
                    # Fallback: simple heuristic based on cleaned text
                    model_answer_text = exam.rubric_details or "Reference answer not provided."
                    final_score, feedback, per_q = evaluate_text_by_questions(
                        extracted.cleaned_text,
                        model_answer_text,
                    )
                except Exception as exc:  # noqa: BLE001
                    current_app.logger.exception(
                        "Gemini evaluation failed, falling back to heuristic scoring: %s",
                        exc,
                    )
                    model_answer_text = exam.rubric_details or "Reference answer not provided."
                    final_score, feedback, per_q = evaluate_text_by_questions(
                        extracted.cleaned_text,
                        model_answer_text,
                    )
            else:
                # No structured exam questions; fall back to simple heuristic.
                model_answer_text = exam.rubric_details or "Reference answer not provided."
                final_score, feedback, per_q = evaluate_text_by_questions(
                    extracted.cleaned_text,
                    model_answer_text,
                )

            # Create or update evaluation
            evaluation = extracted.evaluation
            if evaluation is None:
                evaluation = Evaluation(
                    text_id=extracted.text_id,
                    model_answer_ref=model_answer_text,
                    score=final_score,
                    feedback=feedback,
                    evaluated_on=datetime.utcnow(),
                )
                db.session.add(evaluation)
                db.session.flush()
            else:
                evaluation.model_answer_ref = model_answer_text
                evaluation.score = final_score
                evaluation.feedback = feedback
                evaluation.evaluated_on = datetime.utcnow()

                # Clear existing question scores
                for qs in list(evaluation.question_scores):
                    db.session.delete(qs)

            # Store per-question evaluations, aligned to the exam's questions.
            from .models import QuestionEvaluation

            exam_questions = sorted(
                getattr(exam, "questions", []),
                key=lambda q: q.question_no,
            )

            if exam_questions:
                # Map OCR-numbered answers by question number
                per_q_by_no: dict[int, dict] = {}
                for item in per_q:
                    try:
                        q_no_int = int(item["question_no"])
                    except (TypeError, ValueError):
                        continue
                    per_q_by_no[q_no_int] = item

                for eq in exam_questions:
                    item = per_q_by_no.get(eq.question_no)
                    if item is not None:
                        try:
                            q_score = float(item["score"])
                        except (TypeError, ValueError):
                            q_score = 0.0
                        feedback_text = (item.get("feedback") or "").strip() or None
                    else:
                        # No matching numbered answer in extracted text => unanswered
                        q_score = 0.0
                        feedback_text = "Unanswered"

                    qe = QuestionEvaluation(
                        eval_id=evaluation.eval_id,
                        question_no=eq.question_no,
                        score=q_score,
                        feedback=feedback_text,
                    )
                    db.session.add(qe)
            else:
                # Legacy: if exam has no defined questions, fall back to OCR numbers
                for item in per_q:
                    try:
                        q_no = int(item["question_no"])
                        q_score = float(item["score"])
                    except (TypeError, ValueError):
                        continue
                    qe = QuestionEvaluation(
                        eval_id=evaluation.eval_id,
                        question_no=q_no,
                        score=q_score,
                    )
                    db.session.add(qe)

            sheet.status = AnswerSheetStatus.GRADED
            db.session.commit()

            flash("Answer sheet evaluated.", "success")
            return redirect(url_for("web.review_page", sheet_id=sheet.sheet_id))

        # Otherwise we are in the Extract Text step: upload + OCR only.
        student_name = (request.form.get("student_name") or "").strip()
        roll_no = (request.form.get("roll_no") or "").strip()
        exam_id = request.form.get("exam_id")
        file = request.files.get("file")

        errors = []
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
            selected_exam = None
            try:
                if exam_id_int is not None:
                    selected_exam = Exam.query.get(exam_id_int)
            except Exception:  # noqa: BLE001
                selected_exam = None
            return render_template("upload.html", exams=exams, selected_exam=selected_exam)

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)

        ext = filename.rsplit(".", 1)[1].lower()
        from uuid import uuid4

        # Find or create the student using roll number; name is
        # optional if the student is already registered.
        student = Student.query.filter_by(roll_no=roll_no).first()
        if student is None:
            if not student_name:
                errors.append(
                    "No student with this roll number. Provide a name to create one, or add the student via Manage Students.",
                )
                for e in errors:
                    flash(e, "error")
                exams = Exam.query.order_by(Exam.exam_id.asc()).all()
                selected_exam = Exam.query.get(exam_id_int) if exam_id_int is not None else None
                return render_template("upload.html", exams=exams, selected_exam=selected_exam)

            # Course and semester are required in the model, so we store
            # placeholder values when creating from this flow.
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

        # Run OCR via Gemini first (if configured), otherwise fall back to
        # the existing local OCR pipeline. Evaluation happens later.
        raw_text = ""
        confidence = 0.0

        use_gemini = os.environ.get("USE_GEMINI", "").lower() in {"1", "true", "yes"}
        if use_gemini:
            try:  # pragma: no cover - depends on external API
                from gemini_ocr_client import (
                    GeminiConfigError,
                    extract_text as gemini_extract,
                )

                current_app.logger.info("Attempting Gemini OCR for path: %s", full_path)
                g_text = gemini_extract(full_path)
                if g_text:
                    raw_text = g_text
                    confidence = 0.98
                else:
                    raise RuntimeError("Empty text from Gemini OCR")
            except GeminiConfigError as exc:
                current_app.logger.warning(
                    "Gemini misconfigured, falling back to local OCR: %s",
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                current_app.logger.exception(
                    "Gemini OCR failed, falling back to local OCR: %s",
                    exc,
                )

        if not raw_text:
            raw_text, confidence = run_ocr(full_path)

        cleaned_text = preprocess_text(raw_text)

        extracted = ExtractedText(
            sheet_id=sheet.sheet_id,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extraction_confidence=confidence,
        )
        db.session.add(extracted)

        db.session.commit()

        flash("Text extracted successfully. You can now run evaluation.", "success")

        exams = Exam.query.order_by(Exam.exam_id.asc()).all()
        return render_template(
            "upload.html",
            exams=exams,
            sheet=sheet,
            extracted=extracted,
        )

    exams = Exam.query.order_by(Exam.exam_id.asc()).all()

    selected_exam = None
    exam_id_raw = request.args.get("exam_id")
    if exam_id_raw:
        try:
            exam_id_int = int(exam_id_raw)
            selected_exam = Exam.query.get(exam_id_int)
        except (TypeError, ValueError):
            selected_exam = None

    return render_template("upload.html", exams=exams, selected_exam=selected_exam)


@web_bp.route("/teacher/evaluate/select-exam")
@login_required_view
@role_required_view({UserRole.TEACHER})
def select_exam_for_evaluation():
    """Show a list of exams to start an evaluation flow.

    The teacher first chooses the exam, then the next page only
    requires the student's roll number and the answer sheet.
    """

    exams = Exam.query.order_by(Exam.exam_id.asc()).all()
    return render_template("evaluate_select_exam.html", exams=exams)


@web_bp.route("/teacher/student/lookup")
@login_required_view
@role_required_view({UserRole.TEACHER})
def lookup_student_by_roll():
    """Lookup a student by roll number for the upload page.

    Returns JSON with {found: bool, name, course, semester} so the
    frontend can auto-fill the student name when the roll number
    already exists.
    """

    roll_no = (request.args.get("roll_no") or "").strip()
    if not roll_no:
        return jsonify({"found": False}), 200

    student = Student.query.filter_by(roll_no=roll_no).first()
    if student is None:
        return jsonify({"found": False}), 200

    return jsonify(
        {
            "found": True,
            "name": student.name,
            "course": student.course,
            "semester": student.semester,
        }
    )


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

    # Build per-question details for display from stored question scores.
    per_question_details = []
    if extracted is not None and evaluation is not None:
        # Use raw OCR text for splitting so that question numbers are
        # preserved even if later preprocessing rewrites the text.
        answers_by_q = {
            q_no: ans for q_no, ans in split_numbered_answers(extracted.raw_text)
        }

        for qe in sorted(evaluation.question_scores, key=lambda x: x.question_no):
            per_question_details.append(
                {
                    "question_no": qe.question_no,
                    "answer_text": answers_by_q.get(qe.question_no, ""),
                    "score": qe.score,
                    "feedback": qe.feedback or "",
                }
            )

    # Load any student comments for this sheet; when the teacher opens
    # the review page, mark unresolved comments as resolved so they no
    # longer appear as pending on the dashboard.
    student_comments = (
        QuestionStudentComment.query.filter_by(sheet_id=sheet.sheet_id)
        .order_by(QuestionStudentComment.created_at.asc())
        .all()
    )
    any_unresolved = False
    for c in student_comments:
        if not c.resolved:
            c.resolved = True
            any_unresolved = True
    if any_unresolved:
        db.session.commit()

    # Questions for which the student has requested a review, used
    # to visually highlight those rows in the teacher view.
    commented_q_nos = {c.question_no for c in student_comments}

    if request.method == "POST":
        if sheet.status not in {AnswerSheetStatus.GRADED, AnswerSheetStatus.REVIEWED}:
            flash("Only graded or reviewed answer sheets can be overridden.", "error")
            return redirect(url_for("web.review_page", sheet_id=sheet_id))

        # Update per-question scores and feedback
        question_nos = request.form.getlist("question_no")
        updated_scores = []

        for q_no_str in question_nos:
            qe = next(
                (x for x in evaluation.question_scores if str(x.question_no) == q_no_str),
                None,
            )
            if qe is None:
                continue

            score_field = f"score_{q_no_str}"
            feedback_field = f"feedback_{q_no_str}"
            score_raw = request.form.get(score_field)
            try:
                score_val = float(score_raw)
            except (TypeError, ValueError):
                flash(f"Score for question {q_no_str} must be a number.", "error")
                return render_template(
                    "review.html",
                    sheet=sheet,
                    extracted=extracted,
                    evaluation=evaluation,
                    per_question_details=per_question_details,
                    file_url=url_for(
                        "web.uploaded_file",
                        filename=os.path.basename(sheet.file_path),
                    ),
                    student_comments=student_comments,
                    commented_q_nos=commented_q_nos,
                )

            qe.score = score_val
            qe.feedback = request.form.get(feedback_field) or None
            updated_scores.append(score_val)

        if not updated_scores:
            flash("No question scores were updated.", "error")
            return render_template(
                "review.html",
                sheet=sheet,
                extracted=extracted,
                evaluation=evaluation,
                per_question_details=per_question_details,
                file_url=url_for(
                    "web.uploaded_file",
                    filename=os.path.basename(sheet.file_path),
                ),
                student_comments=student_comments,
                commented_q_nos=commented_q_nos,
            )

        # Recompute overall score as the sum of question scores
        final_score = round(sum(updated_scores), 2)
        evaluation.score = final_score
        evaluation.feedback = (
            evaluation.feedback or ""
        ) + " (Adjusted via question-wise review.)"
        evaluation.evaluated_on = datetime.utcnow()
        sheet.status = AnswerSheetStatus.REVIEWED

        # Ensure the per-exam report reflects the updated total score
        report = Report.query.filter_by(
            student_id=sheet.student_id,
            exam_id=sheet.exam_id,
        ).first()
        if report is None:
            report = Report(
                student_id=sheet.student_id,
                exam_id=sheet.exam_id,
                total_score=final_score,
                remarks=evaluation.feedback,
                generated_on=datetime.utcnow(),
            )
            db.session.add(report)
        else:
            report.total_score = final_score
            report.remarks = evaluation.feedback
            report.generated_on = datetime.utcnow()

        db.session.commit()

        flash("Question-wise scores updated and sheet marked as Reviewed.", "success")
        return redirect(url_for("web.teacher_dashboard"))

    file_url = url_for("web.uploaded_file", filename=os.path.basename(sheet.file_path))
    return render_template(
        "review.html",
        sheet=sheet,
        extracted=extracted,
        evaluation=evaluation,
        per_question_details=per_question_details,
        file_url=file_url,
        student_comments=student_comments,
        commented_q_nos=commented_q_nos,
    )


@web_bp.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    return send_from_directory(upload_folder, filename)


@web_bp.route("/student/report", methods=["GET", "POST"])
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

    # If this is a POST, handle inline updates to course/semester first
    if request.method == "POST":
        course = (request.form.get("course") or "").strip()
        semester = (request.form.get("semester") or "").strip()

        if not course or not semester:
            flash("Course and semester cannot be empty.", "error")
        else:
            student.course = course
            student.semester = semester
            db.session.commit()
            flash("Details updated successfully.", "success")

        return redirect(url_for("web.student_report"))

    # Ensure reports exist for any reviewed sheets for this student
    reviewed_sheets = (
        AnswerSheet.query.filter_by(
            student_id=student.student_id,
            status=AnswerSheetStatus.REVIEWED,
        )
        .order_by(AnswerSheet.exam_id.asc(), AnswerSheet.sheet_id.asc())
        .all()
    )

    sheets_by_exam = {}
    for sheet in reviewed_sheets:
        sheets_by_exam.setdefault(sheet.exam_id, []).append(sheet)

    for exam_id, sheets in sheets_by_exam.items():
        existing = Report.query.filter_by(student_id=student.student_id, exam_id=exam_id).first()
        if existing is not None:
            continue

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
            continue

        # Use the best (maximum) score across attempts for this exam
        total_score = round(max(scores), 2)
        remarks = " \n".join(remarks_parts) if remarks_parts else None

        new_report = Report(
            student_id=student.student_id,
            exam_id=exam_id,
            total_score=total_score,
            remarks=remarks,
            generated_on=datetime.utcnow(),
        )
        db.session.add(new_report)

    if sheets_by_exam:
        db.session.commit()

    reports = Report.query.filter_by(student_id=student.student_id).all()
    return render_template("student_report.html", student=student, reports=reports)


@web_bp.route("/student/report/<int:exam_id>", methods=["GET", "POST"])
@login_required_view
@role_required_view({UserRole.STUDENT})
def student_exam_report(exam_id: int):
    """Detailed question-wise report for a single exam for the logged-in student.

    Also allows the student to submit up to 2 comments about specific
    questions on their latest reviewed answer sheet for this exam.
    """

    user = session.get("user")
    student_id = user.get("student_id")
    student = None
    if student_id is not None:
        try:
            student = Student.query.get(int(student_id))
        except (TypeError, ValueError):
            student = None

    if student is None:
        flash(
            "No student record linked to this user. Ask admin/teacher to create one.",
            "error",
        )
        return redirect(url_for("web.student_report"))

    report = Report.query.filter_by(student_id=student.student_id, exam_id=exam_id).first()
    if report is None:
        flash("No report available for this exam.", "error")
        return redirect(url_for("web.student_report"))

    exam = report.exam

    # Pick the latest reviewed sheet for this student and exam
    sheet = (
        AnswerSheet.query.filter_by(
            student_id=student.student_id,
            exam_id=exam_id,
            status=AnswerSheetStatus.REVIEWED,
        )
        .order_by(AnswerSheet.sheet_id.desc())
        .first()
    )

    if request.method == "POST":
        if sheet is None:
            flash("No reviewed answer sheet found for this exam.", "error")
            return redirect(url_for("web.student_exam_report", exam_id=exam_id))

        comment_text = (request.form.get("comment_text") or "").strip()
        question_no_raw = request.form.get("question_no") or ""
        try:
            question_no = int(question_no_raw)
        except (TypeError, ValueError):
            question_no = None

        if not comment_text or question_no is None:
            flash("Please select a question and enter your comment.", "error")
            return redirect(url_for("web.student_exam_report", exam_id=exam_id))

        # Enforce a maximum of 2 comments per student per sheet
        existing_count = (
            QuestionStudentComment.query.filter_by(
                student_id=student.student_id,
                sheet_id=sheet.sheet_id,
            ).count()
        )
        if existing_count >= 2:
            flash("You have already used your 2 comments for this exam.", "error")
            return redirect(url_for("web.student_exam_report", exam_id=exam_id))

        new_comment = QuestionStudentComment(
            student_id=student.student_id,
            sheet_id=sheet.sheet_id,
            question_no=question_no,
            comment=comment_text,
        )
        db.session.add(new_comment)
        db.session.commit()

        flash("Your comment has been sent to the teacher.", "success")
        return redirect(url_for("web.student_exam_report", exam_id=exam_id))

    per_question_details = []
    evaluation = None
    teacher_feedback_by_q: dict[int, str] = {}
    if sheet is not None and sheet.extracted_text is not None:
        extracted = sheet.extracted_text
        evaluation = extracted.evaluation
        if evaluation is not None:
            # Use raw OCR text so question numbers align exactly with stored scores
            answers_by_q = {
                q_no: ans for q_no, ans in split_numbered_answers(extracted.raw_text)
            }
            for qe in sorted(evaluation.question_scores, key=lambda x: x.question_no):
                per_question_details.append(
                    {
                        "question_no": qe.question_no,
                        "answer_text": answers_by_q.get(qe.question_no, ""),
                        "score": qe.score,
                        "feedback": qe.feedback or "",
                    }
                )

        teacher_feedback_by_q = {
            item["question_no"]: item["feedback"] for item in per_question_details
        }

    # Existing student comments for this sheet and remaining slots
    existing_comments = []
    remaining_comment_slots = 0
    file_url = None
    if sheet is not None:
        existing_comments = (
            QuestionStudentComment.query.filter_by(
                student_id=student.student_id,
                sheet_id=sheet.sheet_id,
            )
            .order_by(QuestionStudentComment.created_at.asc())
            .all()
        )
        remaining_comment_slots = max(0, 2 - len(existing_comments))

        file_url = url_for(
            "web.uploaded_file",
            filename=os.path.basename(sheet.file_path),
        )

    return render_template(
        "student_exam_detail.html",
        student=student,
        exam=exam,
        report=report,
        sheet=sheet,
        evaluation=evaluation,
        per_question_details=per_question_details,
        existing_comments=existing_comments,
        remaining_comment_slots=remaining_comment_slots,
        teacher_feedback_by_q=teacher_feedback_by_q,
        file_url=file_url,
    )


@web_bp.route("/teacher/report/<int:student_id>/<int:exam_id>")
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def teacher_report(student_id: int, exam_id: int):
    """Return or generate a report for a given student/exam for teacher view.

    This mirrors the logic of the JWT-protected /report API but uses the
    server-side session instead of a JWT, so teachers can click from the
    dashboard without providing an Authorization header.
    """

    # If a report already exists, return it / reuse it
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

        # Use the best (maximum) score across attempts for this exam
        total_score = round(max(scores), 2)
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

    # Build detailed per-question view based on the latest reviewed sheet
    exam = report.exam
    student = report.student
    sheet = (
        AnswerSheet.query.filter_by(
            student_id=student_id,
            exam_id=exam_id,
            status=AnswerSheetStatus.REVIEWED,
        )
        .order_by(AnswerSheet.sheet_id.desc())
        .first()
    )

    per_question_details = []
    evaluation = None
    if sheet is not None and sheet.extracted_text is not None:
        extracted = sheet.extracted_text
        evaluation = extracted.evaluation
        if evaluation is not None:
            # Use raw OCR text so question numbers align exactly with stored scores
            answers_by_q = {
                q_no: ans
                for q_no, ans in split_numbered_answers(extracted.raw_text)
            }
            for qe in sorted(evaluation.question_scores, key=lambda x: x.question_no):
                per_question_details.append(
                    {
                        "question_no": qe.question_no,
                        "answer_text": answers_by_q.get(qe.question_no, ""),
                        "score": qe.score,
                        "feedback": qe.feedback or "",
                    }
                )

    return render_template(
        "teacher_exam_report.html",
        student=student,
        exam=exam,
        report=report,
        sheet=sheet,
        evaluation=evaluation,
        per_question_details=per_question_details,
    )


@web_bp.route("/teacher/evaluation/<int:student_id>/<int:exam_id>/delete", methods=["POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def delete_student_evaluation(student_id: int, exam_id: int):
    """Remove a student's evaluation data for a specific exam.

    Deletes all answer sheets for this (student, exam) pair along with
    their extracted text, evaluations, question scores, and any
    generated report. This lets teachers fully reset an exam's
    evaluation for a student.
    """

    sheets = AnswerSheet.query.filter_by(student_id=student_id, exam_id=exam_id).all()

    for sheet in sheets:
        extracted = sheet.extracted_text
        if extracted is not None:
            evaluation = extracted.evaluation
            if evaluation is not None:
                db.session.delete(evaluation)
            db.session.delete(extracted)

        db.session.delete(sheet)

    Report.query.filter_by(student_id=student_id, exam_id=exam_id).delete(
        synchronize_session=False
    )

    db.session.commit()

    flash("Student evaluation for this exam has been removed.", "success")
    return redirect(url_for("web.evaluated_students", exam_id=exam_id))
