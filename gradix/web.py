import os
import re
import json
import tempfile
from datetime import datetime
from functools import wraps
from uuid import uuid4

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


def _parse_students_from_ocr_text(text: str) -> list[dict[str, str]]:
    """Parse OCR text into a list of {name, roll_no} dicts.

    Expected line format (per student):

        Name - RollNumber

    Lines without a dash or with missing parts are ignored.
    """

    students: list[dict[str, str]] = []
    if not text:
        return students

    for raw_line in text.splitlines():
        line = (raw_line or "").strip()
        if not line:
            continue

        # Normalize common dash variants
        line = line.replace("\u2013", "-").replace("\u2014", "-")
        if "-" not in line:
            continue

        name_part, roll_part = line.split("-", 1)
        name = name_part.strip(" \t:-")
        roll_no = roll_part.strip(" \t:-")

        if not name or not roll_no:
            continue

        students.append({"name": name, "roll_no": roll_no})

    return students


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


def _apply_or_group_scoring(exam_questions, per_q: list[dict]) -> tuple[list[dict], float]:
    """Compute total score with OR-groups, without altering per-question scores.

    When two or more questions in ``exam_questions`` share the same
    non-null ``or_group`` value, they are treated as alternatives and
    only the highest-scoring one in that group contributes to the
    overall total. Individual per-question scores in ``per_q`` are
    left unchanged so that teachers can still see marks for all
    answered questions in the review UI.
    """

    if not exam_questions or not per_q:
        total = sum(float(p.get("score", 0.0) or 0.0) for p in per_q) if per_q else 0.0
        return per_q, float(total)

    # Map question number to its raw score from the evaluation output.
    raw_scores: dict[int, float] = {}
    for item in per_q:
        try:
            q_no = int(item.get("question_no"))
        except (TypeError, ValueError):
            continue
        try:
            score_val = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_val = 0.0
        raw_scores[q_no] = score_val

    # Build OR groups from the exam definition.
    groups: dict[int, list[int]] = {}
    for eq in exam_questions:
        group_id = getattr(eq, "or_group", None)
        if group_id is None:
            continue
        groups.setdefault(group_id, []).append(eq.question_no)

    total = 0.0
    used_in_group: set[int] = set()

    # For each OR group, add only the highest-scoring question.
    for _, q_list in groups.items():
        if not q_list:
            continue
        best_score = 0.0
        for q_no in q_list:
            used_in_group.add(q_no)
            score = raw_scores.get(q_no, 0.0)
            if score > best_score:
                best_score = score
        total += best_score

    # Questions not in any OR group contribute their full score.
    for eq in exam_questions:
        if getattr(eq, "or_group", None) is None:
            total += raw_scores.get(eq.question_no, 0.0)

    return per_q, float(total)


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

                # Optional OR-group: when multiple questions share the
                # same positive integer group value, they are treated
                # as alternatives (student should answer any one). At
                # evaluation time only the highest-scoring question in
                # each OR group contributes to the total.
                or_group_raw = request.form.get(f"or_group_{i}") or "" 

                try:
                    marks_val = float(marks_raw) if marks_raw is not None else None
                    if marks_val is not None and marks_val <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"Question {i}: marks must be a positive number.")
                    marks_val = None

                or_group_val = None
                if or_group_raw.strip():
                    try:
                        or_group_val = int(or_group_raw)
                        if or_group_val <= 0:
                            raise ValueError
                    except (TypeError, ValueError):
                        errors.append(
                            f"Question {i}: OR group must be a positive integer when provided.",
                        )
                        or_group_val = None

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
                        "or_group": or_group_val,
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
                or_group=item.get("or_group"),
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

        # First pass: build raw question entries and detect standalone
        # "OR" lines that indicate alternative questions.
        tmp_items: list[dict] = []
        for q_no, q_text in segments:
            original_text = (q_text or "").strip()
            if not original_text:
                continue

            text = original_text

            # Try to extract marks from common patterns such as
            # "(5 marks)", plain "(3)" at the end of the question,
            # "[5]", "5M", etc. If multiple marks are present in
            # the same block (e.g. parts (a) and (b) each "(6)"), we
            # treat the total marks for the question as the SUM of all
            # detected values.
            marks_val = None
            patterns = [
                r"\((\d+(?:\.\d+)?)\s*marks?\)",
                r"\((\d+(?:\.\d+)?)\)",
                r"\[(\d+(?:\.\d+)?)\]",
                r"(\d+(?:\.\d+)?)\s*[Mm]arks?",
                r"(\d+(?:\.\d+)?)\s*[Mm]",
            ]

            all_marks: list[float] = []
            for pat in patterns:
                for m in re.finditer(pat, text):
                    try:
                        all_marks.append(float(m.group(1)))
                    except (TypeError, ValueError):
                        continue

            if all_marks:
                marks_val = float(sum(all_marks))
                # Remove ALL explicit mark fragments so the question
                # text shown in the UI is clean.
                for pat in patterns:
                    text = re.sub(pat, "", text).rstrip()

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

            # Detect if this question's original OCR block contains a
            # standalone "OR" line. This commonly appears between
            # main questions to indicate "Answer any one".
            has_or_line = any(ln.strip().upper() == "OR" for ln in original_text.splitlines())

            tmp_items.append(
                {
                    "question_no": int(q_no),
                    "question_text": text,
                    "marks": marks_val,
                    "_has_or": has_or_line,
                }
            )

        # Second pass: assign OR-group ids based on detected OR lines.
        #
        # Heuristic: when a question's OCR block contains a standalone
        # "OR" line and there is a following question, treat this
        # pair as alternatives (same OR group). This works well for
        # layouts where "OR" appears between two numbered questions.
        questions_out: list[dict] = []
        next_group_id = 1
        i = 0
        while i < len(tmp_items):
            item = dict(tmp_items[i])
            or_group_val = None

            if item.get("_has_or") and i + 1 < len(tmp_items):
                or_group_val = next_group_id
                # Also assign the same group id to the immediate next
                # question.
                next_item = dict(tmp_items[i + 1])
                next_item["or_group"] = or_group_val
                next_item.pop("_has_or", None)
                questions_out.append(
                    {
                        "question_no": item["question_no"],
                        "question_text": item["question_text"],
                        "marks": item["marks"],
                        "or_group": or_group_val,
                    }
                )
                questions_out.append(
                    {
                        "question_no": next_item["question_no"],
                        "question_text": next_item["question_text"],
                        "marks": next_item["marks"],
                        "or_group": or_group_val,
                    }
                )
                next_group_id += 1
                i += 2
                continue

            item.pop("_has_or", None)
            questions_out.append(
                {
                    "question_no": item["question_no"],
                    "question_text": item["question_text"],
                    "marks": item["marks"],
                    "or_group": None,
                }
            )
            i += 1

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
    teachers can quickly find students by department/section. It also
    supports bulk creation via OCR of an uploaded image.
    """

    if request.method == "POST":
        action = (request.form.get("action") or "single_add").strip()

        if action == "single_add":
            name = (request.form.get("name") or "").strip()
            roll_no = (request.form.get("roll_no") or "").strip()
            course = (request.form.get("course") or "").strip()
            semester = (request.form.get("semester") or "").strip()

            errors: list[str] = []
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

        if action == "ocr_extract":
            ocr_course = (request.form.get("ocr_course") or "").strip()
            ocr_semester = (request.form.get("ocr_semester") or "").strip()
            file = request.files.get("ocr_image")

            if not file or file.filename == "":
                flash("Please choose an image file for OCR.", "error")
                return redirect(url_for("web.manage_students"))

            fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1] or ".png")
            os.close(fd)
            try:
                file.save(tmp_path)

                # Prefer Gemini Vision for structured extraction. If it is
                # misconfigured or fails, fall back to local OCR + regex.
                confidence = 0.0
                students_raw: list[dict[str, str]] | None = None
                try:  # pragma: no cover - depends on external API
                    from gemini_ocr_client import (
                        GeminiConfigError,
                        extract_students_from_image,
                    )

                    students_raw = extract_students_from_image(tmp_path)
                    # Heuristic confidence for UI purposes only.
                    confidence = 0.95 if students_raw else 0.0
                except GeminiConfigError as exc:
                    current_app.logger.warning(
                        "Gemini for student OCR not configured, using local OCR: %s",
                        exc,
                    )
                except Exception as exc:  # noqa: BLE001
                    current_app.logger.exception(
                        "Gemini student OCR failed, falling back to local OCR: %s",
                        exc,
                    )

                if students_raw is None:
                    text, confidence = run_ocr(tmp_path)
                    students_raw = _parse_students_from_ocr_text(text)

                if not students_raw:
                    flash(
                        "No valid 'Name - RollNumber' lines were found in the extracted text.",
                        "error",
                    )
                    return redirect(url_for("web.manage_students"))

                # Mark which roll numbers already exist in the database
                roll_nos = [s["roll_no"] for s in students_raw]
                existing = (
                    Student.query.filter(Student.roll_no.in_(roll_nos)).all()
                    if roll_nos
                    else []
                )
                existing_rolls = {s.roll_no for s in existing}

                ocr_candidates: list[dict] = []
                addable: list[dict] = []
                for item in students_raw:
                    exists = item["roll_no"] in existing_rolls
                    record = {
                        "name": item["name"],
                        "roll_no": item["roll_no"],
                        "exists": exists,
                    }
                    ocr_candidates.append(record)
                    if not exists:
                        addable.append(record)

                if not addable:
                    flash("All extracted students already exist. Nothing to add.", "info")
                    return redirect(url_for("web.manage_students"))

                # Build filter options for the page render
                course_filter = (request.args.get("course") or "").strip()
                semester_filter = (request.args.get("semester") or "").strip()

                query = Student.query
                if course_filter:
                    query = query.filter(Student.course == course_filter)
                if semester_filter:
                    query = query.filter(Student.semester == semester_filter)

                students = query.order_by(Student.roll_no.asc()).all()

                courses = [
                    row[0]
                    for row in db.session.query(Student.course)
                    .distinct()
                    .order_by(Student.course.asc())
                    .all()
                ]
                semesters = [
                    row[0]
                    for row in db.session.query(Student.semester)
                    .distinct()
                    .order_by(Student.semester.asc())
                    .all()
                ]

                return render_template(
                    "students_manage.html",
                    students=students,
                    courses=courses,
                    semesters=semesters,
                    selected_course=course_filter,
                    selected_semester=semester_filter,
                    ocr_candidates=ocr_candidates,
                    ocr_addable=addable,
                    ocr_course=ocr_course,
                    ocr_semester=ocr_semester,
                    ocr_confidence=confidence,
                )
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    current_app.logger.info(
                        "Could not remove temporary student OCR image: %s", tmp_path
                    )

        if action == "ocr_add":
            ocr_course = (request.form.get("ocr_course") or "").strip()
            ocr_semester = (request.form.get("ocr_semester") or "").strip()
            payload_raw = request.form.get("ocr_payload") or "[]"

            try:
                records = json.loads(payload_raw)
            except Exception:  # noqa: BLE001
                flash("Invalid OCR data submitted. Please extract again.", "error")
                return redirect(url_for("web.manage_students"))

            created_count = 0
            for item in records:
                name = (item.get("name") or "").strip()
                roll_no = (item.get("roll_no") or "").strip()
                if not name or not roll_no:
                    continue
                if Student.query.filter_by(roll_no=roll_no).first() is not None:
                    continue

                student = Student(
                    name=name,
                    roll_no=roll_no,
                    course=ocr_course,
                    semester=ocr_semester,
                )
                db.session.add(student)
                created_count += 1

            if created_count:
                db.session.commit()
                flash(f"Added {created_count} students from OCR.", "success")
            else:
                flash("No new students were added from OCR data.", "info")

            return redirect(url_for("web.manage_students"))

    # Build filter options
    course_filter = (request.args.get("course") or "").strip()
    semester_filter = (request.args.get("semester") or "").strip()

    query = Student.query
    if course_filter:
        query = query.filter(Student.course == course_filter)
    if semester_filter:
        query = query.filter(Student.semester == semester_filter)

    students = query.order_by(Student.roll_no.asc()).all()

    courses = [row[0] for row in db.session.query(Student.course).distinct().order_by(Student.course.asc()).all()]
    semesters = [row[0] for row in db.session.query(Student.semester).distinct().order_by(Student.semester.asc()).all()]

    return render_template(
        "students_manage.html",
        students=students,
        courses=courses,
        semesters=semesters,
        selected_course=course_filter,
        selected_semester=semester_filter,
        ocr_candidates=None,
        ocr_addable=None,
        ocr_course=None,
        ocr_semester=None,
        ocr_confidence=None,
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

            # Absolute path to the uploaded answer sheet file (image or PDF)
            upload_folder = current_app.config["UPLOAD_FOLDER"]
            sheet_filename = os.path.basename(sheet.file_path)
            sheet_abs_path = os.path.join(upload_folder, sheet_filename)

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

                    gemini_result = evaluate_answers_with_gemini(
                        payload_items,
                        sheet_image_path=sheet_abs_path,
                    )
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

                    # Apply OR-group semantics so that for any set of
                    # alternative questions, only the highest-scoring
                    # one contributes to the total.
                    per_q, total_score = _apply_or_group_scoring(exam_questions, per_q)

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
                    if exam_questions:
                        per_q, total_score = _apply_or_group_scoring(exam_questions, per_q)
                        final_score = round(float(total_score), 2)
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
                    if exam_questions:
                        per_q, total_score = _apply_or_group_scoring(exam_questions, per_q)
                        final_score = round(float(total_score), 2)
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

        # If the teacher clicks Extract Text for an existing sheet
        # (uploaded previously by a student), run OCR and store
        # ExtractedText without creating a new AnswerSheet.
        if action == "extract" and request.form.get("sheet_id"):
            try:
                sheet_id_int = int(request.form.get("sheet_id"))
            except (TypeError, ValueError):
                flash("Invalid sheet id for extraction.", "error")
                exams = Exam.query.order_by(Exam.exam_id.asc()).all()
                return render_template("upload.html", exams=exams)

            sheet = AnswerSheet.query.get(sheet_id_int)
            if sheet is None:
                flash("Answer sheet not found for extraction.", "error")
                exams = Exam.query.order_by(Exam.exam_id.asc()).all()
                return render_template("upload.html", exams=exams)

            upload_folder = current_app.config["UPLOAD_FOLDER"]
            os.makedirs(upload_folder, exist_ok=True)

            sheet_filename = os.path.basename(sheet.file_path)
            full_path = os.path.join(upload_folder, sheet_filename)

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

            extracted = sheet.extracted_text
            if extracted is None:
                extracted = ExtractedText(
                    sheet_id=sheet.sheet_id,
                    raw_text=raw_text,
                    cleaned_text=cleaned_text,
                    extraction_confidence=confidence,
                )
                db.session.add(extracted)
            else:
                extracted.raw_text = raw_text
                extracted.cleaned_text = cleaned_text
                extracted.extraction_confidence = confidence

            db.session.commit()

            flash("Text extracted successfully. You can now run evaluation.", "success")

            exams = Exam.query.order_by(Exam.exam_id.asc()).all()
            file_url = url_for(
                "web.uploaded_file",
                filename=os.path.basename(sheet.file_path),
            )
            selected_exam = sheet.exam
            return render_template(
                "upload.html",
                exams=exams,
                selected_exam=selected_exam,
                sheet=sheet,
                extracted=extracted,
                file_url=file_url,
            )

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
        file_url = url_for(
            "web.uploaded_file",
            filename=os.path.basename(sheet.file_path),
        )
        return render_template(
            "upload.html",
            exams=exams,
            selected_exam=exam,
            sheet=sheet,
            extracted=extracted,
            file_url=file_url,
        )

    exams = Exam.query.order_by(Exam.exam_id.asc()).all()

    sheet = None
    extracted = None
    file_url = None

    sheet_id_raw = request.args.get("sheet_id")
    if sheet_id_raw:
        try:
            sheet_id_int = int(sheet_id_raw)
            sheet = AnswerSheet.query.get(sheet_id_int)
        except (TypeError, ValueError):
            sheet = None

    selected_exam = None
    exam_id_raw = request.args.get("exam_id")
    if exam_id_raw:
        try:
            exam_id_int = int(exam_id_raw)
            selected_exam = Exam.query.get(exam_id_int)
        except (TypeError, ValueError):
            selected_exam = None
    elif sheet is not None:
        selected_exam = sheet.exam

    if sheet is not None:
        file_url = url_for(
            "web.uploaded_file",
            filename=os.path.basename(sheet.file_path),
        )
        extracted = sheet.extracted_text

    return render_template(
        "upload.html",
        exams=exams,
        selected_exam=selected_exam,
        sheet=sheet,
        extracted=extracted,
        file_url=file_url,
    )


@web_bp.route("/teacher/evaluate/select-exam")
@login_required_view
@role_required_view({UserRole.TEACHER})
def select_exam_for_evaluation():
    """Show a list of exams to start an evaluation flow.

    The teacher first chooses the exam, then the next page only
    requires the student's roll number and the answer sheet.
    """

    exams = Exam.query.order_by(Exam.exam_id.asc()).all()

    # For each exam, count how many uploaded sheets are still pending evaluation
    pending_counts_rows = (
        db.session.query(AnswerSheet.exam_id, db.func.count(AnswerSheet.sheet_id))
        .filter(AnswerSheet.status == AnswerSheetStatus.PENDING)
        .group_by(AnswerSheet.exam_id)
        .all()
    )
    pending_by_exam = {exam_id: count for exam_id, count in pending_counts_rows}

    return render_template(
        "evaluate_select_exam.html",
        exams=exams,
        pending_by_exam=pending_by_exam,
    )


@web_bp.route("/teacher/exam/<int:exam_id>/sheets")
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def teacher_exam_sheets(exam_id: int):
    """List all uploaded answer sheets for a given exam for teachers.

    From here a teacher can pick a student's uploaded sheet and run
    OCR + evaluation without uploading files themselves.
    """

    exam = Exam.query.get_or_404(exam_id)

    # All sheets uploaded for this exam
    sheets = (
        AnswerSheet.query.filter_by(exam_id=exam.exam_id)
        .order_by(AnswerSheet.upload_date.desc())
        .all()
    )

    # Group sheets by evaluation status for clearer sections in the UI.
    pending_sheets = [s for s in sheets if s.status == AnswerSheetStatus.PENDING]
    evaluated_sheets = [
        s
        for s in sheets
        if s.status in {AnswerSheetStatus.GRADED, AnswerSheetStatus.REVIEWED}
    ]

    # Students without an uploaded sheet for this exam.
    all_students = Student.query.order_by(Student.roll_no.asc()).all()
    students_with_sheet = {s.student_id for s in sheets}
    missing_students = [
        st for st in all_students if st.student_id not in students_with_sheet
    ]

    return render_template(
        "teacher_exam_sheets.html",
        exam=exam,
        sheets=sheets,
        pending_sheets=pending_sheets,
        evaluated_sheets=evaluated_sheets,
        missing_students=missing_students,
        AnswerSheetStatus=AnswerSheetStatus,
    )


@web_bp.route("/teacher/sheet/<int:sheet_id>/extract-evaluate", methods=["POST"])
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def teacher_extract_evaluate_sheet(sheet_id: int):
    """Run OCR and evaluation for an existing uploaded answer sheet.

    This is used when students upload their own answer sheets; the
    teacher only needs to trigger extraction/evaluation.
    """

    sheet = AnswerSheet.query.get_or_404(sheet_id)
    exam = sheet.exam

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)

    sheet_filename = os.path.basename(sheet.file_path)
    sheet_abs_path = os.path.join(upload_folder, sheet_filename)

    # If OCR has not yet been run, perform extraction now.
    extracted = sheet.extracted_text
    if extracted is None:
        raw_text = ""
        confidence = 0.0

        use_gemini = os.environ.get("USE_GEMINI", "").lower() in {"1", "true", "yes"}
        if use_gemini:
            try:  # pragma: no cover - depends on external API
                from gemini_ocr_client import (
                    GeminiConfigError,
                    extract_text as gemini_extract,
                )

                current_app.logger.info("Attempting Gemini OCR for path: %s", sheet_abs_path)
                g_text = gemini_extract(sheet_abs_path)
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
            raw_text, confidence = run_ocr(sheet_abs_path)

        cleaned_text = preprocess_text(raw_text)

        extracted = ExtractedText(
            sheet_id=sheet.sheet_id,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extraction_confidence=confidence,
        )
        db.session.add(extracted)
        db.session.flush()

    # At this point we must have extracted text for evaluation.
    extracted = sheet.extracted_text
    if extracted is None:
        flash("Could not extract text for this sheet.", "error")
        return redirect(url_for("web.teacher_exam_sheets", exam_id=sheet.exam_id))

    # Use the same evaluation pipeline as the upload page.
    from .models import QuestionEvaluation

    segments = split_numbered_answers(extracted.raw_text)

    exam_questions = sorted(
        getattr(exam, "questions", []),
        key=lambda q: q.question_no,
    )

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
                        "model_answer": eq.answer_text,
                        "max_marks": max_marks_val,
                        "student_answer": answers_by_q.get(eq.question_no, ""),
                    }
                )

            gemini_result = evaluate_answers_with_gemini(
                payload_items,
                sheet_image_path=sheet_abs_path,
            )
            questions_out = gemini_result.get("questions", []) or []

            per_q = []
            for idx, payload in enumerate(payload_items):
                q_no = payload["question_no"]
                out_item = questions_out[idx] if idx < len(questions_out) else {}
                try:
                    raw_score = float(out_item.get("score", 0.0))
                except (TypeError, ValueError):
                    raw_score = 0.0

                max_marks = marks_by_q.get(q_no)
                if (
                    max_marks is not None
                    and max_marks > 1.0
                    and 0.0 <= raw_score <= 1.0
                ):
                    q_score = raw_score * max_marks
                else:
                    q_score = raw_score

                ans_text = payload["student_answer"] or ""
                per_q.append(
                    {
                        "question_no": q_no,
                        "answer_text": ans_text,
                        "score": q_score,
                        "feedback": "",
                    }
                )

            per_q, total_score = _apply_or_group_scoring(exam_questions, per_q)
            final_score = round(float(total_score), 2)
            feedback = "Auto-evaluated using Gemini based on exam rubric."
        except GeminiConfigError as exc:
            current_app.logger.warning(
                "Gemini evaluation misconfigured, falling back to heuristic scoring: %s",
                exc,
            )
            final_score, feedback, per_q = evaluate_text_by_questions(
                extracted.cleaned_text,
                model_answer_text,
            )
            if exam_questions:
                per_q, total_score = _apply_or_group_scoring(exam_questions, per_q)
                final_score = round(float(total_score), 2)
        except Exception as exc:  # noqa: BLE001
            current_app.logger.exception(
                "Gemini evaluation failed, falling back to heuristic scoring: %s",
                exc,
            )
            final_score, feedback, per_q = evaluate_text_by_questions(
                extracted.cleaned_text,
                model_answer_text,
            )
            if exam_questions:
                per_q, total_score = _apply_or_group_scoring(exam_questions, per_q)
                final_score = round(float(total_score), 2)
    else:
        final_score, feedback, per_q = evaluate_text_by_questions(
            extracted.cleaned_text,
            model_answer_text,
        )

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

        for qs in list(evaluation.question_scores):
            db.session.delete(qs)

    exam_questions = sorted(
        getattr(exam, "questions", []),
        key=lambda q: q.question_no,
    )

    if exam_questions:
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
    user = session.get("user") or {}
    teacher_name = user.get("name") or None

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

        # Look up OR-group information for this exam so the review page
        # can indicate which questions are alternatives (e.g. "1 OR 2").
        exam_questions = ExamQuestion.query.filter_by(exam_id=sheet.exam_id).all()
        or_by_q: dict[int, int | None] = {
            eq.question_no: eq.or_group for eq in exam_questions
        }
        group_members: dict[int, list[int]] = {}
        for eq in exam_questions:
            if eq.or_group is None:
                continue
            group_members.setdefault(eq.or_group, []).append(eq.question_no)

        for qe in sorted(evaluation.question_scores, key=lambda x: x.question_no):
            q_no = qe.question_no
            or_group = or_by_q.get(q_no)
            or_peers: list[int] = []
            if or_group is not None:
                members = group_members.get(or_group, [])
                or_peers = sorted(q for q in members if q != q_no)

            per_question_details.append(
                {
                    "question_no": q_no,
                    "answer_text": answers_by_q.get(q_no, ""),
                    "score": qe.score,
                    "feedback": qe.feedback or "",
                    "or_group": or_group,
                    "or_peers": or_peers,
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
        if teacher_name:
            evaluation.reviewed_by = teacher_name
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
        return render_template(
            "student_report.html",
            student=None,
            reports=[],
            exams=[],
            uploads_by_exam={},
            AnswerSheetStatus=AnswerSheetStatus,
        )

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

    # All exams (for allowing uploads by exam from student page)
    exams = Exam.query.order_by(Exam.exam_id.asc()).all()

    # Latest uploaded sheet per exam for this student (if any)
    latest_sheet_by_exam: dict[int, AnswerSheet] = {}
    existing_sheets = (
        AnswerSheet.query.filter_by(student_id=student.student_id)
        .order_by(AnswerSheet.upload_date.desc())
        .all()
    )
    for sheet in existing_sheets:
        if sheet.exam_id not in latest_sheet_by_exam:
            latest_sheet_by_exam[sheet.exam_id] = sheet

    reports = Report.query.filter_by(student_id=student.student_id).all()
    return render_template(
        "student_report.html",
        student=student,
        reports=reports,
        exams=exams,
        uploads_by_exam=latest_sheet_by_exam,
        AnswerSheetStatus=AnswerSheetStatus,
    )


@web_bp.route("/student/exam/<int:exam_id>/upload", methods=["GET", "POST"])
@login_required_view
@role_required_view({UserRole.STUDENT})
def student_exam_upload(exam_id: int):
    """Allow a logged-in student to upload their answer sheet for a given exam.

    A student can upload at most one answer sheet per exam as long as a
    corresponding record exists in the database.
    """

    user = session.get("user")
    student_id = user.get("student_id") if user else None

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

    exam = Exam.query.get_or_404(exam_id)

    # Enforce single upload per student+exam as long as a sheet exists
    existing_sheet = (
        AnswerSheet.query.filter_by(student_id=student.student_id, exam_id=exam.exam_id)
        .order_by(AnswerSheet.upload_date.desc())
        .first()
    )
    if existing_sheet is not None:
        flash(
            "You have already uploaded an answer sheet for this exam.",
            "error",
        )
        return redirect(url_for("web.student_report"))

    if request.method == "POST":
        file = request.files.get("file")

        errors: list[str] = []
        filename = file.filename if file is not None else ""
        if not file or not filename:
            errors.append("File is required.")
        elif not _allowed_file(filename):
            errors.append("Invalid file type. Allowed: PDF, JPG, PNG.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("student_upload.html", exam=exam)

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)

        ext = filename.rsplit(".", 1)[1].lower()
        safe_name = f"{student.student_id}_{exam.exam_id}_{uuid4().hex}.{ext}"
        full_path = os.path.join(upload_folder, safe_name)
        file.save(full_path)

        relative_path = os.path.join("uploads", safe_name)

        sheet = AnswerSheet(
            student_id=student.student_id,
            exam_id=exam.exam_id,
            file_path=relative_path,
            status=AnswerSheetStatus.PENDING,
        )
        db.session.add(sheet)
        db.session.commit()

        flash(
            "Answer sheet uploaded successfully. Your teacher will extract and evaluate it.",
            "success",
        )
        return redirect(url_for("web.student_report"))

    return render_template("student_upload.html", exam=exam)


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
    evaluator_label: str | None = None
    teacher_feedback_by_q: dict[int, str] = {}
    if sheet is not None and sheet.extracted_text is not None:
        extracted = sheet.extracted_text
        evaluation = extracted.evaluation
        if evaluation is not None:
            # Prefer to show the teacher's name only. The UI already
            # labels the field as "Evaluated By", so we don't need a
            # "Reviewed by" prefix here.
            if getattr(evaluation, "reviewed_by", None):
                evaluator_label = evaluation.reviewed_by
            elif sheet.status == AnswerSheetStatus.REVIEWED:
                evaluator_label = "Teacher-reviewed"
            else:
                evaluator_label = "Auto-evaluated using Gemini"

            # Map question numbers to their maximum marks and OR groups from
            # the exam definition.
            marks_by_q: dict[int, float | None] = {}
            or_by_q: dict[int, int | None] = {}
            group_members: dict[int, list[int]] = {}
            if exam is not None:
                for q in sorted(getattr(exam, "questions", []), key=lambda q: q.question_no):
                    marks_by_q[q.question_no] = float(q.marks) if q.marks is not None else None
                    or_by_q[q.question_no] = getattr(q, "or_group", None)
                    if q.or_group is not None:
                        group_members.setdefault(q.or_group, []).append(q.question_no)

            # Use raw OCR text so question numbers align exactly with stored scores
            answers_by_q = {
                q_no: ans for q_no, ans in split_numbered_answers(extracted.raw_text)
            }
            for qe in sorted(evaluation.question_scores, key=lambda x: x.question_no):
                q_no = qe.question_no
                max_marks_val = None
                if exam is not None:
                    max_marks_val = marks_by_q.get(q_no)

                or_group = or_by_q.get(q_no)
                or_peers: list[int] = []
                if or_group is not None:
                    members = group_members.get(or_group, [])
                    or_peers = sorted(q for q in members if q != q_no)

                per_question_details.append(
                    {
                        "question_no": q_no,
                        "answer_text": answers_by_q.get(q_no, ""),
                        "score": qe.score,
                        "max_marks": max_marks_val,
                        "feedback": qe.feedback or "",
                        "or_group": or_group,
                        "or_peers": or_peers,
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
        evaluator_label=evaluator_label,
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
    evaluator_label: str | None = None
    if sheet is not None and sheet.extracted_text is not None:
        extracted = sheet.extracted_text
        evaluation = extracted.evaluation
        if evaluation is not None:
            # Prefer to show the teacher's name only. The UI already
            # labels the field as "Evaluated By".
            if getattr(evaluation, "reviewed_by", None):
                evaluator_label = evaluation.reviewed_by
            elif sheet.status == AnswerSheetStatus.REVIEWED:
                evaluator_label = "Teacher-reviewed"
            else:
                evaluator_label = "Auto-evaluated using Gemini"

            # Map question numbers to their maximum marks and OR groups from the exam definition.
            marks_by_q: dict[int, float | None] = {}
            or_by_q: dict[int, int | None] = {}
            group_members: dict[int, list[int]] = {}
            if exam is not None:
                for q in sorted(getattr(exam, "questions", []), key=lambda q: q.question_no):
                    marks_by_q[q.question_no] = float(q.marks) if q.marks is not None else None
                    or_by_q[q.question_no] = getattr(q, "or_group", None)
                    if q.or_group is not None:
                        group_members.setdefault(q.or_group, []).append(q.question_no)

            # Use raw OCR text so question numbers align exactly with stored scores
            answers_by_q = {
                q_no: ans
                for q_no, ans in split_numbered_answers(extracted.raw_text)
            }
            for qe in sorted(evaluation.question_scores, key=lambda x: x.question_no):
                q_no = qe.question_no
                max_marks_val = None
                if exam is not None:
                    max_marks_val = marks_by_q.get(q_no)

                or_group = or_by_q.get(q_no)
                or_peers: list[int] = []
                if or_group is not None:
                    members = group_members.get(or_group, [])
                    or_peers = sorted(q for q in members if q != q_no)

                per_question_details.append(
                    {
                        "question_no": q_no,
                        "answer_text": answers_by_q.get(q_no, ""),
                        "score": qe.score,
                        "max_marks": max_marks_val,
                        "feedback": qe.feedback or "",
                        "or_group": or_group,
                        "or_peers": or_peers,
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
        evaluator_label=evaluator_label,
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


@web_bp.route("/teacher/exam/<int:exam_id>/report")
@login_required_view
@role_required_view({UserRole.TEACHER, UserRole.ADMIN})
def exam_report_page(exam_id: int):
    """Aggregate report for a single exam.

    Shows submission/evaluation status breakdown, mark distribution,
    and top-performing students for the selected exam.
    """

    exam = Exam.query.get_or_404(exam_id)

    # All students currently in the system
    all_students = Student.query.order_by(Student.roll_no.asc()).all()
    all_student_ids = {s.student_id for s in all_students}

    # All answer sheets for this exam
    sheets = AnswerSheet.query.filter_by(exam_id=exam.exam_id).all()

    pending_student_ids = {
        s.student_id for s in sheets if s.status == AnswerSheetStatus.PENDING
    }
    evaluated_student_ids = {
        s.student_id
        for s in sheets
        if s.status in {AnswerSheetStatus.GRADED, AnswerSheetStatus.REVIEWED}
    }
    students_with_sheet = {s.student_id for s in sheets}
    not_submitted_ids = all_student_ids - students_with_sheet

    status_counts = {
        "submitted_pending": len(pending_student_ids),
        "evaluated": len(evaluated_student_ids),
        "not_submitted": len(not_submitted_ids),
        "total_students": len(all_student_ids),
    }

    # Mark distribution for students who have a Report entry for this exam.
    reports = (
        Report.query.filter_by(exam_id=exam.exam_id)
        .join(Student, Report.student_id == Student.student_id)
        .order_by(Report.total_score.desc())
        .all()
    )

    marks_data = [
        {
            "student_id": r.student.student_id,
            "label": f"{r.student.name} ({r.student.roll_no})",
            "score": float(r.total_score),
        }
        for r in reports
    ]

    top_students = marks_data[:3]

    # Basic mark statistics
    if marks_data:
        scores_only = [m["score"] for m in marks_data]
        avg_score = sum(scores_only) / len(scores_only)
        highest_score = max(scores_only)
        lowest_score = min(scores_only)
    else:
        avg_score = highest_score = lowest_score = None

    marks_stats = {
        "average": avg_score,
        "highest": highest_score,
        "lowest": lowest_score,
        "max_marks": float(exam.max_marks) if exam.max_marks is not None else None,
    }

    # Percentage-based histogram of students in mark ranges
    histogram_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    histogram_counts = [0, 0, 0, 0]
    max_marks_val = (
        float(exam.max_marks)
        if exam.max_marks is not None and float(exam.max_marks) > 0
        else None
    )

    if marks_data and max_marks_val:
        for m in marks_data:
            pct = (m["score"] / max_marks_val) * 100.0
            if pct < 25:
                histogram_counts[0] += 1
            elif pct < 50:
                histogram_counts[1] += 1
            elif pct < 75:
                histogram_counts[2] += 1
            else:
                histogram_counts[3] += 1

    total_marked = sum(histogram_counts)
    if total_marked > 0:
        histogram_percentages = [
            round((c * 100.0) / total_marked, 1) for c in histogram_counts
        ]
    else:
        histogram_percentages = [0.0 for _ in histogram_counts]

    marks_histogram = {
        "labels": histogram_labels,
        "percentages": histogram_percentages,
        "counts": histogram_counts,
        "total": total_marked,
    }

    return render_template(
        "exam_report.html",
        exam=exam,
        status_counts=status_counts,
        marks_stats=marks_stats,
        marks_histogram=marks_histogram,
        top_students=top_students,
    )
