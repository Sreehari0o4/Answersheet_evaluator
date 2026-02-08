import logging
import os
import subprocess
import tempfile
from http import HTTPStatus

from flask import Blueprint, current_app, jsonify
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import AnswerSheet, ExtractedText, UserRole
from ..rbac import role_required


ocr_bp = Blueprint("ocr", __name__, url_prefix="/ocr")

logger = logging.getLogger(__name__)

_trocr_processor = None
_trocr_model = None
_trocr_model_id: str | None = None


def _preprocess_image(file_path: str) -> tuple[str, bool]:
    """Lightly preprocess the image to help OCR.

    Steps (best-effort, all inside try/except):
    - Convert to grayscale
    - Apply blur + Otsu thresholding for better contrast
    - Optionally upscale smaller images

    Returns (path_to_use, is_temporary_file).
    On any error, returns the original path and False.
    """

    abs_path = os.path.abspath(file_path)

    try:  # pragma: no cover - depends on OpenCV being available
        import cv2  # type: ignore

        image = cv2.imread(abs_path)
        if image is None:
            return abs_path, False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        height, width = thresh.shape[:2]
        if height < 1000:
            scale = 1000.0 / float(height)
            new_size = (int(width * scale), 1000)
            thresh = cv2.resize(thresh, new_size, interpolation=cv2.INTER_LINEAR)

        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp_path, thresh)

        return tmp_path, True

    except Exception as exc:
        logger.info("Image preprocessing failed or OpenCV missing, using original image: %s", exc)
        return abs_path, False


def _run_hf_ocr(file_path: str) -> tuple[str, float] | tuple[None, float]:
    """Use a local Hugging Face TrOCR model via transformers.

    By default uses ``microsoft/trocr-small-handwritten``, which is
    suitable for handwriting. You can override the model by setting
    ``HF_OCR_MODEL`` in the environment.

    The model and processor are loaded lazily and cached at module
    level so they are only loaded once per process.
    """

    global _trocr_processor, _trocr_model, _trocr_model_id

    try:  # pragma: no cover - heavy optional dependency
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        logger.exception("Transformers/Pillow not available for Hugging Face OCR: %s", exc)
        return None, 0.0

    model_id = os.environ.get("HF_OCR_MODEL", "microsoft/trocr-small-handwritten")

    try:
        if _trocr_processor is None or _trocr_model is None or _trocr_model_id != model_id:
            logger.info("Loading TrOCR model: %s", model_id)
            _trocr_processor = TrOCRProcessor.from_pretrained(model_id)
            _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_id)
            _trocr_model_id = model_id

        abs_path = os.path.abspath(file_path)
        image = Image.open(abs_path).convert("RGB")

        pixel_values = _trocr_processor(images=image, return_tensors="pt").pixel_values
        generated_ids = _trocr_model.generate(pixel_values)
        generated_texts = _trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        if not generated_texts:
            return None, 0.0

        text = (generated_texts[0] or "").strip()
        if not text:
            return None, 0.0

        return text, 0.95

    except Exception as exc:
        logger.exception("Local Hugging Face TrOCR inference failed: %s", exc)
        return None, 0.0


def _run_easyocr(file_path: str) -> tuple[str, float] | tuple[None, float]:
    """Try to extract text using EasyOCR.

    This tends to work better for handwriting than plain Tesseract.
    Returns (text, confidence) or (None, 0.0) if EasyOCR is not usable.
    """

    try:  # pragma: no cover - depends on optional heavy dependency
        import easyocr  # type: ignore

        # GPU=False keeps it simple and works on most machines.
        reader = easyocr.Reader(["en"], gpu=False)
        # detail=0 -> list of strings; paragraph=True -> join nearby words.
        result = reader.readtext(file_path, detail=0, paragraph=True)
        text = "\n".join(line.strip() for line in result if line and line.strip())

        if not text:
            return None, 0.0

        # EasyOCR usually gives better quality for handwriting.
        return text, 0.9
    except Exception as exc:
        logger.info("EasyOCR not available or failed, falling back to Tesseract: %s", exc)
        return None, 0.0


def _run_tesseract(file_path: str) -> tuple[str, float]:
    """Run OCR on the given file by invoking the Tesseract binary.

    - Expects Tesseract to be installed (for example in
      C:\\Program Files\\Tesseract-OCR\\tesseract.exe).
    - If the environment variable ``TESSERACT_CMD`` is set, its value will
      be used as the Tesseract executable path; otherwise ``tesseract`` is
      used and must be on PATH.
    """

    tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")

    # tesseract <image> stdout -l eng --psm 6
    result = subprocess.run(
        [tesseract_cmd, os.path.abspath(file_path), "stdout", "-l", "eng", "--psm", "6"],
        capture_output=True,
        text=True,
        check=True,
    )

    extracted_text = (result.stdout or "").strip()
    if not extracted_text:
        raise RuntimeError("Tesseract returned empty text.")

    # Confidence is heuristic here; we call it medium.
    return extracted_text, 0.7


def run_ocr(file_path: str) -> tuple[str, float]:
    """High-level OCR wrapper.

    Local backends only (used as fallback if Scripily is disabled or
    unavailable):

    1. Preprocess the image (grayscale + threshold + optional upscale).
    2. Try EasyOCR (works well for handwriting).
    3. As a last resort, fall back to Tesseract if available.
    """
    processed_path, is_temp = _preprocess_image(file_path)

    try:
        # First, try EasyOCR (lighter than TrOCR and avoids large model
        # downloads).
        text, confidence = _run_easyocr(processed_path)
        if text:
            return text, confidence

        # Finally, attempt plain Tesseract as a last fallback
        try:
            return _run_tesseract(processed_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tesseract OCR failed as well: %s", exc)
            return "", 0.0

    finally:
        if is_temp and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except OSError:
                logger.info("Could not remove temporary preprocessed image: %s", processed_path)


@ocr_bp.post("/run/<int:sheet_id>")
@jwt_required()
@role_required({UserRole.TEACHER})
def ocr_run(sheet_id: int):
    sheet = AnswerSheet.query.get(sheet_id)
    if sheet is None:
        return (
            jsonify({"message": "AnswerSheet not found."}),
            HTTPStatus.NOT_FOUND,
        )

    # Build absolute file path for local OCR backends
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    filename = os.path.basename(sheet.file_path)
    abs_path = os.path.join(upload_folder, filename)

    raw_text = ""
    confidence = 0.0

    # 1) Prefer Scripily cloud OCR if enabled and properly configured.
    use_scripily = os.environ.get("USE_SCRIPILY", "").lower() in {"1", "true", "yes"}
    if use_scripily:
        try:  # pragma: no cover - depends on external API
            from scripily_client import ScripilyConfigError, extract_text as scripily_extract

            public_base = os.environ.get("SCRIPILY_PUBLIC_BASE_URL", "").rstrip("/")
            if not public_base:
                raise ScripilyConfigError(
                    "SCRIPILY_PUBLIC_BASE_URL not set; cannot build public image URL for Scripily."
                )

            # sheet.file_path is stored as a relative path like "uploads/<filename>".
            rel_path = sheet.file_path.replace("\\", "/").lstrip("/")
            image_url = f"{public_base}/{rel_path}"
            logger.info("Attempting Scripily OCR for URL: %s", image_url)
            s_text = scripily_extract(image_url)
            if s_text:
                raw_text = s_text
                confidence = 0.98
            else:
                raise RuntimeError("Empty text from Scripily")
        except ScripilyConfigError as exc:
            logger.warning("Scripily misconfigured, falling back to local OCR: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scripily OCR failed, falling back to local OCR: %s", exc)

    # 2) Local OCR fallback if Scripily is disabled or failed.
    if not raw_text:
        raw_text, confidence = run_ocr(abs_path)

    cleaned_text = raw_text.strip().lower()

    # Upsert ExtractedText for this sheet
    extracted = ExtractedText.query.filter_by(sheet_id=sheet.sheet_id).first()
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

    return (
        jsonify(
            {
                "text_id": extracted.text_id,
                "sheet_id": extracted.sheet_id,
                "raw_text": extracted.raw_text,
                "cleaned_text": extracted.cleaned_text,
                "extraction_confidence": extracted.extraction_confidence,
            }
        ),
        HTTPStatus.CREATED,
    )
