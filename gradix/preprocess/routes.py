"""Text preprocessing and helpers.

This module also provides utilities to split OCR output into
numbered question/answer segments, which are used for per-question
evaluation and review in the web UI.
"""

import logging
import os
import re
from typing import Optional
from http import HTTPStatus

from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required

from ..extensions import db
from ..models import ExtractedText, UserRole
from ..rbac import role_required


preprocess_bp = Blueprint("preprocess", __name__, url_prefix="/")

logger = logging.getLogger(__name__)

_grammar_tokenizer = None
_grammar_model = None
_grammar_model_id: Optional[str] = None
_grammar_device: Optional[str] = None


def _hf_grammar_correct(text: str) -> Optional[str]:
    """Use a local Hugging Face text model to grammar-correct text.

    Loads a seq2seq model from the Hugging Face Hub via
    ``transformers`` and runs it locally. If a GPU is available,
    the model will run on CUDA for faster inference.

    The model ID can be overridden via ``HF_GRAMMAR_MODEL``; by
    default we use a T5-based grammar-correction model.
    """

    global _grammar_tokenizer, _grammar_model, _grammar_model_id, _grammar_device

    try:  # pragma: no cover - heavy optional dependency
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
    except Exception as exc:
        logger.info("Transformers/torch not available for grammar correction: %s", exc)
        return None

    model_id = os.environ.get(
        "HF_GRAMMAR_MODEL",
        "vennify/t5-base-grammar-correction",
    )

    try:
        if _grammar_tokenizer is None or _grammar_model is None or _grammar_model_id != model_id:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading grammar model: %s on %s", model_id, device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model.to(device)

            _grammar_tokenizer = tokenizer
            _grammar_model = model
            _grammar_model_id = model_id
            _grammar_device = device

        tokenizer = _grammar_tokenizer
        model = _grammar_model
        device = _grammar_device or "cpu"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Allow enough tokens so the model can rewrite the
            # entire passage instead of cutting it off too early.
            input_len = int(inputs["input_ids"].shape[1])
            max_new = max(96, min(384, input_len + 64))
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
            )

        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return corrected or None
    except Exception as exc:  # pragma: no cover - runtime errors
        logger.warning("Local HF grammar correction failed: %s", exc)
        return None


def preprocess_text(raw_text: str) -> str:
    """Normalize OCR text and correct spelling/grammar with Hugging Face.

    Order of operations:
    1. Try Hugging Face Inference API grammar correction.
    2. If that is unavailable or fails, fall back to LanguageTool
       via ``language_tool_python``.
    3. On any error, return the raw text unchanged so the pipeline
       still succeeds.
    """

    if not raw_text:
        return ""

    # First preference: Hugging Face Inference API
    corrected = _hf_grammar_correct(raw_text)
    if corrected:
        return corrected

    # Fallback: LanguageTool public API if installed
    try:  # pragma: no cover - external service / heavy dependency
        import language_tool_python  # type: ignore

        tool = language_tool_python.LanguageToolPublicAPI("en-US")
        return tool.correct(raw_text)
    except Exception as exc:
        logger.info("LanguageTool correction unavailable or failed: %s", exc)
        return raw_text


def split_numbered_answers(text: str):
    """Split OCR text into (question_no, answer_text) segments.

    Assumes answers are numbered in the OCR output, e.g.::

        1. First answer text...
        2) Second answer text...

    If no numbering is detected, the whole text is treated as a
    single answer for question 1.
    """

    if not text or not text.strip():
        return []

    pattern = re.compile(r"(?m)^\s*(\d+)[\).\s]+")
    matches = list(pattern.finditer(text))

    if not matches:
        return [(1, text.strip())]

    segments = []
    for i, match in enumerate(matches):
        q_no = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        answer = text[start:end].strip()
        if answer:
            segments.append((q_no, answer))

    return segments


@preprocess_bp.post("preprocess/<int:sheet_id>")
@jwt_required()
@role_required({UserRole.TEACHER})
def preprocess(sheet_id: int):
    extracted = ExtractedText.query.filter_by(sheet_id=sheet_id).first()
    if extracted is None:
        return (
            jsonify({"message": "No extracted text found for this sheet. Run OCR first."}),
            HTTPStatus.BAD_REQUEST,
        )

    extracted.cleaned_text = preprocess_text(extracted.raw_text)
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
        HTTPStatus.OK,
    )
