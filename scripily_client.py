"""Minimal Scripily API client.

You need to log in to https://app.scripily.com, open the API
section, and check the docs for:

- The base URL and path of the extraction endpoint
- How the API key is passed (header name and format)
- The multipart/form-data field name for the file
- The JSON field that contains the extracted text

Then update the placeholders below accordingly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests


# Load .env from the project root (same folder as this file)
DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(DOTENV_PATH)

# Set these from your Scripily dashboard / docs
SCRIPILY_API_URL = os.environ.get("SCRIPILY_API_URL", "https://app.scripily.com/api/v1/extract")
# IMPORTANT: Do NOT commit your real key to git. Prefer setting it as an
# environment variable (SCRIPILY_API_KEY) in .env. The default here is empty.
SCRIPILY_API_KEY = os.environ.get("SCRIPILY_API_KEY", "")

# These names MUST match what the Scripily docs say
# For header: "Authorization: Bearer <API_KEY>"
AUTH_HEADER_NAME = "Authorization"  # header name only
AUTH_HEADER_VALUE_FORMAT = "Bearer {key}"  # how the value is built

TEXT_FIELD_NAME = "text"  # JSON field in response containing extracted text


class ScripilyConfigError(RuntimeError):
    pass


def _build_headers() -> dict[str, str]:
    if not SCRIPILY_API_KEY or "REPLACE" in SCRIPILY_API_KEY:
        raise ScripilyConfigError(
            "Set SCRIPILY_API_KEY (env var) or in scripily_client.py using the value from your Scripily account."
        )
    if "REPLACE" in AUTH_HEADER_NAME or "REPLACE" in AUTH_HEADER_VALUE_FORMAT:
        raise ScripilyConfigError(
            "Update AUTH_HEADER_NAME and AUTH_HEADER_VALUE_FORMAT in scripily_client.py according to the Scripily docs."
        )

    return {
        AUTH_HEADER_NAME: AUTH_HEADER_VALUE_FORMAT.format(key=SCRIPILY_API_KEY),
    }


def extract_text(image_url: str, timeout: float = 30.0) -> str:
    """Send an image URL to Scripily and return extracted text.

    The Scripily API expects JSON of the form::

        {"imageUrl": "https://...", "language": "en", "enhance": true}
    """

    if not SCRIPILY_API_URL:
        raise ScripilyConfigError(
            "Set SCRIPILY_API_URL (env var) or in scripily_client.py to the full Scripily extraction endpoint URL."
        )

    headers = _build_headers()

    payload: dict[str, Any] = {
        "imageUrl": image_url,
        "language": "en",
        "enhance": True,
    }

    resp = requests.post(
        SCRIPILY_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    resp.raise_for_status()
    data: Any = resp.json()

    if TEXT_FIELD_NAME not in data:
        raise RuntimeError(
            f"TEXT_FIELD_NAME='{TEXT_FIELD_NAME}' not in Scripily response; "
            "check the docs and update scripily_client.py accordingly."
        )

    text = str(data[TEXT_FIELD_NAME] or "").strip()
    return text


if __name__ == "__main__":  # manual test helper
    # Example: direct Google Drive "uc" link or any public image URL
    test_url = "https://drive.google.com/uc?export=view&id=19hvkBZBslqwkc4rimqhSJPKlivoTiXEs"
    print(f"Sending {test_url} to Scripily...")
    extracted = extract_text(test_url)
    print("\n--- Scripily extracted text ---\n")
    print(extracted)
