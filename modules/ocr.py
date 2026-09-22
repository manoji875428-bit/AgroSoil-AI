from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps


FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "N": (r"\bnitrogen\b", r"\bN\b"),
    "P": (r"\bphosphorus\b", r"\bP\b"),
    "K": (r"\bpotassium\b", r"\bK\b"),
    "pH": (r"\bpH\b", r"\bph\b"),
    "Organic_Carbon": (r"\borganic\s*carbon\b", r"\borganic\s*C\b", r"\bOC\b"),
}
NUMBER_PATTERN = r"([-+]?\d+(?:[.,]\d+)?)"


def _preprocess_image(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    resized = grayscale.resize((grayscale.width * 2, grayscale.height * 2)) if grayscale.width < 1600 else grayscale
    enhanced = ImageEnhance.Contrast(resized).enhance(1.6)
    return ImageOps.autocontrast(enhanced)


def _tesseract_status() -> tuple[Any | None, str | None]:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return pytesseract, None
    except ImportError:
        return None, "The Python OCR package is unavailable. Install pytesseract and the Tesseract executable."
    except Exception as error:
        return None, f"Tesseract OCR is unavailable: {error}"


def extract_text_from_image(image_data: bytes | Image.Image) -> dict[str, Any]:
    """Run local Tesseract OCR when installed and return text plus confidence."""
    pytesseract, error = _tesseract_status()
    if error or pytesseract is None:
        return {"text": "", "confidence": None, "error": error}
    try:
        image = image_data if isinstance(image_data, Image.Image) else Image.open(io.BytesIO(image_data))
        prepared_image = _preprocess_image(image.convert("RGB"))
        data = pytesseract.image_to_data(prepared_image, output_type=pytesseract.Output.DICT)
        confidences = [float(value) for value in data.get("conf", []) if str(value).strip() not in {"", "-1"}]
        text = pytesseract.image_to_string(prepared_image)
        return {"text": text, "confidence": sum(confidences) / len(confidences) if confidences else None, "error": None}
    except Exception as error:
        return {"text": "", "confidence": None, "error": f"The uploaded image could not be read: {error}"}


def extract_text_from_pdf(pdf_data: bytes) -> dict[str, Any]:
    """Extract embedded PDF text locally; scanned PDFs need a PDF-to-image tool."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return {"text": "", "confidence": None, "error": "PDF text extraction is unavailable. Install pypdf, or upload a report image with Tesseract configured."}
    try:
        reader = PdfReader(io.BytesIO(pdf_data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            return {"text": "", "confidence": None, "error": "No embedded text was found. This may be a scanned PDF; install a PDF-to-image tool and Tesseract for scanned reports."}
        return {"text": text, "confidence": None, "error": None}
    except Exception as error:
        return {"text": "", "confidence": None, "error": f"The PDF could not be read: {error}"}


def _number_after_label(line: str, label_pattern: str) -> float | None:
    match = re.search(rf"{label_pattern}\s*(?:\([^)]*\))?\s*[:=\-]?\s*{NUMBER_PATTERN}", line, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def extract_soil_values(text: str) -> dict[str, float | None]:
    """Extract only numbers directly associated with known nutrient labels."""
    values: dict[str, float | None] = {field: None for field in FIELD_PATTERNS}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for field, patterns in FIELD_PATTERNS.items():
        for line in lines:
            for pattern in patterns:
                value = _number_after_label(line, pattern)
                if value is not None:
                    values[field] = value
                    break
            if values[field] is not None:
                break
    return values


def validate_extracted_values(values: dict[str, Any]) -> dict[str, Any]:
    required = list(FIELD_PATTERNS)
    missing = [field for field in required if values.get(field) is None]
    invalid: list[str] = []
    normalized: dict[str, float] = {}
    for field in required:
        if values.get(field) is None:
            continue
        try:
            number = float(values[field])
        except (TypeError, ValueError):
            invalid.append(field)
            continue
        if field != "pH" and number < 0:
            invalid.append(field)
        elif field == "pH" and not 0 <= number <= 14:
            invalid.append(field)
        else:
            normalized[field] = number
    return {"valid": not missing and not invalid, "values": normalized, "missing": missing, "invalid": invalid}


def process_soil_report(file_name: str, file_data: bytes) -> dict[str, Any]:
    """Route an uploaded report to image OCR or embedded PDF text extraction."""
    suffix = Path(file_name).suffix.lower()
    if not file_data:
        return {"text": "", "values": {}, "confidence": None, "error": "The uploaded report is empty."}
    if suffix in {".png", ".jpg", ".jpeg"}:
        extracted = extract_text_from_image(file_data)
    elif suffix == ".pdf":
        extracted = extract_text_from_pdf(file_data)
    else:
        return {"text": "", "values": {}, "confidence": None, "error": "Unsupported file type. Upload a PNG, JPG, JPEG, or PDF report."}
    text = extracted.get("text", "")
    values = extract_soil_values(text) if text else {field: None for field in FIELD_PATTERNS}
    validation = validate_extracted_values(values)
    return {**extracted, "values": values, "validation": validation}
