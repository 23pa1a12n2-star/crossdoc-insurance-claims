"""
Document Reader Utility
========================

Converts uploaded files (PDF, PNG, JPG, TXT) into plain text
for consumption by the existing CrossDoc extraction pipeline.

- TXT: read directly
- PDF: extract text via PyMuPDF; fallback to Gemini vision for scanned PDFs
- Images (PNG/JPG/JPEG): extract text via Gemini vision
"""

import os
import io
import logging
from typing import Optional

import fitz  # PyMuPDF

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Gemini client for vision-based OCR
_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
_VISION_MODEL = "gemini-3.5-flash-lite"

# Supported MIME types
SUPPORTED_TYPES = {
    "text/plain": "txt",
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpg",
}

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def extract_text_from_file(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> str:
    """
    Extract text from an uploaded file.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename.
        content_type: MIME type of the file.

    Returns:
        Extracted text string.

    Raises:
        ValueError: If the file type is unsupported, or extraction fails.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_SIZE_MB}MB size limit.")

    # Normalize content type
    ct = content_type.lower().split(";")[0].strip()

    # Fallback: detect by extension if content_type is generic
    if ct == "application/octet-stream" or ct not in SUPPORTED_TYPES:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        ext_map = {"txt": "text/plain", "pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
        if ext in ext_map:
            ct = ext_map[ext]
        else:
            raise ValueError(f"Unsupported file type: {content_type} (filename: {filename}). Supported: TXT, PDF, PNG, JPG/JPEG.")

    if ct == "text/plain":
        return _read_txt(file_bytes)
    elif ct == "application/pdf":
        return _read_pdf(file_bytes, filename)
    elif ct in ("image/png", "image/jpeg"):
        return _read_image(file_bytes, ct)
    else:
        raise ValueError(f"Unsupported file type: {ct}")


def _read_txt(file_bytes: bytes) -> str:
    """Read plain text, trying UTF-8 then Latin-1."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")
    text = text.strip()
    if not text:
        raise ValueError("TXT file is empty.")
    return text


def _read_pdf(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF. Falls back to Gemini vision for scanned PDFs."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{filename}': {e}")

    text_parts = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(page_text.strip())
    doc.close()

    full_text = "\n\n".join(text_parts).strip()

    # If we got meaningful text, return it
    if len(full_text) > 30:
        logger.info(f"PDF '{filename}': extracted {len(full_text)} chars via PyMuPDF")
        return full_text

    # Scanned/image PDF fallback: re-open and render pages as images, send to Gemini
    logger.info(f"PDF '{filename}': insufficient text ({len(full_text)} chars), attempting Gemini vision OCR...")
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        ocr_parts = []
        for page_num, page in enumerate(doc):
            if page_num >= 5:  # Limit to first 5 pages
                break
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            page_text = _ocr_with_gemini(img_bytes, "image/png")
            if page_text:
                ocr_parts.append(page_text)
        doc.close()

        if ocr_parts:
            result = "\n\n".join(ocr_parts).strip()
            logger.info(f"PDF '{filename}': Gemini OCR extracted {len(result)} chars")
            return result
    except Exception as e:
        logger.warning(f"Gemini OCR fallback failed for PDF '{filename}': {e}")

    if full_text:
        return full_text

    raise ValueError(f"Could not extract text from PDF '{filename}'. The file may be empty or contain only images that could not be read.")


def _read_image(file_bytes: bytes, mime_type: str) -> str:
    """Extract text from an image using Gemini vision."""
    text = _ocr_with_gemini(file_bytes, mime_type)
    if not text or len(text.strip()) < 5:
        raise ValueError("Could not extract meaningful text from the image. The image may be blank or unreadable.")
    return text.strip()


def _ocr_with_gemini(image_bytes: bytes, mime_type: str) -> Optional[str]:
    """
    Send an image to Gemini and ask it to extract all text.
    Returns the extracted text or None on failure.
    """
    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = _client.models.generate_content(
            model=_VISION_MODEL,
            contents=[
                "Extract ALL text from this document image exactly as it appears. "
                "Preserve the original structure, headings, labels, dates, names, and numbers. "
                "Do not summarize or interpret. Only output the raw text content.",
                image_part,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini OCR failed: {e}")
        return None
