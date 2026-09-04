"""
Gemini-powered structured extraction layer.

Responsibilities:
- Takes raw unstructured document text
- Returns a strict DocumentExtraction Pydantic model
- Forces JSON output via system prompt
- Preserves exact source citations
- Handles API failures gracefully (returns low-confidence extraction)

Gemini ONLY extracts facts. It does NOT decide outcomes.
"""

import os
import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from src.schema import (
    DocumentExtraction,
    Precision,
    ClaimType,
    DamageArea,
    DamageSeverity,
)

logger = logging.getLogger(__name__)

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

MODEL = "gemini-2.0-flash"

EXTRACTION_SYSTEM_PROMPT = """You are a precise insurance document fact extractor.
Given a document, extract structured facts into JSON format.

RULES:
1. Extract ONLY information explicitly stated in the document.
2. Do NOT infer, assume, or fabricate any information.
3. For every extracted value, provide the EXACT text from the document as citation.
4. If a field is not mentioned, set it to null.
5. Use EXACT precision when a specific value is given (e.g., "10:00 AM").
6. Use APPROXIMATE precision when vague language is used (e.g., "around 10 PM").
7. Use UNKNOWN precision when you cannot determine the precision.
8. For dates, output in YYYY-MM-DD format.
9. For times, output in HH:MM 24-hour format.
10. For damage_area, classify as: FRONT, REAR, LEFT, RIGHT, MULTIPLE, TOTAL_LOSS, UNKNOWN, or NONE.
11. For damage_severity, classify as: MINOR, MODERATE, MAJOR, TOTAL_LOSS, or UNKNOWN.
12. For claim_type, classify as: ACCIDENT, THEFT, MALICIOUS_DAMAGE, or OTHER.
13. Normalize driver names to lowercase.
14. For estimated_cost, extract the numeric value only (e.g., 1250.00).
15. Set confidence between 0.0 and 1.0 based on how clearly the information was stated.

OUTPUT FORMAT: Return ONLY valid JSON matching this schema:
{
    "source_document": "<document type>",
    "incident_date": "<YYYY-MM-DD or null>",
    "incident_date_original": "<exact text or null>",
    "incident_date_precision": "<EXACT|APPROXIMATE|UNKNOWN>",
    "incident_time": "<HH:MM or null>",
    "incident_time_original": "<exact text or null>",
    "incident_time_precision": "<EXACT|APPROXIMATE|UNKNOWN>",
    "claim_type": "<ACCIDENT|THEFT|MALICIOUS_DAMAGE|OTHER or null>",
    "driver_name": "<lowercase name or null>",
    "driver_name_original": "<exact text or null>",
    "damage_area": "<FRONT|REAR|LEFT|RIGHT|MULTIPLE|TOTAL_LOSS|UNKNOWN|NONE>",
    "damage_area_original": "<exact text or null>",
    "damage_severity": "<MINOR|MODERATE|MAJOR|TOTAL_LOSS|UNKNOWN>",
    "estimated_cost": <number or null>,
    "estimated_cost_original": "<exact text or null>",
    "vehicle_description": "<text or null>",
    "incident_summary": "<brief summary or null>",
    "report_date": "<YYYY-MM-DD or null>",
    "report_date_original": "<exact text or null>",
    "confidence": <0.0 to 1.0>
}"""


def extract_document(document_text: str, document_type: str) -> DocumentExtraction:
    """
    Extract structured facts from raw document text using Gemini.

    Args:
        document_text: The raw text of the document.
        document_type: One of 'claim_form', 'customer_description', 'fir', 'repair_estimate'.

    Returns:
        DocumentExtraction with all available fields populated.
    """
    if not document_text or not document_text.strip():
        logger.warning(f"Empty document text for {document_type}")
        return DocumentExtraction(
            source_document=document_type,
            confidence=0.0,
        )

    user_prompt = f"""Document type: {document_type}

Document text:
\"\"\"
{document_text.strip()}
\"\"\"

Extract all facts from this document into the specified JSON format."""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )

        raw_text = response.text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines)

        parsed = json.loads(raw_text)

        # Force the source_document field
        parsed["source_document"] = document_type

        extraction = DocumentExtraction(**parsed)
        return extraction

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {document_type}: {e}")
        return DocumentExtraction(
            source_document=document_type,
            confidence=0.0,
        )
    except Exception as e:
        logger.error(f"Gemini extraction failed for {document_type}: {e}")
        return DocumentExtraction(
            source_document=document_type,
            confidence=0.0,
        )


def extract_all_documents(
    claim_form_text: str,
    description_text: str,
    third_doc_text: str,
    third_doc_type: str,
) -> list[DocumentExtraction]:
    """
    Extract structured facts from all three input documents.

    Args:
        claim_form_text: Raw claim form text.
        description_text: Raw customer incident description.
        third_doc_text: Raw FIR or Repair Estimate text.
        third_doc_type: Either 'fir' or 'repair_estimate'.

    Returns:
        List of 3 DocumentExtraction models.
    """
    extractions = []

    extractions.append(extract_document(claim_form_text, "claim_form"))
    extractions.append(extract_document(description_text, "customer_description"))
    extractions.append(extract_document(third_doc_text, third_doc_type))

    return extractions
