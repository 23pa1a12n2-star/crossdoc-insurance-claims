"""
CrossDoc: Insurance Claims Evidence Review Assistant
====================================================

Main application entry point.
Start with: python app.py

Serves the full application (backend + frontend) on port 8000.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Load env EARLY before any local modules init their API clients
load_dotenv()

from src.schema import EvidencePackage, ThirdDocType
from src.extractor import extract_all_documents
from src.crossdoc import run_crossdoc
from src.policy import evaluate_claim, semantic_policy_search
from src.synthesizer import synthesize_report
from src.doc_reader import extract_text_from_file

# Logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("crossdoc")

# FastAPI app
app = FastAPI(title="CrossDoc - Insurance Claims Evidence Review")

# Templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Serve static files if needed
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main UI."""
    # Load demo cases for quick testing
    cases_dir = BASE_DIR / "data" / "cases"
    demo_cases = {}
    if cases_dir.exists():
        for case_file in sorted(cases_dir.glob("*.txt")):
            with open(case_file, "r", encoding="utf-8") as f:
                demo_cases[case_file.stem] = f.read()

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "demo_cases": json.dumps(demo_cases)},
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "crossdoc"}


@app.post("/api/review", response_class=JSONResponse)
async def review_claim(
    claim_form: str = Form(...),
    customer_description: str = Form(...),
    third_doc: str = Form(...),
    third_doc_type: str = Form("repair_estimate"),
):
    """
    Main claim review endpoint.

    Accepts three document texts and runs the full CrossDoc pipeline:
    1. Gemini extraction
    2. Python normalization + CrossDoc comparison
    3. Policy evaluation
    4. Decision routing
    5. Gemini synthesis
    """
    try:
        # Validate third_doc_type
        if third_doc_type not in ("fir", "repair_estimate"):
            third_doc_type = "repair_estimate"

        logger.info("Starting claim review...")

        # Step 1: Extract structured facts from all documents
        logger.info("Step 1: Extracting facts with Gemini...")
        extractions = extract_all_documents(
            claim_form_text=claim_form,
            description_text=customer_description,
            third_doc_text=third_doc,
            third_doc_type=third_doc_type,
        )

        # Step 2: Run CrossDoc comparison matrix
        logger.info("Step 2: Running CrossDoc comparison...")
        crossdoc_result = run_crossdoc(extractions)

        # Step 3: Evaluate against policy rules
        logger.info("Step 3: Evaluating policy rules...")

        # Semantic policy search using incident description
        semantic_findings = []
        try:
            from google import genai as genai_embed
            embed_client = genai_embed.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

            # Build query from incident summaries
            query_parts = []
            for ext in extractions:
                if ext.incident_summary:
                    query_parts.append(ext.incident_summary)

            if query_parts:
                query_text = " ".join(query_parts)
                embed_result = embed_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=query_text,
                )
                search_results = semantic_policy_search(
                    embed_result.embeddings[0].values,
                    top_k=2,
                )
                logger.info(f"Semantic search returned {len(search_results)} results")
                # Note: Semantic results are for citation/evidence only.
                # The keyword-based exclusion check in policy.py handles actual REJECT decisions.
        except Exception as e:
            logger.warning(f"Semantic policy search failed (non-critical): {e}")

        # Build evidence package with decision
        evidence = evaluate_claim(
            extractions=extractions,
            crossdoc=crossdoc_result,
            third_doc_type=third_doc_type,
        )

        # Step 4: Generate human-readable report
        logger.info("Step 4: Generating synthesis report...")
        evidence.synthesis = synthesize_report(evidence)

        logger.info(f"Review complete. Decision: {evidence.decision.value}")

        # Build response
        response_data = {
            "decision": evidence.decision.value,
            "reasons": evidence.decision_reasons,
            "synthesis": evidence.synthesis,
            "crossdoc": {
                "summary": evidence.crossdoc.summary,
                "has_contradiction": evidence.crossdoc.has_contradiction,
                "has_missing": evidence.crossdoc.has_missing,
                "has_ambiguous": evidence.crossdoc.has_ambiguous,
                "comparisons": [
                    {
                        "field": c.field_name,
                        "status": c.status.value,
                        "values": c.values,
                        "original_values": c.original_values,
                        "citations": c.citations,
                        "explanation": c.explanation,
                    }
                    for c in evidence.crossdoc.comparisons
                ],
            },
            "policy_findings": [
                {
                    "rule": f.rule_name,
                    "triggered": f.triggered,
                    "detail": f.detail,
                    "clause": f.policy_clause,
                    "reference": f.policy_reference,
                }
                for f in evidence.policy_findings
            ],
            "extractions": [
                {
                    "source": ext.source_document,
                    "incident_date": ext.incident_date,
                    "incident_time": ext.incident_time,
                    "claim_type": ext.claim_type.value if ext.claim_type else None,
                    "driver_name": ext.driver_name,
                    "damage_area": ext.damage_area.value if ext.damage_area else None,
                    "damage_severity": ext.damage_severity.value if ext.damage_severity else None,
                    "estimated_cost": ext.estimated_cost,
                    "confidence": ext.confidence,
                }
                for ext in evidence.extractions
            ],
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Review failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "decision": "ESCALATE",
                "reasons": [f"System error occurred: {str(e)}. Manual review required."],
                "synthesis": "The automated review system encountered an error. This claim has been escalated for manual investigation.",
                "crossdoc": {"summary": "Analysis could not be completed.", "comparisons": []},
                "policy_findings": [],
                "extractions": [],
            },
        )


@app.post("/api/upload", response_class=JSONResponse)
async def upload_review(
    third_doc_type: str = Form("repair_estimate"),
    claim_form_text: Optional[str] = Form(None),
    customer_desc_text: Optional[str] = Form(None),
    third_doc_text: Optional[str] = Form(None),
    claim_form_file: Optional[UploadFile] = File(None),
    customer_desc_file: Optional[UploadFile] = File(None),
    third_doc_file: Optional[UploadFile] = File(None),
):
    """
    Upload-aware claim review endpoint.

    For each of the 3 documents, accepts EITHER pasted text OR an uploaded file.
    Uploaded files (PDF/PNG/JPG/TXT) are converted to text, then the
    SAME existing CrossDoc pipeline is executed.
    """
    try:
        # Resolve each document: prefer file upload over pasted text
        async def resolve_doc(text_val, file_val, label):
            if file_val and file_val.filename:
                file_bytes = await file_val.read()
                if not file_bytes:
                    raise ValueError(f"{label}: Uploaded file is empty.")
                return extract_text_from_file(
                    file_bytes=file_bytes,
                    filename=file_val.filename,
                    content_type=file_val.content_type or "",
                )
            elif text_val and text_val.strip():
                return text_val.strip()
            else:
                raise ValueError(f"{label}: No text pasted and no file uploaded.")

        claim_form = await resolve_doc(claim_form_text, claim_form_file, "Claim Form")
        customer_desc = await resolve_doc(customer_desc_text, customer_desc_file, "Customer Description")
        third_doc = await resolve_doc(third_doc_text, third_doc_file, "Supporting Document")

        if third_doc_type not in ("fir", "repair_estimate"):
            third_doc_type = "repair_estimate"

        logger.info("Starting uploaded-document claim review...")

        # Step 1: Extract structured facts
        logger.info("Step 1: Extracting facts with Gemini...")
        extractions = extract_all_documents(
            claim_form_text=claim_form,
            description_text=customer_desc,
            third_doc_text=third_doc,
            third_doc_type=third_doc_type,
        )

        # Step 2: CrossDoc comparison
        logger.info("Step 2: Running CrossDoc comparison...")
        crossdoc_result = run_crossdoc(extractions)

        # Step 3: Policy evaluation
        logger.info("Step 3: Evaluating policy rules...")
        semantic_findings = []
        try:
            from google import genai as genai_embed
            embed_client = genai_embed.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
            query_parts = [ext.incident_summary for ext in extractions if ext.incident_summary]
            if query_parts:
                embed_result = embed_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=" ".join(query_parts),
                )
                search_results = semantic_policy_search(
                    embed_result.embeddings[0].values, top_k=2,
                )
                logger.info(f"Semantic search returned {len(search_results)} results")
        except Exception as e:
            logger.warning(f"Semantic policy search failed (non-critical): {e}")

        evidence = evaluate_claim(
            extractions=extractions,
            crossdoc=crossdoc_result,
            third_doc_type=third_doc_type,
        )

        # Step 4: Synthesis
        logger.info("Step 4: Generating synthesis report...")
        evidence.synthesis = synthesize_report(evidence)
        logger.info(f"Review complete. Decision: {evidence.decision.value}")

        # Build response (same format as /api/review)
        response_data = {
            "decision": evidence.decision.value,
            "reasons": evidence.decision_reasons,
            "synthesis": evidence.synthesis,
            "crossdoc": {
                "summary": evidence.crossdoc.summary,
                "has_contradiction": evidence.crossdoc.has_contradiction,
                "has_missing": evidence.crossdoc.has_missing,
                "has_ambiguous": evidence.crossdoc.has_ambiguous,
                "comparisons": [
                    {
                        "field": c.field_name,
                        "status": c.status.value,
                        "values": c.values,
                        "original_values": c.original_values,
                        "citations": c.citations,
                        "explanation": c.explanation,
                    }
                    for c in evidence.crossdoc.comparisons
                ],
            },
            "policy_findings": [
                {
                    "rule": f.rule_name,
                    "triggered": f.triggered,
                    "detail": f.detail,
                    "clause": f.policy_clause,
                    "reference": f.policy_reference,
                }
                for f in evidence.policy_findings
            ],
            "extractions": [
                {
                    "source": ext.source_document,
                    "incident_date": ext.incident_date,
                    "incident_time": ext.incident_time,
                    "claim_type": ext.claim_type.value if ext.claim_type else None,
                    "driver_name": ext.driver_name,
                    "damage_area": ext.damage_area.value if ext.damage_area else None,
                    "damage_severity": ext.damage_severity.value if ext.damage_severity else None,
                    "estimated_cost": ext.estimated_cost,
                    "confidence": ext.confidence,
                }
                for ext in evidence.extractions
            ],
        }

        return JSONResponse(content=response_data)

    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={"error": str(ve)},
        )
    except Exception as e:
        logger.error(f"Upload review failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "decision": "ESCALATE",
                "reasons": [f"System error occurred: {str(e)}. Manual review required."],
                "synthesis": "The automated review system encountered an error. This claim has been escalated for manual investigation.",
                "crossdoc": {"summary": "Analysis could not be completed.", "comparisons": []},
                "policy_findings": [],
                "extractions": [],
            },
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
