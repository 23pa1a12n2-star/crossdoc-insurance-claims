"""
Policy Engine and Decision Router.

Responsibilities:
- Evaluate hard deterministic policy rules (claim window, required docs)
- Perform semantic policy retrieval using pre-computed NumPy embeddings
- Route the final decision using strict precedence:
  REJECT > REQUEST_INFO > ESCALATE > APPROVE

Python owns the final decision. Gemini never decides outcomes.
"""

import os
import json
import logging
import numpy as np
from typing import List, Optional, Tuple

from src.schema import (
    DocumentExtraction,
    CrossDocResult,
    CrossDocStatus,
    PolicyFinding,
    EvidencePackage,
    Decision,
    ClaimType,
)

logger = logging.getLogger(__name__)

# =====================================================
# HARD POLICY RULES (Deterministic)
# =====================================================

MAX_CLAIM_WINDOW_DAYS = 30

# Required docs per claim type
REQUIRED_DOCS = {
    ClaimType.THEFT: "fir",
    ClaimType.MALICIOUS_DAMAGE: "fir",
}

# Policy exclusion keywords mapped to clauses
POLICY_EXCLUSION_MAP = {
    "commercial_use": {
        "keywords": ["delivery", "uber", "ola", "rideshare", "hire", "reward", "commercial", "package delivery"],
        "clause": "Section 4.1: Commercial Use Exclusion",
        "text": "Coverage is entirely invalidated if the insured vehicle is being used for hire, reward, ridesharing, package delivery, or any other commercial enterprise at the time of the incident.",
    },
    "racing": {
        "keywords": ["racetrack", "racing", "track day", "speed testing", "competitive driving", "circuit", "timed laps", "race"],
        "clause": "Section 4.2: Racing Exclusion",
        "text": "Coverage is entirely invalidated if the vehicle is operating on a racetrack, engaging in speed testing, or participating in organized competitive driving.",
    },
    "intoxication": {
        "keywords": ["drunk", "alcohol", "intoxicated", "substances", "impaired", "under the influence"],
        "clause": "Section 4.3: Intoxication Exclusion",
        "text": "The company is not liable for accidents occurring while the driver is under the influence of alcohol or restricted substances.",
    },
}

# =====================================================
# POLICY INDEX (NumPy-based semantic search)
# =====================================================

POLICY_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "policy_index.json"
)


def _load_policy_index() -> Optional[dict]:
    """Load pre-computed policy embeddings from JSON."""
    if not os.path.exists(POLICY_INDEX_PATH):
        logger.warning(f"Policy index not found at {POLICY_INDEX_PATH}")
        return None
    try:
        with open(POLICY_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load policy index: {e}")
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def semantic_policy_search(
    query_embedding: List[float],
    top_k: int = 2,
) -> List[dict]:
    """
    Search the pre-computed policy index for relevant clauses.

    Args:
        query_embedding: Embedding vector from gemini-embedding-001.
        top_k: Number of top results to return.

    Returns:
        List of dicts with keys: clause_id, text, score.
    """
    index = _load_policy_index()
    if not index or "chunks" not in index:
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)
    results = []

    for chunk in index["chunks"]:
        chunk_vec = np.array(chunk["embedding"], dtype=np.float32)
        score = _cosine_similarity(query_vec, chunk_vec)
        results.append({
            "clause_id": chunk.get("clause_id", ""),
            "text": chunk.get("text", ""),
            "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =====================================================
# DETERMINISTIC POLICY CHECKS
# =====================================================

def check_claim_window(extractions: List[DocumentExtraction]) -> PolicyFinding:
    """Check if the claim was filed within the allowed window."""
    from datetime import datetime

    incident_date = None
    report_date = None

    for ext in extractions:
        if ext.incident_date and not incident_date:
            try:
                incident_date = datetime.strptime(ext.incident_date, "%Y-%m-%d")
            except ValueError:
                pass
        if ext.report_date and not report_date:
            try:
                report_date = datetime.strptime(ext.report_date, "%Y-%m-%d")
            except ValueError:
                pass

    if not incident_date or not report_date:
        return PolicyFinding(
            rule_name="claim_window",
            triggered=False,
            detail="Unable to determine claim window: incident date or report date missing.",
            policy_reference="Section 2.1",
        )

    days_elapsed = (report_date - incident_date).days

    if days_elapsed > MAX_CLAIM_WINDOW_DAYS:
        return PolicyFinding(
            rule_name="claim_window",
            triggered=True,
            detail=f"Claim was reported {days_elapsed} days after the incident. Policy maximum is {MAX_CLAIM_WINDOW_DAYS} days.",
            policy_clause="All accident and damage claims must be formally reported to the company within thirty (30) days of the incident occurrence date.",
            policy_reference="Section 2.1",
        )

    return PolicyFinding(
        rule_name="claim_window",
        triggered=False,
        detail=f"Claim reported within {days_elapsed} days (within {MAX_CLAIM_WINDOW_DAYS}-day limit).",
        policy_reference="Section 2.1",
    )


def check_required_documents(
    extractions: List[DocumentExtraction],
    third_doc_type: str,
) -> PolicyFinding:
    """Check if the required third document type matches the claim type."""
    claim_type = None
    for ext in extractions:
        if ext.claim_type:
            claim_type = ext.claim_type
            break

    if not claim_type:
        return PolicyFinding(
            rule_name="required_documents",
            triggered=False,
            detail="Claim type could not be determined from documents.",
        )

    required = REQUIRED_DOCS.get(claim_type)
    if not required:
        return PolicyFinding(
            rule_name="required_documents",
            triggered=False,
            detail=f"No specific document requirement for claim type: {claim_type.value}.",
        )

    if third_doc_type != required:
        if claim_type == ClaimType.THEFT:
            clause_text = "Claims specifically relating to Theft or Malicious Damage strictly require a First Information Report (FIR) from the police jurisdiction where the incident occurred."
            ref = "Section 3.2"
        else:
            clause_text = "Claims relating to Accidental Collision require a Repair Estimate from a certified garage."
            ref = "Section 3.3"

        return PolicyFinding(
            rule_name="required_documents",
            triggered=True,
            detail=f"Claim type is {claim_type.value}, which requires a {required.upper().replace('_', ' ')}. A {third_doc_type.upper().replace('_', ' ')} was provided instead.",
            policy_clause=clause_text,
            policy_reference=ref,
        )

    return PolicyFinding(
        rule_name="required_documents",
        triggered=False,
        detail=f"Required document ({required.upper().replace('_', ' ')}) is present.",
    )


def check_keyword_exclusions(
    extractions: List[DocumentExtraction],
) -> List[PolicyFinding]:
    """Check if any document text triggers a keyword-based policy exclusion."""
    findings = []

    # Combine all text for keyword scanning
    combined_text = ""
    for ext in extractions:
        for field in [ext.incident_summary, ext.damage_area_original, ext.vehicle_description]:
            if field:
                combined_text += " " + field.lower()

    for exclusion_name, config in POLICY_EXCLUSION_MAP.items():
        matched_keywords = [
            kw for kw in config["keywords"]
            if kw.lower() in combined_text
        ]
        if matched_keywords:
            findings.append(PolicyFinding(
                rule_name=f"exclusion_{exclusion_name}",
                triggered=True,
                detail=f"Potential policy exclusion detected. Keywords found: {', '.join(matched_keywords)}.",
                policy_clause=config["text"],
                policy_reference=config["clause"],
            ))

    return findings


# =====================================================
# DECISION ROUTER
# =====================================================

def route_decision(
    crossdoc: CrossDocResult,
    policy_findings: List[PolicyFinding],
) -> Tuple[Decision, List[str]]:
    """
    Apply the strict decision precedence:
    1. REJECT: Hard policy violation or exclusion
    2. REQUEST_INFO: Missing required document/field
    3. ESCALATE: Contradiction or ambiguity
    4. APPROVE: Everything checks out

    Returns:
        Tuple of (Decision, list of reason strings).
    """
    reasons = []

    # 1. Check for REJECT conditions
    reject_findings = [f for f in policy_findings if f.triggered and f.rule_name in (
        "claim_window",
        "exclusion_commercial_use",
        "exclusion_racing",
        "exclusion_intoxication",
    )]
    if reject_findings:
        for f in reject_findings:
            reasons.append(f"POLICY VIOLATION ({f.policy_reference}): {f.detail}")
        return Decision.REJECT, reasons

    # 2. Check for REQUEST_INFO conditions
    missing_doc_findings = [
        f for f in policy_findings
        if f.triggered and f.rule_name == "required_documents"
    ]
    if missing_doc_findings:
        for f in missing_doc_findings:
            reasons.append(f"MISSING REQUIRED DOCUMENT ({f.policy_reference}): {f.detail}")
        return Decision.REQUEST_INFO, reasons

    # 3. Check for ESCALATE conditions
    if crossdoc.has_contradiction:
        contradicted_fields = [
            c.field_name for c in crossdoc.comparisons
            if c.status == CrossDocStatus.CONTRADICTION
        ]
        reasons.append(
            f"CROSS-DOCUMENT CONTRADICTION detected in: {', '.join(contradicted_fields)}. "
            f"Human investigation required."
        )
        return Decision.ESCALATE, reasons

    if crossdoc.has_ambiguous:
        ambiguous_fields = [
            c.field_name for c in crossdoc.comparisons
            if c.status == CrossDocStatus.AMBIGUOUS
        ]
        reasons.append(
            f"AMBIGUOUS evidence in: {', '.join(ambiguous_fields)}. "
            f"Human review recommended."
        )
        return Decision.ESCALATE, reasons

    # 4. APPROVE
    reasons.append("All documents are consistent. No policy violations detected. Claim is within allowed timeframe.")
    return Decision.APPROVE, reasons


def evaluate_claim(
    extractions: List[DocumentExtraction],
    crossdoc: CrossDocResult,
    third_doc_type: str,
    semantic_findings: Optional[List[PolicyFinding]] = None,
) -> EvidencePackage:
    """
    Build the complete evidence package and route the final decision.

    Args:
        extractions: Extracted facts from all documents.
        crossdoc: CrossDoc comparison matrix result.
        third_doc_type: 'fir' or 'repair_estimate'.
        semantic_findings: Optional findings from semantic policy search.

    Returns:
        Complete EvidencePackage ready for synthesis.
    """
    policy_findings = []

    # Deterministic checks
    policy_findings.append(check_claim_window(extractions))
    policy_findings.append(check_required_documents(extractions, third_doc_type))
    policy_findings.extend(check_keyword_exclusions(extractions))

    # Add semantic search findings if provided
    if semantic_findings:
        policy_findings.extend(semantic_findings)

    # Route decision
    decision, reasons = route_decision(crossdoc, policy_findings)

    return EvidencePackage(
        extractions=extractions,
        crossdoc=crossdoc,
        policy_findings=policy_findings,
        decision=decision,
        decision_reasons=reasons,
    )
