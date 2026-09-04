"""
Gemini Synthesis Layer.

Responsibility:
- Takes the Python-generated EvidencePackage (decision + evidence)
- Produces a polished, human-readable investigation summary
- Gemini ONLY formats/explains; it does NOT change the decision

The decision has already been made by Python.
Gemini translates structured findings into clear investigator language.
"""

import os
import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from src.schema import EvidencePackage

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

MODEL = "gemini-3.5-flash-lite"

SYNTHESIS_SYSTEM_PROMPT = """You are an insurance claims report writer.
You are given a structured evidence package that contains:
1. Extracted facts from claim documents
2. Cross-document comparison results (MATCH, COMPATIBLE, CONTRADICTION, MISSING, AMBIGUOUS)
3. Policy evaluation findings
4. A FINAL DECISION that has already been made by the review system

Your job is to write a clear, professional, neutral summary for a claims investigator.

RULES:
1. Do NOT change the decision. The decision is final.
2. Do NOT accuse the claimant of fraud, lying, or dishonesty.
3. Use neutral, professional language throughout.
4. Reference specific findings from the evidence package.
5. Quote the exact policy clause when a policy finding is involved.
6. Clearly state what was consistent and what was inconsistent.
7. For ESCALATE decisions, explain what needs human investigation.
8. For REQUEST_INFO decisions, specify exactly what document is missing.
9. Keep the summary concise: 3-5 paragraphs maximum.
10. Start with the recommendation, then explain the reasoning."""


def synthesize_report(evidence: EvidencePackage) -> str:
    """
    Generate a human-readable report from the evidence package.

    Args:
        evidence: Complete EvidencePackage from the decision router.

    Returns:
        Formatted string report for the investigator.
    """
    # Build a structured summary for Gemini to format
    evidence_summary = {
        "decision": evidence.decision.value,
        "reasons": evidence.decision_reasons,
        "crossdoc_summary": evidence.crossdoc.summary,
        "comparisons": [
            {
                "field": c.field_name,
                "status": c.status.value,
                "values": c.values,
                "original_values": c.original_values,
                "explanation": c.explanation,
            }
            for c in evidence.crossdoc.comparisons
        ],
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
    }

    user_prompt = f"""Based on the following evidence package, write a professional claims review summary for the investigator.

Evidence Package:
{json.dumps(evidence_summary, indent=2)}

Write the summary now."""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini synthesis failed: {e}")
        # Fallback: return a deterministic summary from Python
        return _fallback_synthesis(evidence)


def _fallback_synthesis(evidence: EvidencePackage) -> str:
    """Generate a basic report without Gemini if the API fails."""
    lines = []
    lines.append(f"RECOMMENDATION: {evidence.decision.value}")
    lines.append("")

    for reason in evidence.decision_reasons:
        lines.append(f"• {reason}")
    lines.append("")

    lines.append("CROSS-DOCUMENT ANALYSIS:")
    for comp in evidence.crossdoc.comparisons:
        icon = {
            "MATCH": "✓",
            "COMPATIBLE": "~",
            "CONTRADICTION": "✗",
            "MISSING": "?",
            "AMBIGUOUS": "⚠",
        }.get(comp.status.value, "-")
        lines.append(f"  {icon} {comp.field_name}: {comp.status.value} — {comp.explanation}")
    lines.append("")

    triggered_policies = [f for f in evidence.policy_findings if f.triggered]
    if triggered_policies:
        lines.append("POLICY FINDINGS:")
        for pf in triggered_policies:
            lines.append(f"  • {pf.policy_reference}: {pf.detail}")
            if pf.policy_clause:
                lines.append(f'    Clause: "{pf.policy_clause}"')

    return "\n".join(lines)
