"""
CrossDoc Evidence Reconciliation Engine.

This is a PURE PYTHON deterministic module.
It does NOT call any LLM or external API.

Responsibilities:
- Compare extracted facts across 2-3 documents
- Classify each field comparison as: MATCH, COMPATIBLE, CONTRADICTION, MISSING, AMBIGUOUS
- Preserve exact citations for traceability
- Never accuse the claimant; only flag evidence

Rules:
- Exact equality on normalized values → MATCH
- Approximate/range values that overlap with exact values → COMPATIBLE
- Mutually exclusive values → CONTRADICTION
- Null in a document where the field exists elsewhere → MISSING
- Low confidence or unparseable extraction → AMBIGUOUS
"""

from datetime import datetime, timedelta
from typing import List, Optional

from src.schema import (
    DocumentExtraction,
    FieldComparison,
    CrossDocResult,
    CrossDocStatus,
    Precision,
    DamageArea,
    DamageSeverity,
)


# Configurable thresholds
TIME_TOLERANCE_HOURS = 2
COST_TOLERANCE_RATIO = 0.3  # 30% difference
CONFIDENCE_THRESHOLD = 0.70  # Below this → AMBIGUOUS


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Safely parse ISO date string."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


def _parse_time(time_str: Optional[str]) -> Optional[int]:
    """Parse HH:MM to total minutes since midnight."""
    if not time_str:
        return None
    try:
        parts = time_str.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError, AttributeError):
        return None


def _compare_dates(
    extractions: List[DocumentExtraction],
) -> FieldComparison:
    """Compare incident_date across documents."""
    values = {}
    originals = {}
    citations = {}
    precisions = {}

    for ext in extractions:
        if ext.incident_date is not None:
            values[ext.source_document] = ext.incident_date
            originals[ext.source_document] = ext.incident_date_original or ext.incident_date
            citations[ext.source_document] = ext.incident_date_original or ext.incident_date
            precisions[ext.source_document] = ext.incident_date_precision

    if len(values) == 0:
        return FieldComparison(
            field_name="incident_date",
            status=CrossDocStatus.MISSING,
            values=values,
            original_values=originals,
            citations=citations,
            explanation="No document provided an incident date.",
        )

    if len(values) == 1:
        return FieldComparison(
            field_name="incident_date",
            status=CrossDocStatus.MISSING,
            values=values,
            original_values=originals,
            citations=citations,
            explanation="Incident date found in only one document. Cannot cross-verify.",
        )

    # Parse all dates
    parsed = {}
    for doc, val in values.items():
        p = _parse_date(val)
        if p is None:
            return FieldComparison(
                field_name="incident_date",
                status=CrossDocStatus.AMBIGUOUS,
                values=values,
                original_values=originals,
                citations=citations,
                explanation=f"Unable to parse date from {doc}: '{val}'",
            )
        parsed[doc] = p

    dates = list(parsed.values())
    date_set = set(d.date() for d in dates)

    if len(date_set) == 1:
        # Check if any are approximate
        has_approx = any(
            precisions.get(d) == Precision.APPROXIMATE for d in values
        )
        status = CrossDocStatus.COMPATIBLE if has_approx else CrossDocStatus.MATCH
        return FieldComparison(
            field_name="incident_date",
            status=status,
            values=values,
            original_values=originals,
            citations=citations,
            explanation="Incident dates are consistent across documents.",
        )

    # Dates differ
    return FieldComparison(
        field_name="incident_date",
        status=CrossDocStatus.CONTRADICTION,
        values=values,
        original_values=originals,
        citations=citations,
        explanation="Incident dates conflict across documents.",
    )


def _compare_times(
    extractions: List[DocumentExtraction],
) -> FieldComparison:
    """Compare incident_time across documents."""
    values = {}
    originals = {}
    citations = {}
    precisions = {}

    for ext in extractions:
        if ext.incident_time is not None:
            values[ext.source_document] = ext.incident_time
            originals[ext.source_document] = ext.incident_time_original or ext.incident_time
            citations[ext.source_document] = ext.incident_time_original or ext.incident_time
            precisions[ext.source_document] = ext.incident_time_precision

    if len(values) <= 1:
        status = CrossDocStatus.MISSING if len(values) == 0 else CrossDocStatus.MISSING
        return FieldComparison(
            field_name="incident_time",
            status=status,
            values=values,
            original_values=originals,
            citations=citations,
            explanation="Incident time not available in enough documents to compare.",
        )

    parsed = {}
    for doc, val in values.items():
        p = _parse_time(val)
        if p is None:
            return FieldComparison(
                field_name="incident_time",
                status=CrossDocStatus.AMBIGUOUS,
                values=values,
                original_values=originals,
                citations=citations,
                explanation=f"Unable to parse time from {doc}: '{val}'",
            )
        parsed[doc] = p

    times = list(parsed.values())
    min_t = min(times)
    max_t = max(times)
    diff_minutes = max_t - min_t

    if diff_minutes == 0:
        has_approx = any(
            precisions.get(d) == Precision.APPROXIMATE for d in values
        )
        status = CrossDocStatus.COMPATIBLE if has_approx else CrossDocStatus.MATCH
        return FieldComparison(
            field_name="incident_time",
            status=status,
            values=values,
            original_values=originals,
            citations=citations,
            explanation="Incident times are consistent across documents.",
        )

    tolerance = TIME_TOLERANCE_HOURS * 60
    if diff_minutes <= tolerance:
        return FieldComparison(
            field_name="incident_time",
            status=CrossDocStatus.COMPATIBLE,
            values=values,
            original_values=originals,
            citations=citations,
            explanation=f"Times differ by {diff_minutes} minutes, within {TIME_TOLERANCE_HOURS}-hour tolerance.",
        )

    return FieldComparison(
        field_name="incident_time",
        status=CrossDocStatus.CONTRADICTION,
        values=values,
        original_values=originals,
        citations=citations,
        explanation=f"Incident times conflict by {diff_minutes} minutes across documents.",
    )


def _compare_enum_field(
    extractions: List[DocumentExtraction],
    field_name: str,
    attr_name: str,
    original_attr: str,
) -> FieldComparison:
    """Generic comparison for Enum-type fields (damage_area, damage_severity, claim_type)."""
    values = {}
    originals = {}

    for ext in extractions:
        val = getattr(ext, attr_name, None)
        if val is not None:
            values[ext.source_document] = val.value if hasattr(val, "value") else str(val)
            orig = getattr(ext, original_attr, None)
            originals[ext.source_document] = orig or values[ext.source_document]

    if len(values) <= 1:
        return FieldComparison(
            field_name=field_name,
            status=CrossDocStatus.MISSING,
            values=values,
            original_values=originals,
            explanation=f"{field_name} not available in enough documents to compare.",
        )

    unique_vals = set(values.values())
    # Filter out UNKNOWN
    meaningful = {v for v in unique_vals if v != "UNKNOWN"}

    if len(meaningful) == 0:
        return FieldComparison(
            field_name=field_name,
            status=CrossDocStatus.AMBIGUOUS,
            values=values,
            original_values=originals,
            explanation=f"All {field_name} values are UNKNOWN.",
        )

    if len(meaningful) == 1:
        return FieldComparison(
            field_name=field_name,
            status=CrossDocStatus.MATCH,
            values=values,
            original_values=originals,
            explanation=f"{field_name} values are consistent across documents.",
        )

    # Specific fix to handle MULTIPLE overlapping with a specific damage area
    if field_name == "damage_area" and "MULTIPLE" in meaningful:
        return FieldComparison(
            field_name=field_name,
            status=CrossDocStatus.COMPATIBLE,
            values=values,
            original_values=originals,
            explanation=f"{field_name} values overlap (one document specifies a precise area, another indicates multiple).",
        )

    return FieldComparison(
        field_name=field_name,
        status=CrossDocStatus.CONTRADICTION,
        values=values,
        original_values=originals,
        explanation=f"{field_name} values conflict across documents.",
    )


def _compare_driver_name(
    extractions: List[DocumentExtraction],
) -> FieldComparison:
    """Compare driver names using normalized lowercase comparison."""
    values = {}
    originals = {}

    for ext in extractions:
        if ext.driver_name:
            values[ext.source_document] = ext.driver_name.strip().lower()
            originals[ext.source_document] = ext.driver_name_original or ext.driver_name

    if len(values) <= 1:
        return FieldComparison(
            field_name="driver_name",
            status=CrossDocStatus.MISSING,
            values=values,
            original_values=originals,
            explanation="Driver name not available in enough documents to compare.",
        )

    unique_names = set(values.values())

    if len(unique_names) == 1:
        return FieldComparison(
            field_name="driver_name",
            status=CrossDocStatus.MATCH,
            values=values,
            original_values=originals,
            explanation="Driver names are consistent across documents.",
        )

    return FieldComparison(
        field_name="driver_name",
        status=CrossDocStatus.CONTRADICTION,
        values=values,
        original_values=originals,
        explanation="Driver names differ across documents.",
    )


def _compare_costs(
    extractions: List[DocumentExtraction],
) -> FieldComparison:
    """Compare estimated costs using tolerance ratio."""
    values = {}
    originals = {}

    for ext in extractions:
        if ext.estimated_cost is not None:
            values[ext.source_document] = ext.estimated_cost
            originals[ext.source_document] = ext.estimated_cost_original or str(ext.estimated_cost)

    if len(values) <= 1:
        return FieldComparison(
            field_name="estimated_cost",
            status=CrossDocStatus.MISSING,
            values={k: str(v) for k, v in values.items()},
            original_values=originals,
            explanation="Estimated cost not available in enough documents to compare.",
        )

    costs = list(values.values())
    min_c = min(costs)
    max_c = max(costs)

    if min_c == 0 and max_c == 0:
        return FieldComparison(
            field_name="estimated_cost",
            status=CrossDocStatus.MATCH,
            values={k: str(v) for k, v in values.items()},
            original_values=originals,
            explanation="All estimated costs are zero.",
        )

    if min_c == 0:
        reference = max_c
    else:
        reference = min_c

    ratio = (max_c - min_c) / reference if reference > 0 else 0

    if ratio == 0:
        return FieldComparison(
            field_name="estimated_cost",
            status=CrossDocStatus.MATCH,
            values={k: str(v) for k, v in values.items()},
            original_values=originals,
            explanation="Estimated costs are identical.",
        )

    if ratio <= COST_TOLERANCE_RATIO:
        return FieldComparison(
            field_name="estimated_cost",
            status=CrossDocStatus.COMPATIBLE,
            values={k: str(v) for k, v in values.items()},
            original_values=originals,
            explanation=f"Costs differ by {ratio:.0%}, within {COST_TOLERANCE_RATIO:.0%} tolerance.",
        )

    return FieldComparison(
        field_name="estimated_cost",
        status=CrossDocStatus.CONTRADICTION,
        values={k: str(v) for k, v in values.items()},
        original_values=originals,
        explanation=f"Costs differ by {ratio:.0%}, exceeding {COST_TOLERANCE_RATIO:.0%} tolerance.",
    )


def _check_low_confidence(
    extractions: List[DocumentExtraction],
) -> Optional[FieldComparison]:
    """Flag if any extraction has dangerously low confidence."""
    low_docs = [
        ext.source_document
        for ext in extractions
        if ext.confidence < CONFIDENCE_THRESHOLD
    ]
    if low_docs:
        return FieldComparison(
            field_name="extraction_confidence",
            status=CrossDocStatus.AMBIGUOUS,
            values={d: "LOW" for d in low_docs},
            explanation=f"Extraction confidence below {CONFIDENCE_THRESHOLD} for: {', '.join(low_docs)}. Results may be unreliable.",
        )
    return None


def run_crossdoc(extractions: List[DocumentExtraction]) -> CrossDocResult:
    """
    Execute the full CrossDoc comparison matrix.

    Args:
        extractions: List of 2-3 DocumentExtraction models.

    Returns:
        CrossDocResult with all field comparisons and summary flags.
    """
    comparisons: List[FieldComparison] = []

    # Confidence check
    conf_check = _check_low_confidence(extractions)
    if conf_check:
        comparisons.append(conf_check)

    # Core field comparisons
    comparisons.append(_compare_dates(extractions))
    comparisons.append(_compare_times(extractions))
    comparisons.append(_compare_driver_name(extractions))
    comparisons.append(
        _compare_enum_field(extractions, "damage_area", "damage_area", "damage_area_original")
    )
    comparisons.append(
        _compare_enum_field(extractions, "damage_severity", "damage_severity", "damage_area_original")
    )
    comparisons.append(
        _compare_enum_field(extractions, "claim_type", "claim_type", "incident_summary")
    )
    comparisons.append(_compare_costs(extractions))

    # Compute summary flags
    has_contradiction = any(c.status == CrossDocStatus.CONTRADICTION for c in comparisons)
    has_missing = any(c.status == CrossDocStatus.MISSING for c in comparisons)
    has_ambiguous = any(c.status == CrossDocStatus.AMBIGUOUS for c in comparisons)

    # Build human-readable summary
    contradictions = [c.field_name for c in comparisons if c.status == CrossDocStatus.CONTRADICTION]
    summary_parts = []
    if contradictions:
        summary_parts.append(f"Contradictions found in: {', '.join(contradictions)}.")
    if has_ambiguous:
        ambig = [c.field_name for c in comparisons if c.status == CrossDocStatus.AMBIGUOUS]
        summary_parts.append(f"Ambiguous values in: {', '.join(ambig)}.")
    if not contradictions and not has_ambiguous:
        summary_parts.append("All compared fields are consistent.")

    return CrossDocResult(
        comparisons=comparisons,
        has_contradiction=has_contradiction,
        has_missing=has_missing,
        has_ambiguous=has_ambiguous,
        summary=" ".join(summary_parts),
    )
