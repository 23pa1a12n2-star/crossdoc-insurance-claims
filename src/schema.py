"""
Pydantic schemas defining the strict data contracts for the CrossDoc pipeline.

These models enforce structured extraction from Gemini and deterministic
comparison throughout the Python layers. Every extracted fact carries its
source citation for full traceability.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# =====================================================
# ENUMS
# =====================================================

class Precision(str, Enum):
    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class ClaimType(str, Enum):
    ACCIDENT = "ACCIDENT"
    THEFT = "THEFT"
    MALICIOUS_DAMAGE = "MALICIOUS_DAMAGE"
    OTHER = "OTHER"


class DamageArea(str, Enum):
    FRONT = "FRONT"
    REAR = "REAR"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    MULTIPLE = "MULTIPLE"
    TOTAL_LOSS = "TOTAL_LOSS"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class DamageSeverity(str, Enum):
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    MAJOR = "MAJOR"
    TOTAL_LOSS = "TOTAL_LOSS"
    UNKNOWN = "UNKNOWN"


class ThirdDocType(str, Enum):
    FIR = "FIR"
    REPAIR_ESTIMATE = "REPAIR_ESTIMATE"


class CrossDocStatus(str, Enum):
    MATCH = "MATCH"
    COMPATIBLE = "COMPATIBLE"
    CONTRADICTION = "CONTRADICTION"
    MISSING = "MISSING"
    AMBIGUOUS = "AMBIGUOUS"


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_INFO = "REQUEST_INFO"
    ESCALATE = "ESCALATE"


# =====================================================
# EXTRACTED FACT MODEL
# =====================================================

class ExtractedFact(BaseModel):
    """A single fact extracted from one document, preserving the source quote."""
    field_name: str = Field(description="Canonical field identifier")
    normalized_value: Optional[str] = Field(default=None, description="Python-normalized value")
    original_value: Optional[str] = Field(default=None, description="Raw value as written in document")
    precision: Precision = Field(default=Precision.UNKNOWN, description="How precise the value is")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraction confidence heuristic")
    exact_citation: Optional[str] = Field(default=None, description="Exact substring from source")
    source_document: str = Field(description="Which document this fact came from")


# =====================================================
# DOCUMENT EXTRACTION MODEL
# =====================================================

class DocumentExtraction(BaseModel):
    """Structured extraction from a single document."""
    source_document: str = Field(description="claim_form | customer_description | fir | repair_estimate")
    incident_date: Optional[str] = Field(default=None, description="ISO date string YYYY-MM-DD")
    incident_date_original: Optional[str] = Field(default=None, description="Original date text")
    incident_date_precision: Precision = Field(default=Precision.UNKNOWN)
    incident_time: Optional[str] = Field(default=None, description="HH:MM 24hr format")
    incident_time_original: Optional[str] = Field(default=None, description="Original time text")
    incident_time_precision: Precision = Field(default=Precision.UNKNOWN)
    claim_type: Optional[ClaimType] = Field(default=None)
    driver_name: Optional[str] = Field(default=None, description="Lowercase normalized")
    driver_name_original: Optional[str] = Field(default=None)
    damage_area: Optional[DamageArea] = Field(default=None)
    damage_area_original: Optional[str] = Field(default=None, description="Original description of damage location")
    damage_severity: Optional[DamageSeverity] = Field(default=None)
    estimated_cost: Optional[float] = Field(default=None, description="Numeric cost in base currency")
    estimated_cost_original: Optional[str] = Field(default=None)
    vehicle_description: Optional[str] = Field(default=None)
    incident_summary: Optional[str] = Field(default=None, description="Brief summary of what happened")
    report_date: Optional[str] = Field(default=None, description="Date the claim was reported/filed ISO")
    report_date_original: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Overall extraction confidence")


# =====================================================
# CROSSDOC COMPARISON RESULT
# =====================================================

class FieldComparison(BaseModel):
    """Result of comparing a single field across documents."""
    field_name: str
    status: CrossDocStatus
    values: dict = Field(default_factory=dict, description="source_doc -> value mapping")
    original_values: dict = Field(default_factory=dict, description="source_doc -> original text")
    citations: dict = Field(default_factory=dict, description="source_doc -> exact citation")
    explanation: str = Field(default="", description="Why this status was assigned")


class CrossDocResult(BaseModel):
    """Full CrossDoc matrix output."""
    comparisons: List[FieldComparison] = Field(default_factory=list)
    has_contradiction: bool = False
    has_missing: bool = False
    has_ambiguous: bool = False
    summary: str = ""


# =====================================================
# POLICY FINDING
# =====================================================

class PolicyFinding(BaseModel):
    """A single policy evaluation finding."""
    rule_name: str
    triggered: bool
    detail: str = ""
    policy_clause: Optional[str] = Field(default=None, description="The exact clause text retrieved")
    policy_reference: Optional[str] = Field(default=None, description="e.g. Section 2.1")


# =====================================================
# EVIDENCE PACKAGE
# =====================================================

class EvidencePackage(BaseModel):
    """Complete evidence assembled for the Decision Router."""
    extractions: List[DocumentExtraction] = Field(default_factory=list)
    crossdoc: CrossDocResult = Field(default_factory=CrossDocResult)
    policy_findings: List[PolicyFinding] = Field(default_factory=list)
    decision: Decision = Decision.ESCALATE
    decision_reasons: List[str] = Field(default_factory=list)
    synthesis: str = Field(default="", description="Gemini-generated human-readable explanation")
