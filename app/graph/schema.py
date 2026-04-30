from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegulationNode:
    name: str
    description: Optional[str] = None
    effective_date: Optional[str] = None
    issuing_body: Optional[str] = None


@dataclass
class CapitalRatioNode:
    name: str
    minimum_threshold: Optional[float] = None
    description: Optional[str] = None


@dataclass
class JurisdictionNode:
    name: str
    region: Optional[str] = None


@dataclass
class InstitutionNode:
    name: str
    institution_type: Optional[str] = None


@dataclass
class MetricNode:
    name: str
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Relationship:
    source: str
    target: str
    type: str
    properties: dict = field(default_factory=dict)


VALID_NODE_TYPES: set[str] = {
    "Regulation",
    "CapitalRatio",
    "Jurisdiction",
    "Institution",
    "Metric",
}

VALID_RELATIONSHIP_TYPES: set[str] = {
    "AFFECTS",
    "APPLIES_IN",
    "REFERENCES",
    "DEFINES",
    "REQUIRES",
}

DEFAULT_SCHEMA_CONTEXT = """
Node types and key properties:
  - Regulation      : name, description, effective_date, issuing_body
  - CapitalRatio    : name, minimum_threshold, description
  - Jurisdiction    : name, region
  - Institution     : name, institution_type
  - Metric          : name, unit, description

Relationship types:
  - AFFECTS         : (Regulation)-[:AFFECTS]->(CapitalRatio)
  - APPLIES_IN      : (Regulation)-[:APPLIES_IN]->(Jurisdiction)
  - REFERENCES      : (Regulation)-[:REFERENCES]->(Regulation)
  - DEFINES         : (Regulation)-[:DEFINES]->(Metric)
  - REQUIRES        : (Regulation)-[:REQUIRES]->(Institution)
"""
