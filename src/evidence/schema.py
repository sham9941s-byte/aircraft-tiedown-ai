from dataclasses import dataclass, field
from typing import Literal, Optional


Quality = Literal["good", "bad", "unknown"]
Decision = Literal["GOOD_TO_GO", "TIE_DOWN_INCORRECT", "MORE_IMAGES_REQUIRED"]


@dataclass
class Evidence:
    image_id: str
    view: str
    straps: int = 0
    chains: int = 0
    connection_points_verified: int = 0
    suspension: Quality = "unknown"
    tie_down_quality: Quality = "unknown"
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class Assessment:
    decision: Decision
    confidence: float
    images_analyzed: int
    evidence: list[Evidence] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    additional_images_required: bool = False
