from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils import TYPE_ALIASES, VALID_TYPES, SEVERITY_ALIASES

Severity = Literal["none", "mild", "moderate", "severe", "unknown"]
AbnormalityType = Literal[
    "annular_bulge",
    "disc_bulge",
    "protrusion",
    "extrusion",
    "stenosis",
    "foraminal_narrowing",
    "facet_arthropathy",
    "alignment",
    "fracture",
    "edema",
    "cord_compression",
    "nerve_root_impingement",
    "disc_height_loss",
    "other",
]
Laterality = Literal["left", "right", "bilateral", "midline", "unknown"]
Region = Literal["central", "paracentral", "foraminal", "extraforaminal", "unknown"]


class MorphRequest(BaseModel):
    text: str
    use_deid: bool = True


class Abnormality(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: AbnormalityType
    severity: Severity = "unknown"
    size_mm: Optional[float] = None
    laterality: Optional[Laterality] = None
    region: Optional[Region] = None
    notes: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> Any:
        if v is None:
            return "other"
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        s = TYPE_ALIASES.get(s, s)
        return s if s in VALID_TYPES else "other"

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> Any:
        if v is None:
            return "unknown"
        s = str(v).strip().lower()
        if s in {"none", "mild", "moderate", "severe", "unknown"}:
            return s
        s = SEVERITY_ALIASES.get(s, s)
        for keyword in ("mild", "moderate", "severe"):
            if keyword in s:
                return keyword
        return "unknown"

    @field_validator("laterality", mode="before")
    @classmethod
    def normalize_laterality(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        aliases = {
            "l": "left", "left": "left",
            "r": "right", "right": "right",
            "bilateral": "bilateral", "bilat": "bilateral", "both": "bilateral",
            "midline": "midline", "central": "midline",
            "none": "unknown", "unknown": "unknown",
        }
        return aliases.get(s, "unknown")

    @field_validator("region", mode="before")
    @classmethod
    def normalize_region(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "central": "central", "midline": "central", "disc": "central",
            "disc_space": "central", "canal": "central", "central_canal": "central",
            "paracentral": "paracentral", "para_central": "paracentral",
            "foraminal": "foraminal", "foramen": "foraminal", "neural_foramen": "foraminal",
            "extraforaminal": "extraforaminal", "far_lateral": "extraforaminal",
            "unknown": "unknown", "none": "unknown",
        }
        return aliases.get(s, "unknown")


class ExtractedLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str
    abnormalities: List[Abnormality] = Field(default_factory=list)


Gender = Literal["male", "female", "unknown"]


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gender: Gender = "unknown"


class DiscTarget(BaseModel):
    disc: str
    index: Optional[int] = None
    joint_1: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_2: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_3: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_4: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_center: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class MorphResponse(BaseModel):
    patient: Patient = Field(default_factory=Patient)
    targets: List[DiscTarget] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
