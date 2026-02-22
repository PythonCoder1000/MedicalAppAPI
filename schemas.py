from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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

JointName = Literal["center", "topright", "bottomright", "topleft", "bottomleft"]
JointId = Literal["joint2", "joint4", "joint5", "joint6", "joint7"]


class LabelRequest(BaseModel):
    text: str


class LabelResponse(BaseModel):
    labels: List[str] = Field(default_factory=list)


class MorphRequest(BaseModel):
    text: str
    use_ai_deid: bool = True


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
        aliases = {
            "disc_height_loss": "disc_height_loss",
            "disc_height": "disc_height_loss",
            "disc_space_narrowing": "disc_height_loss",
            "disc_space_narrowing.": "disc_height_loss",
            "disc_space_loss": "disc_height_loss",
            "disc_space_collapse": "disc_height_loss",
            "disc_collapse": "disc_height_loss",
            "loss_of_disc_height": "disc_height_loss",
            "decreased_disc_height": "disc_height_loss",
            "disc_height_reduction": "disc_height_loss",
        }
        s = aliases.get(s, s)
        allowed = {
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
            "other",
        }
        return s if s in allowed else "other"

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> Any:
        if v is None:
            return "unknown"
        s = str(v).strip().lower()
        if s in {"none", "mild", "moderate", "severe", "unknown"}:
            return s
        if "mild" in s or s in {"minimal", "slight"}:
            return "mild"
        if "moderate" in s:
            return "moderate"
        if "severe" in s:
            return "severe"
        return "unknown"

    @field_validator("laterality", mode="before")
    @classmethod
    def normalize_laterality(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        aliases = {
            "l": "left",
            "left": "left",
            "r": "right",
            "right": "right",
            "bilateral": "bilateral",
            "bilat": "bilateral",
            "both": "bilateral",
            "midline": "midline",
            "central": "midline",
            "none": "unknown",
            "unknown": "unknown",
        }
        return aliases.get(s, "unknown")

    @field_validator("region", mode="before")
    @classmethod
    def normalize_region(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "central": "central",
            "midline": "central",
            "disc": "central",
            "disc_space": "central",
            "canal": "central",
            "central_canal": "central",
            "paracentral": "paracentral",
            "para_central": "paracentral",
            "foraminal": "foraminal",
            "foramen": "foraminal",
            "neural_foramen": "foraminal",
            "extraforaminal": "extraforaminal",
            "far_lateral": "extraforaminal",
            "unknown": "unknown",
            "none": "unknown",
        }
        return aliases.get(s, "unknown")


class ExtractedLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str
    abnormalities: List[Abnormality] = Field(default_factory=list)


class GlobalFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cord_compression: bool = False
    nerve_root_impingement: bool = False
    alignment_notes: str = ""
    incidental: List[str] = Field(default_factory=list)


class ExtractedJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    levels: List[ExtractedLevel] = Field(default_factory=list)
    global_findings: GlobalFindings = Field(default_factory=GlobalFindings)
    meta: Dict[str, Any] = Field(default_factory=dict)


class DiscOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str
    top_bone: str
    bottom_bone: str
    joints: Dict[JointId, float] = Field(default_factory=dict)


class MorphResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    morph_targets: Dict[str, float] = Field(default_factory=dict)
    discs: List[DiscOut] = Field(default_factory=list)
    global_findings: GlobalFindings = Field(default_factory=GlobalFindings)
    meta: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
