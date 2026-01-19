from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from anthropic import Anthropic, transform_schema
from extract import deidentify_report, extract_report
from pydantic import BaseModel, ConfigDict, Field

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
    "other",
]


class Abnormality(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: AbnormalityType
    severity: Severity = "unknown"
    size_mm: Optional[float] = None
    laterality: Optional[Literal["left", "right", "bilateral", "midline", "unknown"]] = None
    region: Optional[Literal["central", "paracentral", "foraminal", "extraforaminal", "unknown"]] = None
    notes: str = ""


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


class UnrealLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str
    weights: Dict[str, float] = Field(default_factory=dict)


class UnrealJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    levels: List[UnrealLevel] = Field(default_factory=list)
    global_findings: GlobalFindings = Field(default_factory=GlobalFindings)
    meta: Dict[str, Any] = Field(default_factory=dict)
    morph_targets: Dict[str, float] = Field(default_factory=dict)


_ALLOWED_LEVELS = {"T2-3", "T12-L1"}


def _normalize_level(level: str) -> str:
    s = (level or "").strip().upper()
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(" ", "")
    if s == "T2-T3":
        return "T2-3"
    if s == "T3-T4":
        return "T3-4"
    if s == "T11-T12":
        return "T11-12"
    return s


def _sev_to_weight(sev: str) -> Optional[float]:
    if sev == "mild":
        return 0.3
    if sev == "moderate":
        return 0.6
    if sev == "severe":
        return 0.9
    return None


def _compute_weights(abns: List[Abnormality]) -> Dict[str, float]:
    disc_bulge_mm: Optional[float] = None
    canal_weight: Optional[float] = None
    foram_weight: Optional[float] = None

    for a in abns:
        if a.type in ("annular_bulge", "disc_bulge"):
            if isinstance(a.size_mm, (int, float)):
                x = float(a.size_mm)
                if disc_bulge_mm is None or x > disc_bulge_mm:
                    disc_bulge_mm = x

        if a.type == "stenosis":
            w = _sev_to_weight(a.severity)
            if w is not None:
                canal_weight = w if canal_weight is None else max(canal_weight, w)

        if a.type == "foraminal_narrowing":
            w = _sev_to_weight(a.severity)
            if w is not None:
                foram_weight = w if foram_weight is None else max(foram_weight, w)

    out: Dict[str, float] = {}

    if disc_bulge_mm is not None:
        out["disc_bulge"] = min(max(disc_bulge_mm / 5.0, 0.0), 1.0)

    if canal_weight is not None:
        out["canal_stenosis"] = min(max(canal_weight, 0.0), 1.0)

    if foram_weight is not None:
        out["foraminal_narrowing"] = min(max(foram_weight, 0.0), 1.0)

    return out


def _build_morph_targets(levels: List[UnrealLevel]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lvl in levels:
        tag = lvl.level.replace("-", "_")
        for k, v in lvl.weights.items():
            out[f"{k}_{tag}"] = float(v)
    return out


_SYSTEM = (
    "Return ONLY JSON matching the schema.\n"
    "Extract ONLY abnormal spine levels explicitly described.\n"
    "Do NOT output normal levels.\n"
    "Level format MUST be like T2-3 and T12-L1.\n"
    "For each abnormal level, output abnormalities with type, severity, size_mm when present.\n"
    "If the report explicitly states no cord compression and no nerve root impingement, set those false.\n"
    "Include alignment_notes if described.\n"
    "Include incidental non-spine findings in incidental.\n"
)


def ask_model(report: str, *, model: str = "claude-sonnet-4-5") -> ExtractedJson:
    client = Anthropic()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        betas=["structured-outputs-2025-11-13"],
        system=_SYSTEM,
        messages=[{"role": "user", "content": report}],
        output_format={"type": "json_schema", "schema": transform_schema(ExtractedJson)},
    )
    text = resp.content[0].text
    return ExtractedJson.model_validate_json(text)


def to_unreal_ready(extracted: ExtractedJson) -> UnrealJson:
    unreal_levels: List[UnrealLevel] = []

    for lvl in extracted.levels:
        nl = _normalize_level(lvl.level)
        if nl not in _ALLOWED_LEVELS:
            continue

        weights = _compute_weights(lvl.abnormalities)
        if not weights:
            continue

        unreal_levels.append(UnrealLevel(level=nl, weights=weights))

    meta = dict(extracted.meta) if isinstance(extracted.meta, dict) else {}
    meta["kept_levels"] = [x.level for x in unreal_levels]

    out = UnrealJson(
        levels=unreal_levels,
        global_findings=extracted.global_findings,
        meta=meta,
        morph_targets=_build_morph_targets(unreal_levels),
    )
    return out


test_report = """
Sunnyvale Imaging Center PMC
568 S. Mathilda Avenue
Sunnyvale, California 94086
T 408 738 0232 F 408 738 0242
https://pacs.sunnyvaleimaging.com/Portal
PATIENT NAME: ZHAOQI, JIN
Patient ID: 38568
Gender: Female
Date of Birth: October 21, 1986
Referring Physician: SHU, VICTOR
Accession Number: 257679-2
Report Status: Final
Type of Study: THORACIC SPINE
Modality: MR
Sunnyvale Imaging Center DBA for Golden State Diagnostic, Inc (Licensed IDTF) - Page 1 of 3
Date of Service: December 30, 2025
IMPRESSION
1. No focal disc protrusions or extrusions identified.
2. T2-3, T3-4, T11-12 and T12-L1 levels show 2 mm posterior annular bulges and crowding
of the ventral cord, mild canal stenosis AP dimension 9-10 mm, mild foraminal compromise.
No cord compression or nerve root impingement detected.
3. Visualized portions of the abdomen suggest enlargement and fatty infiltration of the liver
with 5 mm rounded high T2 signal intensity possibly cystic structure posterior segment right
lobe. Correlation with abdominal ultrasound is suggested for further evaluation.
INDICATION
Mid back pain. No DOI.
TECHNIQUE
Sagittal T2 SE, sagittal T1 FSE, sagittal STIR, axial T2. Scan performed on Siemens
Highfield 1.5T magnet.
FINDINGS
Correlation is made to a prior MRI of the lumbar spine dated 03/16/2023.
Prominent straightening of normal thoracic kyphosis that may be due to muscle spasm.
Sunnyvale Imaging Center PMC
568 S. Mathilda Avenue
Sunnyvale, California 94086
T 408 738 0232 F 408 738 0242
https://pacs.sunnyvaleimaging.com/Portal
PATIENT NAME: ZHAOQI, JIN
Patient ID: 38568
Gender: Female
Date of Birth: October 21, 1986
Referring Physician: SHU, VICTOR
Accession Number: 257679-2
Report Status: Final
Type of Study: THORACIC SPINE
Modality: MR
Sunnyvale Imaging Center DBA for Golden State Diagnostic, Inc (Licensed IDTF) - Page 2 of 3
Normal thoracic spinal cord without signal or morphologic alteration.
There is no spondylolisthesis or spondylolysis.
Normal signal intensity of the bone marrow, without infiltrative or destructive processes.
Visualized portions of the abdomen suggest enlargement and fatty infiltration of the liver
with 5 mm rounded high T2 signal intensity possibly cystic structure posterior segment right
lobe, axial T2 image #25.
T1-T2 level shows normal intervertebral disc with no spondylosis or facet joint arthrosis.
No disc bulge or disc protrusion present. Normal central canal and intervertebral neural
foramina, with no cord or radicular impingement.
T2-T3 level shows a 2 mm posterior annular bulge with effacement of ventral CSF and
crowding of the ventral cord, mild canal stenosis AP dimension 9-10 mm, mild foraminal
compromise. No cord compression or nerve root impingement detected.
T3-T4 level shows a 2 mm posterior annular bulge with effacement of ventral CSF and
crowding of the ventral cord, mild canal stenosis AP dimension 9-10 mm, mild foraminal
compromise. No cord compression or nerve root impingement detected.
T4-T5 level shows normal intervertebral disc with no spondylosis or facet joint arthrosis.
No disc bulge or disc protrusion present. Normal central canal and intervertebral neural
foramina, with no cord or radicular impingement.
T5-T6 level shows normal intervertebral disc with no spondylosis or facet joint arthrosis.
No disc bulge or disc protrusion present. Normal central canal and intervertebral neural
foramina, with no cord or radicular impingement.
T6-T7 level shows normal intervertebral disc with no spondylosis or facet joint arthrosis.
No disc bulge or disc protrusion present. Normal central canal and intervertebral neural
foramina, with no cord or radicular impingement.
T7-T8 level shows normal intervertebral disc with no spondylosis or facet joint arthrosis.
No disc bulge or disc protrusion present. Normal central canal and intervertebral neural
foramina, with no cord or radicular impingement.
Sunnyvale Imaging Center PMC
568 S. Mathilda Avenue
Sunnyvale, California 94086
T 408 738 0232 F 408 738 0242
https://pacs.sunnyvaleimaging.com/Portal
PATIENT NAME: ZHAOQI, JIN
Patient ID: 38568
Gender: Female
Date of Birth: October 21, 1986
Referring Physician: SHU, VICTOR
Accession Number: 257679-2
Report Status: Final
Type of Study: THORACIC SPINE
Modality: MR
Sunnyvale Imaging Center DBA for Golden State Diagnostic, Inc (Licensed IDTF) - Page 3 of 3
T8-T9 level shows normal intervertebral disc with no spondylosis or facet joint arthrosis.
No disc bulge or disc protrusion present. Normal central canal and intervertebral neural
foramina, with no cord or radicular impingement.
T9-T10 level shows normal intervertebral disc with no spondylosis or facet joint arthrosis.
No disc bulge or disc protrusion present. Normal central canal and intervertebral neural
foramina, with no cord or radicular impingement.
T10-T11 level shows normal intervertebral disc with no spondylosis or facet joint arthrosis.
No disc bulge or disc protrusion present. Normal central canal and intervertebral neural
foramina, with no cord or radicular impingement.
T11-T12 level shows a 2 mm posterior annular bulge with effacement of ventral CSF and
crowding of the ventral cord, mild canal stenosis AP dimension 9-10 mm, mild foraminal
compromise. No cord compression or nerve root impingement detected.
T12-L1 level shows a 2 mm posterior annular bulge with effacement of ventral CSF and
crowding of the ventral cord, mild canal stenosis AP dimension 9-10 mm, mild foraminal
compromise. No cord compression or nerve root impingement detected.
Electronically signed by: Tatiana Voci (Dec 30, 2025 16:26:31)
"""


if __name__ == "__main__":
    deid = deidentify_report(test_report)
    extracted_text = extract_report(deid.text)
    extracted_json = ask_model(extracted_text)
    unreal_ready = to_unreal_ready(extracted_json)
    print(json.dumps(unreal_ready.model_dump(), indent=2, ensure_ascii=False))
