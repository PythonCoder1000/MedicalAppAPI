from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, transform_schema
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from schemas import Abnormality, DiscTarget, MorphResponse, Patient
from utils import (
    ANTHROPIC_API_KEY, DEID_MODEL, DISC_INDEX, EXTRACT_MODEL,
    JOINT_NAMES, MAX_CORNER, MAX_SCALER,
    detect_gender, disc_name, insert_field_breaks, validate_axis,
)


class _JointAxes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    joint_1: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_2: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_3: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_4: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_center: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class _LevelMorph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str
    abnormalities: List[Abnormality] = Field(default_factory=list)
    joints: _JointAxes = Field(default_factory=_JointAxes)


class _CombinedExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient: Patient = Field(default_factory=Patient)
    levels: List[_LevelMorph] = Field(default_factory=list)


_DEID_SYSTEM = """
You are a medical report de-identification system.

Your job is to rewrite the input report as clean plain text with protected health information removed, while preserving all medical meaning.

Primary goal:
1. Remove or replace all patient-identifying information.
2. Preserve all clinically relevant content.
3. Fix formatting problems caused by missing spaces, run-together fields, or broken line structure.

Input quality:
- The input may contain missing whitespace.
- Multiple fields may be merged together on one line.
- Headers and values may run into each other.
- You must intelligently separate fields and words before producing the final cleaned text.

Replace the following with placeholders:
- Patient names -> [PATIENT_NAME]
- Medical record number / MRN -> [MRN]
- Accession number -> [ACCESSION]
- Date of birth / DOB -> [DOB]
- Any explicit date -> [DATE]
- Address -> [ADDRESS]
- Phone number -> [PHONE]
- Email address -> [EMAIL]
- Referring physician or provider name -> [PROVIDER]
- Facility, hospital, clinic, imaging center, or site name -> [FACILITY]

Preserve exactly:
- Sex or gender
- Age
- Modality
- Exam type
- Comparison statements
- Section headers such as INDICATION, TECHNIQUE, FINDINGS, IMPRESSION, COMPARISON
- All anatomy
- All measurements
- All pathology
- All clinical findings
- All medical meaning

Formatting rules:
- Insert spaces between run-together words when needed.
- Insert line breaks between distinct administrative fields and report sections when needed.
- Keep the report readable and naturally structured.
- Do not convert the report into JSON.
- Do not summarize.
- Do not explain what you changed.
- Do not add any content that was not in the report.

Important constraints:
- Only remove or replace identifying information.
- Do not remove clinical content just because it appears near identifying text.
- If a field is ambiguous, preserve it unless it is clearly identifying.
- Do not alter the medical meaning.

Output rules:
- Return only the cleaned report text.
- No JSON.
- No markdown.
- No prefatory text.
- No notes.
"""

_EXTRACT_SYSTEM = """
You are a board-certified spine radiologist and structured medical extraction system.

Your task is to read a spine MRI or CT report and return ONLY structured JSON that matches the provided schema exactly.

You must do two things:
1. Extract only explicitly stated abnormal findings from the report.
2. Convert those explicit findings into conservative UE5 disc morph offsets using the rules below.

Core rule:
Everything in the output must be supported by the report text.
If it is not explicitly stated, do not output it.

Absolute extraction rules:
1. Extract only findings explicitly written in the report.
2. Never infer, speculate, or invent pathology.
3. Skip levels described as normal, unremarkable, or without significant abnormality.
4. If severity is not stated, use "unknown".
5. If laterality is not stated, use "unknown".
6. If region is not stated, use "unknown".
7. If no numeric size is given, use null for size_mm.
8. When uncertain, output fewer findings and smaller morph values.
9. Every abnormality in the JSON must be traceable to wording in the report.

Level rules:
- Normalize levels to format like C5-C6, T12-L1, L4-L5.
- Use only levels explicitly mentioned.
- Do not create levels.
- If the report describes a global finding without a specific level, place it only in the appropriate global field if the schema supports that.
- Skip normal levels.

Patient gender:
Scan the report for gender indicators such as:
- male, female
- he, she, his, her
- Mr, Mrs
- M, F in header text
Output patient.gender as one of:
- "male"
- "female"
- "unknown"

Per-level output:
For each abnormal level, output:
- level
- abnormalities
- joints

Each abnormality must include:
- type
- severity
- size_mm
- laterality
- region
- notes

Valid type values:
- annular_bulge
- disc_bulge
- protrusion
- extrusion
- stenosis
- foraminal_narrowing
- facet_arthropathy
- alignment
- fracture
- edema
- cord_compression
- nerve_root_impingement
- disc_height_loss
- other

Valid severity values:
- none
- mild
- moderate
- severe
- unknown

Valid laterality values:
- left
- right
- bilateral
- midline
- unknown

Valid region values:
- central
- paracentral
- foraminal
- extraforaminal
- unknown

Type normalization rules:
Map report wording to these normalized types:
- "disc space narrowing", "loss of disc height", "disc collapse" -> disc_height_loss
- "broad-based bulge", "diffuse bulge", "circumferential bulge" -> disc_bulge
- "focal protrusion", "broad-based protrusion" -> protrusion
- "herniation", "disc extrusion", "sequestration" -> extrusion
- "central canal narrowing", "spinal stenosis" -> stenosis
- "neural foraminal narrowing", "foraminal stenosis" -> foraminal_narrowing

Joint output:
Each level must contain a joints object with exactly these 5 keys:
- joint_1
- joint_2
- joint_3
- joint_4
- joint_center

Each key must contain an array of exactly 3 numbers:
[x, y, z]

Axis meanings:
- x = left-right width change
- y = height/thickness change
- z = anterior-posterior depth or protrusion change

These are OFFSET values only.
Do not output final scale values.
The server adds 1.0 afterward.
Examples:
- 0.10 means final scale 1.10
- -0.08 means final scale 0.92

Numeric constraints:
- Every joint axis value must be between -0.5 and 0.5
- Use conservative values
- Most corner-joint values should usually remain in the range 0.03 to 0.15 unless the report clearly supports something larger

Anatomic joint meaning:
- joint_1 = top-right corner of the disc
- joint_2 = bottom-right corner of the disc
- joint_3 = bottom-left corner of the disc
- joint_4 = top-left corner of the disc
- joint_center = the PARENT bone of the entire disc

CRITICAL — joint_center is a parent bone:
- joint_center controls the ENTIRE disc including all four corners.
- When joint_center is scaled, ALL child bones (joint_1 through joint_4) are scaled WITH it.
- This means joint_center values COMPOUND with corner values.
- Example: if joint_center z = 0.10 and joint_1 z = 0.10, the effective scale on joint_1 z is 1.10 * 1.10 = 1.21 (not 1.20).
- Therefore: when you put a value on joint_center, REDUCE or ZERO OUT the same axis on corners to avoid double-scaling.
- Use joint_center for uniform whole-disc changes (disc height loss, central bulges, stenosis).
- Use corners ONLY for the focal/lateralized component that differs from the uniform change.
- If a pathology is purely central or uniform, put it ONLY on joint_center and leave corners at zero.

Laterality to corners:
- left -> joint_3 and joint_4
- right -> joint_1 and joint_2
- bilateral -> all four corners
- midline or unknown -> use joint_center only, leave corners at zero

Disc naming note:
For any downstream disc naming convention based on top vertebra:
- L4-L5 corresponds to L4
- T12-L1 corresponds to T12
Do not change the reported level text in the JSON unless the schema explicitly asks for both forms.

Pathology-to-morph mapping:
Use the following table conservatively.

disc_bulge or annular_bulge
- central -> joint_center z
  - mild 0.04
  - moderate 0.08
  - severe 0.12
- paracentral right -> joint_1 and joint_2 z
  - mild 0.04 to 0.08
  - add joint_center z at half of the corner value
- paracentral left -> joint_3 and joint_4 z
  - mild 0.04 to 0.08
  - add joint_center z at half of the corner value
- foraminal right -> joint_1 and joint_2 z
  - mild 0.05
  - moderate 0.10
  - severe 0.15
- foraminal left -> joint_3 and joint_4 z
  - mild 0.05
  - moderate 0.10
  - severe 0.15

protrusion
- central -> joint_center z
  - mild 0.06
  - moderate 0.12
  - severe 0.18
- paracentral right -> joint_1 and joint_2 z
  - mild to severe 0.06 to 0.15
  - add joint_center z at 60 percent of corner value
- paracentral left -> joint_3 and joint_4 z
  - mild to severe 0.06 to 0.15
  - add joint_center z at 60 percent of corner value
- foraminal on affected side -> affected corners z
  - mild 0.08
  - moderate 0.15
  - severe 0.22

extrusion
- central -> joint_center z
  - mild 0.10
  - moderate 0.18
  - severe 0.28
- paracentral right -> joint_1 and joint_2 z
  - 0.10 to 0.22
  - add joint_center z at half of corner value
- paracentral left -> joint_3 and joint_4 z
  - 0.10 to 0.22
  - add joint_center z at half of corner value
- foraminal on affected side -> affected corners z
  - mild 0.12
  - moderate 0.20
  - severe 0.30
- For clearly large extrusions, you may also add a small positive y value of 0.02 to 0.05 on affected corners only if the report wording strongly supports vertical displacement

disc_height_loss
- Use joint_center only
- Use y only
- Always negative
- mild -0.08
- moderate -0.18
- severe -0.30
- Never apply disc height loss to corner joints

stenosis
- Use joint_center z
- mild 0.04
- moderate 0.08
- severe 0.14

foraminal_narrowing
- Use affected side corner x values as inward squeeze
- mild -0.04
- moderate -0.08
- severe -0.14

cord_compression
- Use joint_center z
- moderate 0.15
- severe 0.25

nerve_root_impingement
- Use affected side corner z
- mild 0.06
- moderate 0.12
- severe 0.20

facet_arthropathy
- Use affected side corner x as widening
- mild 0.03
- moderate 0.06
- severe 0.10

Size calibration:
If size_mm is explicitly provided, you may calibrate the primary morph axis using:
(size_mm / 35.0) * 0.5

Example:
- 6 mm protrusion -> about 0.086 on the primary z axis

Use size_mm as a calibration guide, but do not exceed the pathology table by a large amount unless the report clearly justifies it.

Conflict resolution:
If multiple instructions could apply:
1. Prefer explicit report wording over all mapping defaults.
2. Prefer conservative values over aggressive values.
3. Prefer exact laterality and region when stated.
4. If the report is vague, keep the abnormality but use "unknown" fields and modest morph values.
5. If a level is described as normal, do not output it.

Final output rules:
- Return only valid JSON.
- The JSON must match the provided schema exactly.
- Do not include markdown.
- Do not include explanation.
- Do not include prose outside the JSON.
"""


def _get_client(api_key: Optional[str] = None) -> Anthropic:
    key = api_key or ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("missing_anthropic_api_key")
    return Anthropic(api_key=key)


def deid_report(text: str, *, model: str = DEID_MODEL, api_key: Optional[str] = None) -> str:
    text = insert_field_breaks(text)
    client = _get_client(api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=_DEID_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text


def extract_morph(text: str, *, model: str = EXTRACT_MODEL, api_key: Optional[str] = None) -> _CombinedExtraction:
    client = _get_client(api_key)
    resp = client.beta.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        betas=["structured-outputs-2025-11-13"],
        system=_EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": text}],
        output_format={"type": "json_schema", "schema": transform_schema(_CombinedExtraction)},
    )
    return _CombinedExtraction.model_validate_json(resp.content[0].text)


def to_api_payload(combined: _CombinedExtraction, gender_override: str = "unknown", warnings: Optional[List[str]] = None) -> MorphResponse:
    targets: List[DiscTarget] = []

    for lvl in combined.levels:
        name = disc_name(lvl.level)
        j = lvl.joints

        c1 = validate_axis(j.joint_1, MAX_CORNER)
        c2 = validate_axis(j.joint_2, MAX_CORNER)
        c3 = validate_axis(j.joint_3, MAX_CORNER)
        c4 = validate_axis(j.joint_4, MAX_CORNER)
        cc = validate_axis(j.joint_center, MAX_SCALER)

        all_vals = [c1, c2, c3, c4, cc]
        if all(max(abs(a - 1.0) for a in axes) <= 0.001 for axes in all_vals):
            continue

        targets.append(DiscTarget(
            disc=name,
            index=DISC_INDEX.get(name),
            joint_1=c1,
            joint_2=c2,
            joint_3=c3,
            joint_4=c4,
            joint_center=cc,
        ))

    patient = combined.patient
    if patient.gender == "unknown" and gender_override != "unknown":
        patient = Patient(gender=gender_override) # type: ignore

    return MorphResponse(
        patient=patient,
        targets=targets,
        meta={"kept_levels": [t.disc for t in targets]},
        warnings=list(warnings or []),
    )


def process_report_to_payload(
    raw_report: str,
    *,
    use_deid: bool = True,
    deid_model: str = DEID_MODEL,
    extract_model: str = EXTRACT_MODEL,
    api_key: Optional[str] = None,
) -> MorphResponse:
    warnings: List[str] = []
    gender = detect_gender(raw_report)

    if use_deid:
        try:
            text = deid_report(raw_report, model=deid_model, api_key=api_key)
        except Exception as e:
            warnings.append(f"deid failed, using raw text: {e}")
            text = insert_field_breaks(raw_report)
    else:
        text = insert_field_breaks(raw_report)

    combined = extract_morph(text, model=extract_model, api_key=api_key)
    return to_api_payload(combined, gender_override=gender, warnings=warnings)
