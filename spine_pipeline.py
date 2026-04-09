from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, transform_schema
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from extract import DeidResult, deidentify_report, extract_report
from schemas import Abnormality, DiscMorph, GlobalFindings, MorphResponse, Patient


class _LevelMorph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str
    abnormalities: List[Abnormality] = Field(default_factory=list)
    morph: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])


class _CombinedExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient: Patient = Field(default_factory=Patient)
    levels: List[_LevelMorph] = Field(default_factory=list)
    global_findings: GlobalFindings = Field(default_factory=GlobalFindings)


_SYSTEM = (
    "You are a board-certified spine radiologist extracting findings from MRI/CT reports "
    "and producing disc morph values for a UE5 skeletal mesh visualization.\n\n"

    "ACCURACY RULES (most important):\n"
    "1. Extract ONLY findings explicitly stated in the report text. Do NOT infer or invent.\n"
    "2. Skip levels described as normal, unremarkable, or 'no significant abnormality'.\n"
    "3. Quote-test: every abnormality you output must correspond to a phrase in the report.\n"
    "4. If severity is not stated, use 'unknown' — do not guess.\n"
    "5. If laterality is not stated, use 'unknown' — do not assume bilateral.\n"
    "6. If size is not given numerically, leave size_mm null.\n"
    "7. When in doubt, output fewer findings with lower morph values.\n\n"

    "OUTPUT per abnormal level:\n"
    "- level: format like L4-L5 or T12-L1\n"
    "- abnormalities: each with type, severity, size_mm, laterality, region, notes\n"
    "- morph: exactly 5 floats [joint1_topright, joint2_bottomright, joint3_bottomleft, "
    "joint4_topleft, scaler]. Range -1.0 to 1.0.\n\n"

    "GENDER: scan the report for gender indicators (Mr/Mrs, he/she, his/her, male/female, "
    "M/F in patient header). Output patient.gender as 'male', 'female', or 'unknown'.\n\n"

    "UE5 BONE SCALE CONTEXT:\n"
    "- Each morph value is ADDED to 1.0 to become the bone scale.\n"
    "- Corner joints (indices 0-3) scale ALL 3 AXES simultaneously.\n"
    "  A value of 0.2 → scale 1.2 per axis → 1.2^3 ≈ 1.73x volume.\n"
    "  A value of 0.3 → scale 1.3 per axis → 1.3^3 ≈ 2.2x volume — already very large.\n"
    "  KEEP corner values small. Most should be 0.03-0.15.\n"
    "- Scaler (index 4) scales the disc HEIGHT ONLY (single Y axis).\n"
    "  Positive = taller/thicker disc. Negative = shorter/thinner disc.\n\n"

    "PATHOLOGY-TO-MORPH MAPPING TABLE:\n\n"

    "disc_bulge / annular_bulge:\n"
    "  central → scaler only: mild 0.04, moderate 0.08, severe 0.12\n"
    "  paracentral → affected corners 0.04-0.10 + scaler at half\n"
    "  foraminal → affected corners only: mild 0.05, moderate 0.10, severe 0.15\n\n"

    "protrusion:\n"
    "  central → scaler: mild 0.06, moderate 0.12, severe 0.18\n"
    "  paracentral → affected corners 0.06-0.15 + scaler at 60%\n"
    "  foraminal → affected corners: mild 0.08, moderate 0.15, severe 0.22\n\n"

    "extrusion:\n"
    "  central → scaler: mild 0.10, moderate 0.18, severe 0.28\n"
    "  paracentral → affected corners 0.10-0.22 + scaler at half\n"
    "  foraminal → affected corners: mild 0.12, moderate 0.20, severe 0.30\n\n"

    "disc_height_loss → SCALER ONLY (always negative): mild -0.08, moderate -0.18, severe -0.30\n"
    "  NEVER apply disc height loss to corner joints. It is uniform disc thinning.\n\n"

    "stenosis → scaler only: mild 0.04, moderate 0.08, severe 0.14\n"
    "foraminal_narrowing → affected corners: mild 0.04, moderate 0.08, severe 0.14\n"
    "cord_compression → scaler: moderate 0.15, severe 0.25\n"
    "nerve_root_impingement → affected corners: mild 0.06, moderate 0.12, severe 0.20\n"
    "facet_arthropathy → affected corners: mild 0.03, moderate 0.06, severe 0.10\n\n"

    "SIZE CALIBRATION (when size_mm is available):\n"
    "  corner value ≈ (size_mm / 35.0) × 0.5\n"
    "  Example: 6mm protrusion → (6/35) × 0.5 ≈ 0.086\n\n"

    "LATERALITY → corner indices:\n"
    "  left → indices 2 (bottom-left) and 3 (top-left)\n"
    "  right → indices 0 (top-right) and 1 (bottom-right)\n"
    "  bilateral → all four corners at full magnitude\n"
    "  midline/unknown → all four at 60% magnitude\n\n"

    "DISC NAMING: TOP vertebra. L4-L5 → L4. T12-L1 → T12.\n\n"

    "TYPE NORMALIZATION:\n"
    "- 'disc space narrowing', 'loss of disc height', 'disc collapse' → disc_height_loss\n"
    "- 'broad-based bulge', 'diffuse bulge', 'circumferential bulge' → disc_bulge\n"
    "- 'focal protrusion', 'broad-based protrusion' → protrusion\n"
    "- 'herniation', 'disc extrusion', 'sequestration' → extrusion\n"
    "- 'central canal narrowing', 'spinal stenosis' → stenosis\n"
    "- 'neural foraminal narrowing', 'foraminal stenosis' → foraminal_narrowing\n\n"

    "GLOBAL FINDINGS:\n"
    "- Set cord_compression true ONLY if report explicitly states cord compression or "
    "cord flattening with signal change.\n"
    "- Set nerve_root_impingement true if any level has nerve root impingement.\n"
    "- Put incidental non-spine findings in global_findings.incidental.\n"
    "- Put alignment notes in global_findings.alignment_notes.\n\n"

    "Valid types: annular_bulge, disc_bulge, protrusion, extrusion, stenosis, "
    "foraminal_narrowing, facet_arthropathy, alignment, fracture, edema, "
    "cord_compression, nerve_root_impingement, disc_height_loss, other\n"
    "Valid severities: none, mild, moderate, severe, unknown\n"
    "Valid lateralities: left, right, bilateral, midline, unknown\n"
    "Valid regions: central, paracentral, foraminal, extraforaminal, unknown\n\n"

    "Return ONLY JSON matching the schema.\n"
)


_JSON_RE = re.compile(r"(?s)\{.*\}\s*$")

_DISC_INDEX: Dict[str, int] = {
    "C1": 0,  "C2": 1,  "C3": 2,  "C4": 3,  "C5": 4,  "C6": 5,  "C7": 6,
    "T1": 7,  "T2": 8,  "T3": 9,  "T4": 10, "T5": 11, "T6": 12,
    "T7": 13, "T8": 14, "T9": 15, "T10": 16, "T11": 17, "T12": 18,
    "L1": 19, "L2": 20, "L3": 21, "L4": 22, "L5": 23,
}

MAX_CORNER = 0.5
MAX_SCALER = 0.5


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _disc_name(level: str) -> str:
    s = (level or "").strip().upper().replace("\u2013", "-").replace("\u2014", "-").replace(" ", "")
    reps = {
        "T2-T3": "T2-3", "T3-T4": "T3-4", "T4-T5": "T4-5", "T5-T6": "T5-6",
        "T6-T7": "T6-7", "T7-T8": "T7-8", "T8-T9": "T8-9", "T9-T10": "T9-10",
        "T10-T11": "T10-11", "T11-T12": "T11-12",
    }
    lv = reps.get(s, s)
    return lv.split("-", 1)[0] if "-" in lv else lv


def _validate_morph(values: List[float]) -> List[float]:
    if len(values) != 5:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    corners = [round(_clamp(v, -MAX_CORNER, MAX_CORNER), 4) for v in values[:4]]
    scaler = round(_clamp(values[4], -MAX_SCALER, MAX_SCALER), 4)
    return corners + [scaler]


def ask_combined(report: str, *, model: str, api_key: Optional[str] = None) -> _CombinedExtraction:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("missing_anthropic_api_key")

    client = Anthropic(api_key=key)
    resp = client.beta.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        betas=["structured-outputs-2025-11-13"],
        system=_SYSTEM,
        messages=[{"role": "user", "content": report}],
        output_format={"type": "json_schema", "schema": transform_schema(_CombinedExtraction)},
    )
    raw = resp.content[0].text
    try:
        return _CombinedExtraction.model_validate_json(raw)
    except ValidationError:
        m = _JSON_RE.search((raw or "").strip())
        if m is None:
            raise RuntimeError("AI did not return valid JSON")
        return _CombinedExtraction.model_validate(json.loads(m.group(0)))


def to_api_payload(combined: _CombinedExtraction, warnings: Optional[List[str]] = None) -> MorphResponse:
    targets: List[DiscMorph] = []

    for lvl in combined.levels:
        clamped = _validate_morph(lvl.morph)
        if max(abs(v) for v in clamped) <= 0.001:
            continue
        disc = _disc_name(lvl.level)
        targets.append(DiscMorph(name=disc, index=_DISC_INDEX.get(disc), values=clamped))

    meta: Dict[str, Any] = {"kept_levels": [t.name for t in targets]}

    return MorphResponse(
        patient=combined.patient,
        morph_targets=targets,
        global_findings=combined.global_findings,
        meta=meta,
        warnings=list(warnings or []),
    )


def process_report_to_payload(
    raw_report: str,
    *,
    deid_with_ai: bool = True,
    deid_model: str = "gpt-5-mini",
    deid_api_key: Optional[str] = None,
    extract_model: str = "claude-sonnet-4-6",
    anthropic_api_key: Optional[str] = None,
) -> MorphResponse:
    deid: DeidResult = deidentify_report(raw_report, use_ai=deid_with_ai, model=deid_model, api_key=deid_api_key)
    text = extract_report(deid.text)
    combined = ask_combined(text, model=extract_model, api_key=anthropic_api_key)
    return to_api_payload(combined, warnings=deid.warnings)
