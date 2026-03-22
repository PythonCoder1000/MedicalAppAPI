from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from anthropic import Anthropic, transform_schema
from pydantic import ValidationError

from extract import DeidResult, deidentify_report, extract_report
from schemas import Abnormality, DiscMorph, ExtractedJson, MorphResponse

_SPINE_EXTRACT_INSTRUCTIONS = (
    "Return ONLY JSON matching the schema.\n"
    "Extract ONLY abnormal spine levels explicitly described.\n"
    "Do NOT output normal levels.\n"
    "Level format MUST be like T12-L1.\n"
    "For each abnormal level, output abnormalities with fields type, severity, size_mm, laterality, region, notes.\n"
    "Valid region values are only central, paracentral, foraminal, extraforaminal, unknown.\n"
    "Valid laterality values are only left, right, bilateral, midline, unknown.\n"
    "Use type='disc_height_loss' when the report states disc height loss, disc space narrowing, or disc collapse.\n"
    "If report says no cord compression or no nerve root impingement, set those globals false.\n"
    "Put non-spine incidental findings into global_findings.incidental.\n"
    "Put alignment notes into global_findings.alignment_notes.\n"
)

_MORPH_INSTRUCTIONS = """You are a spine radiologist and biomechanical expert.

Given structured spine pathology findings and a heuristic baseline, produce accurate disc morphing displacement values for a 3D visualization. Use web search to look up realistic displacement magnitudes for specific pathology types and severities when needed.

Joint layout for each disc (5 values, range -1.0 to 1.0):
  index 0: joint1 = top-right corner
  index 1: joint2 = bottom-right corner
  index 2: joint3 = bottom-left corner
  index 3: joint4 = top-left corner
  index 4: scaler = uniform whole-disc size change

Rules:
- Positive = expansion/protrusion outward, negative = compression/collapse
- Laterality determines which corner joints are affected (left → indices 2,3 / right → indices 0,1 / bilateral → all)
- Region determines distribution (central → scaler / foraminal → corner joints / paracentral → both)
- Severity and size_mm calibrate magnitude
- Disc name must use the bottom vertebra (L4-L5 → "L5", T12-L1 → "L1")

The heuristic_baseline is a reasonable starting point — improve it using real biomechanical data.

Return ONLY valid JSON:
{"morph_targets": {"DISC": [j1, j2, j3, j4, scaler], ...}}"""

_JSON_ONLY_RE = re.compile(r"(?s)\{.*\}\s*$")


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def normalize_level(level: str) -> str:
    s = (level or "").strip().upper().replace("–", "-").replace("—", "-").replace(" ", "")
    replacements = {
        "T2-T3": "T2-3", "T3-T4": "T3-4", "T4-T5": "T4-5", "T5-T6": "T5-6",
        "T6-T7": "T6-7", "T7-T8": "T7-8", "T8-T9": "T8-9", "T9-T10": "T9-10",
        "T10-T11": "T10-11", "T11-T12": "T11-12",
    }
    return replacements.get(s, s)


def _disc_name(level: str) -> str:
    lv = normalize_level(level)
    return lv.split("-", 1)[0] if "-" in lv else lv


def _sev_to_pos(sev: str) -> float:
    return {"none": 0.0, "mild": 0.3, "moderate": 0.6, "severe": 1.0}.get(sev, 0.25)


def _sev_to_height_loss(sev: str) -> float:
    return {"none": 0.0, "mild": 0.2, "moderate": 0.35, "severe": 0.5}.get(sev, 0.25)


def _vert_from_notes(notes: str) -> str:
    s = (notes or "").lower()
    top = any(w in s for w in ("superior", "upper", "cranial", "cephalad", "top"))
    bot = any(w in s for w in ("inferior", "lower", "caudal", "caudad", "bottom"))
    if top and not bot:
        return "top"
    if bot and not top:
        return "bottom"
    return "both"


def _apply_side(joints: List[float], lat: str, v: float, which: str) -> None:
    if lat == "left":
        if which in ("top", "both"):
            joints[3] = _clamp(joints[3] + v)
        if which in ("bottom", "both"):
            joints[2] = _clamp(joints[2] + v)
    elif lat == "right":
        if which in ("top", "both"):
            joints[0] = _clamp(joints[0] + v)
        if which in ("bottom", "both"):
            joints[1] = _clamp(joints[1] + v)
    elif lat == "bilateral":
        if which in ("top", "both"):
            joints[0] = _clamp(joints[0] + v)
            joints[3] = _clamp(joints[3] + v)
        if which in ("bottom", "both"):
            joints[1] = _clamp(joints[1] + v)
            joints[2] = _clamp(joints[2] + v)
    else:
        v2 = v * 0.7
        if which in ("top", "both"):
            joints[0] = _clamp(joints[0] + v2)
            joints[3] = _clamp(joints[3] + v2)
        if which in ("bottom", "both"):
            joints[1] = _clamp(joints[1] + v2)
            joints[2] = _clamp(joints[2] + v2)


def compute_joint_moves(abns: List[Abnormality]) -> List[float]:
    joints = [0.0, 0.0, 0.0, 0.0, 0.0]

    for a in abns:
        lat = a.laterality or "unknown"
        region = a.region or "unknown"
        which = _vert_from_notes(a.notes)

        if a.type == "disc_height_loss":
            mag = _sev_to_height_loss(a.severity)
            joints[4] = _clamp(joints[4] + (-mag * 0.3))
            _apply_side(joints, lat if lat != "midline" else "unknown", -mag, which)
            continue

        sev_w = _sev_to_pos(a.severity)
        size_w = _clamp(float(a.size_mm) / 5.0) if isinstance(a.size_mm, (int, float)) else None

        if a.type in ("annular_bulge", "disc_bulge"):
            w = size_w if size_w is not None else max(sev_w, 0.2)
            if region in ("foraminal", "extraforaminal"):
                _apply_side(joints, lat, w, which)
            elif region == "paracentral":
                joints[4] = _clamp(joints[4] + w * 0.8)
                _apply_side(joints, lat, w * 0.8 if lat in ("left", "right", "bilateral") else w * 0.5, which)
            else:
                joints[4] = _clamp(joints[4] + w)
            continue

        if a.type == "stenosis":
            joints[4] = _clamp(joints[4] + sev_w)
            continue

        if a.type == "foraminal_narrowing":
            _apply_side(joints, lat, sev_w if sev_w > 0 else 0.2, which)
            continue

        if a.type in ("protrusion", "extrusion"):
            w = size_w if size_w is not None else max(sev_w, 0.4)
            if region in ("central", "unknown"):
                joints[4] = _clamp(joints[4] + w)
            elif region == "paracentral":
                joints[4] = _clamp(joints[4] + w * 0.75)
                _apply_side(joints, lat, w, which)
            else:
                _apply_side(joints, lat, w, which)
            continue

        if a.type == "facet_arthropathy":
            _apply_side(joints, lat, max(sev_w, 0.2), which)
            continue

        if a.type == "cord_compression":
            joints[4] = _clamp(joints[4] + max(sev_w, 0.8))
            continue

        if a.type == "nerve_root_impingement":
            _apply_side(joints, lat, max(sev_w, 0.7), which)

    return [round(_clamp(v), 4) for v in joints]


def ask_spine_model(report: str, *, model: str, api_key: Optional[str] = None) -> ExtractedJson:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("missing_anthropic_api_key")

    client = Anthropic(api_key=key)
    resp = client.beta.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        betas=["structured-outputs-2025-11-13"],
        system=_SPINE_EXTRACT_INSTRUCTIONS,
        messages=[{"role": "user", "content": report}],
        output_format={"type": "json_schema", "schema": transform_schema(ExtractedJson)},
    )
    raw = resp.content[0].text
    try:
        return ExtractedJson.model_validate_json(raw)
    except ValidationError:
        m = _JSON_ONLY_RE.search((raw or "").strip())
        if m is None:
            raise RuntimeError("AI did not return valid JSON")
        return ExtractedJson.model_validate(json.loads(m.group(0)))


def ask_morph_model(
    extracted: ExtractedJson,
    heuristic: Dict[str, List[float]],
    *,
    model: str,
    api_key: Optional[str] = None,
) -> tuple[Dict[str, List[float]], str]:
    client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_MORPH_INSTRUCTIONS,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        messages=[{
            "role": "user",
            "content": json.dumps({
                "findings": extracted.model_dump(exclude_defaults=True),
                "heuristic_baseline": heuristic,
            }, indent=2),
        }],
    )

    text = next((b.text for b in resp.content if hasattr(b, "text")), "")
    m = _JSON_ONLY_RE.search(text.strip())
    if not m:
        return heuristic, "heuristic"

    try:
        raw = json.loads(m.group(0)).get("morph_targets", {})
        refined = {
            k: [round(_clamp(float(v)), 4) for v in vals]
            for k, vals in raw.items()
            if isinstance(vals, list) and len(vals) == 5
        }
        return (refined, "ai") if refined else (heuristic, "heuristic")
    except (json.JSONDecodeError, ValueError, TypeError):
        return heuristic, "heuristic"


def to_api_payload(
    extracted: ExtractedJson,
    morph_targets: Dict[str, List[float]],
    morph_source: str = "heuristic",
    warnings: Optional[List[str]] = None,
) -> MorphResponse:
    meta = dict(extracted.meta) if isinstance(extracted.meta, dict) else {}
    meta["kept_levels"] = list(morph_targets.keys())
    meta["morph_source"] = morph_source

    return MorphResponse(
        morph_targets=[DiscMorph(name=k, values=v) for k, v in morph_targets.items()],
        global_findings=extracted.global_findings,
        meta=meta,
        warnings=list(warnings or []),
    )


def process_report_to_payload(
    raw_report: str,
    *,
    deid_with_ai: bool = True,
    deid_model: str = "gpt-5-mini",
    deid_api_key: Optional[str] = None,
    extract_model: str = "claude-sonnet-4-5",
    anthropic_api_key: Optional[str] = None,
) -> MorphResponse:
    deid: DeidResult = deidentify_report(raw_report, use_ai=deid_with_ai, model=deid_model, api_key=deid_api_key)
    extracted_text = extract_report(deid.text)
    extracted = ask_spine_model(extracted_text, model=extract_model, api_key=anthropic_api_key)

    heuristic: Dict[str, List[float]] = {
        _disc_name(lvl.level): compute_joint_moves(lvl.abnormalities)
        for lvl in extracted.levels
        if max(abs(v) for v in compute_joint_moves(lvl.abnormalities)) > 0.0
    }

    morph_targets, morph_source = ask_morph_model(extracted, heuristic, model=extract_model, api_key=anthropic_api_key)

    return to_api_payload(extracted, morph_targets, morph_source=morph_source, warnings=deid.warnings)
