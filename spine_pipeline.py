from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, transform_schema
from pydantic import ValidationError

from extract import DeidResult, deidentify_report, extract_report
from schemas import Abnormality, ExtractedJson, JointName, MorphResponse, UnrealDiscJointJson, UnrealDiscJointLevel


_SPINE_EXTRACT_INSTRUCTIONS = (
    "Return ONLY valid JSON matching the schema hint.\n"
    "Extract ONLY abnormal spine levels explicitly described.\n"
    "Do NOT output normal levels.\n"
    "Level format MUST be like T2-3 or T12-L1.\n"
    "For each abnormal level, output abnormalities with fields type, severity, size_mm, laterality, region, notes.\n"
    "Valid region values are only central, paracentral, foraminal, extraforaminal, unknown.\n"
    "If unsure, set region to unknown.\n"
    "Valid laterality values are only left, right, bilateral, midline, unknown.\n"
    "If report says no cord compression or no nerve root impingement, set those globals false.\n"
    "Put non-spine incidental findings into global_findings.incidental.\n"
    "Put alignment/kyphosis/lordosis notes into global_findings.alignment_notes.\n"
)

_SPINE_SCHEMA_HINT = {
    "type": "object",
    "required": ["levels", "global_findings", "meta"],
    "properties": {
        "levels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["level", "abnormalities"],
                "properties": {
                    "level": {"type": "string"},
                    "abnormalities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "severity", "notes"],
                            "properties": {
                                "type": {"type": "string"},
                                "severity": {"type": "string"},
                                "size_mm": {"type": ["number", "null"]},
                                "laterality": {"type": ["string", "null"]},
                                "region": {"type": ["string", "null"]},
                                "notes": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "global_findings": {
            "type": "object",
            "required": ["cord_compression", "nerve_root_impingement", "alignment_notes", "incidental"],
            "properties": {
                "cord_compression": {"type": "boolean"},
                "nerve_root_impingement": {"type": "boolean"},
                "alignment_notes": {"type": "string"},
                "incidental": {"type": "array", "items": {"type": "string"}},
            },
        },
        "meta": {"type": "object"},
    },
}


def normalize_level(level: str) -> str:
    s = (level or "").strip().upper().replace("–", "-").replace("—", "-").replace(" ", "")
    replacements = {
        "T2-T3": "T2-3",
        "T3-T4": "T3-4",
        "T4-T5": "T4-5",
        "T5-T6": "T5-6",
        "T6-T7": "T6-7",
        "T7-T8": "T7-8",
        "T8-T9": "T8-9",
        "T9-T10": "T9-10",
        "T10-T11": "T10-11",
        "T11-T12": "T11-12",
    }
    return replacements.get(s, s)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _sev_to_weight(sev: str) -> float:
    if sev == "none":
        return 0.0
    if sev == "mild":
        return 0.3
    if sev == "moderate":
        return 0.6
    if sev == "severe":
        return 0.9
    return 0.2


def _empty_joint_map() -> Dict[JointName, float]:
    return {
        "center": 0.0,
        "topright": 0.0,
        "bottomright": 0.0,
        "topleft": 0.0,
        "bottomleft": 0.0,
    }


def _max_set(d: Dict[JointName, float], key: JointName, value: float) -> None:
    d[key] = _clamp01(max(float(d.get(key, 0.0)), float(value)))


def _apply_side_pair(d: Dict[JointName, float], side: str, value: float) -> None:
    if side == "left":
        _max_set(d, "topleft", value)
        _max_set(d, "bottomleft", value)
        return
    if side == "right":
        _max_set(d, "topright", value)
        _max_set(d, "bottomright", value)
        return
    if side == "bilateral":
        _max_set(d, "topleft", value)
        _max_set(d, "bottomleft", value)
        _max_set(d, "topright", value)
        _max_set(d, "bottomright", value)
        return
    _max_set(d, "topleft", value * 0.7)
    _max_set(d, "bottomleft", value * 0.7)
    _max_set(d, "topright", value * 0.7)
    _max_set(d, "bottomright", value * 0.7)


def compute_joint_moves(abns: List[Abnormality]) -> Dict[JointName, float]:
    joints = _empty_joint_map()

    for a in abns:
        sev_w = _sev_to_weight(a.severity)
        size_w = _clamp01(float(a.size_mm) / 5.0) if isinstance(a.size_mm, (int, float)) else None
        lat = a.laterality or "unknown"
        region = a.region or "unknown"

        if a.type in ("annular_bulge", "disc_bulge"):
            w = size_w if size_w is not None else max(sev_w, 0.2)
            if region in ("foraminal", "extraforaminal"):
                _apply_side_pair(joints, lat, w)
            elif region == "paracentral":
                _max_set(joints, "center", w * 0.8)
                _apply_side_pair(joints, lat, w * 0.8 if lat in ("left", "right", "bilateral") else w * 0.5)
            else:
                _max_set(joints, "center", w)
            continue

        if a.type == "stenosis":
            _max_set(joints, "center", sev_w)
            continue

        if a.type == "foraminal_narrowing":
            _apply_side_pair(joints, lat, sev_w if sev_w > 0 else 0.2)
            continue

        if a.type in ("protrusion", "extrusion"):
            w = size_w if size_w is not None else max(sev_w, 0.4)
            if region in ("central", "unknown"):
                _max_set(joints, "center", w)
            elif region == "paracentral":
                _max_set(joints, "center", w * 0.75)
                _apply_side_pair(joints, lat, w)
            else:
                _apply_side_pair(joints, lat, w)
            continue

        if a.type == "facet_arthropathy":
            _apply_side_pair(joints, lat, max(sev_w, 0.2))
            continue

        if a.type == "cord_compression":
            _max_set(joints, "center", max(sev_w, 0.8))
            continue

        if a.type == "nerve_root_impingement":
            _apply_side_pair(joints, lat, max(sev_w, 0.7))

    return {k: round(_clamp01(v), 4) for k, v in joints.items()}


def _build_skeletal_mesh_controls(levels: List[UnrealDiscJointLevel]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lvl in levels:
        tag = lvl.level.replace("-", "_")
        for joint_name, amount in lvl.joints.items():
            out[f"disc_{tag}_{joint_name}"] = float(amount)
    return out


def _sanitize_extracted_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "levels": [],
            "global_findings": {
                "cord_compression": False,
                "nerve_root_impingement": False,
                "alignment_notes": "",
                "incidental": [],
            },
            "meta": {},
        }

    out = dict(obj)
    if not isinstance(out.get("levels"), list):
        out["levels"] = []

    for lvl in out["levels"]:
        if not isinstance(lvl, dict):
            continue
        if not isinstance(lvl.get("abnormalities"), list):
            lvl["abnormalities"] = []
        for a in lvl["abnormalities"]:
            if not isinstance(a, dict):
                continue
            if "notes" not in a or a["notes"] is None:
                a["notes"] = ""
            if "size_mm" in a and a["size_mm"] is not None:
                try:
                    a["size_mm"] = float(a["size_mm"])
                except Exception:
                    a["size_mm"] = None

    gf = out.get("global_findings")
    if not isinstance(gf, dict):
        gf = {}
        out["global_findings"] = gf
    if not isinstance(gf.get("cord_compression"), bool):
        gf["cord_compression"] = False
    if not isinstance(gf.get("nerve_root_impingement"), bool):
        gf["nerve_root_impingement"] = False
    if not isinstance(gf.get("alignment_notes"), str):
        gf["alignment_notes"] = ""
    if not isinstance(gf.get("incidental"), list):
        gf["incidental"] = []

    if not isinstance(out.get("meta"), dict):
        out["meta"] = {}

    return out


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
        try:
            obj = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"AI returned invalid JSON: {e}") from e
        fixed = _sanitize_extracted_obj(obj)
        return ExtractedJson.model_validate(fixed)


def to_unreal_joint_ready(extracted: ExtractedJson, allowed_levels: Optional[List[str]] = None) -> UnrealDiscJointJson:
    allowed = None
    if allowed_levels:
        allowed = {normalize_level(x) for x in allowed_levels if isinstance(x, str) and x.strip()}

    unreal_levels: List[UnrealDiscJointLevel] = []
    for lvl in extracted.levels:
        nl = normalize_level(lvl.level)
        if allowed is not None and nl not in allowed:
            continue
        joints = compute_joint_moves(lvl.abnormalities)
        if max(joints.values()) <= 0.0:
            continue
        unreal_levels.append(UnrealDiscJointLevel(level=nl, joints=joints))

    meta = dict(extracted.meta) if isinstance(extracted.meta, dict) else {}
    meta["kept_levels"] = [x.level for x in unreal_levels]
    meta["joint_names"] = ["center", "topright", "bottomright", "topleft", "bottomleft"]

    return UnrealDiscJointJson(
        levels=unreal_levels,
        global_findings=extracted.global_findings,
        meta=meta,
        skeletal_mesh_controls=_build_skeletal_mesh_controls(unreal_levels),
    )


def process_report_to_unreal_joints(
    raw_report: str,
    *,
    allowed_levels: Optional[List[str]] = None,
    deid_with_ai: bool = True,
    deid_model: str = "gpt-5-mini",
    deid_api_key: Optional[str] = None,
    extract_model: str = "claude-sonnet-4-5",
    anthropic_api_key: Optional[str] = None,
) -> tuple[DeidResult, ExtractedJson, UnrealDiscJointJson]:
    deid = deidentify_report(raw_report, use_ai=deid_with_ai, model=deid_model, api_key=deid_api_key)
    extracted_text = extract_report(deid.text)
    extracted = ask_spine_model(extracted_text, model=extract_model, api_key=anthropic_api_key)
    unreal = to_unreal_joint_ready(extracted, allowed_levels=allowed_levels)
    return deid, extracted, unreal


def to_morph_response(unreal: UnrealDiscJointJson, warnings: Optional[List[str]] = None) -> MorphResponse:
    return MorphResponse(
        levels=unreal.levels,
        global_findings=unreal.global_findings,
        meta=unreal.meta,
        skeletal_mesh_controls=unreal.skeletal_mesh_controls,
        warnings=list(warnings or []),
    )
