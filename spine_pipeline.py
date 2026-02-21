from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic, transform_schema
from pydantic import ValidationError

from extract import DeidResult, deidentify_report, extract_report
from schemas import Abnormality, DiscOut, ExtractedJson, JointId, JointName, MorphResponse

JOINT_ID_BY_NAME: Dict[JointName, JointId] = {
    "topright": "joint2",
    "topleft": "joint5",
    "bottomright": "joint6",
    "bottomleft": "joint4",
    "center": "joint7",
}

FEATURE_TO_JOINTS: Dict[str, List[JointId]] = {
    "canal_stenosis": ["joint7"],
    "disc_bulge": ["joint7"],
    "foraminal_narrowing": ["joint2", "joint6", "joint5", "joint4"],
    "foraminal_narrowing_left": ["joint5", "joint4"],
    "foraminal_narrowing_right": ["joint2", "joint6"],
    "foraminal_narrowing_bilateral": ["joint2", "joint6", "joint5", "joint4"],
    "foraminal_narrowing_unknown": ["joint2", "joint6", "joint5", "joint4"],
}

_SPINE_EXTRACT_INSTRUCTIONS = (
    "Return ONLY JSON matching the schema.\n"
    "Extract ONLY abnormal spine levels explicitly described.\n"
    "Do NOT output normal levels.\n"
    "Level format MUST be like T2-3 or T12-L1.\n"
    "For each abnormal level, output abnormalities with fields type, severity, size_mm, laterality, region, notes.\n"
    "Valid region values are only central, paracentral, foraminal, extraforaminal, unknown.\n"
    "If unsure, set region to unknown.\n"
    "Valid laterality values are only left, right, bilateral, midline, unknown.\n"
    "If report says no cord compression or no nerve root impingement, set those globals false.\n"
    "Put non-spine incidental findings into global_findings.incidental.\n"
    "Put alignment notes into global_findings.alignment_notes.\n"
)

_JSON_ONLY_RE = re.compile(r"(?s)\{.*\}\s*$")


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


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


def split_level_to_bones(level: str) -> Tuple[str, str]:
    lv = normalize_level(level)
    if "-" not in lv:
        return lv, lv
    a, b = lv.split("-", 1)
    if b and b[0].isdigit():
        prefix = ""
        for ch in a:
            if ch.isalpha():
                prefix += ch
            else:
                break
        b = prefix + b
    return a, b


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


def _semantic_to_joint_ids(semantic: Dict[JointName, float]) -> Dict[JointId, float]:
    out: Dict[JointId, float] = {}
    for k, v in semantic.items():
        out[JOINT_ID_BY_NAME[k]] = float(v)
    return out


def _compute_fancy_feature_weights(abns: List[Abnormality]) -> Dict[str, float]:
    disc_bulge_mm: Optional[float] = None
    canal_weight: Optional[float] = None
    foram_left: Optional[float] = None
    foram_right: Optional[float] = None
    foram_any: Optional[float] = None

    for a in abns:
        if a.type in ("annular_bulge", "disc_bulge"):
            if isinstance(a.size_mm, (int, float)):
                x = float(a.size_mm)
                if disc_bulge_mm is None or x > disc_bulge_mm:
                    disc_bulge_mm = x

        if a.type == "stenosis":
            w = _sev_to_weight(a.severity)
            canal_weight = w if canal_weight is None else max(canal_weight, w)

        if a.type == "foraminal_narrowing":
            w = _sev_to_weight(a.severity)
            lat = a.laterality or "unknown"
            if lat == "left":
                foram_left = w if foram_left is None else max(foram_left, w)
            elif lat == "right":
                foram_right = w if foram_right is None else max(foram_right, w)
            elif lat == "bilateral":
                foram_any = w if foram_any is None else max(foram_any, w)
            else:
                foram_any = w if foram_any is None else max(foram_any, w)

    out: Dict[str, float] = {}

    if disc_bulge_mm is not None:
        out["disc_bulge"] = _clamp01(disc_bulge_mm / 5.0)

    if canal_weight is not None:
        out["canal_stenosis"] = _clamp01(canal_weight)

    if foram_left is not None:
        out["foraminal_narrowing_left"] = _clamp01(foram_left)

    if foram_right is not None:
        out["foraminal_narrowing_right"] = _clamp01(foram_right)

    if foram_any is not None:
        out["foraminal_narrowing"] = _clamp01(foram_any)

    return out


def _fancy_weights_to_joint_ids(weights: Dict[str, float]) -> Dict[JointId, float]:
    out: Dict[JointId, float] = {}
    for feature, w in weights.items():
        jids = FEATURE_TO_JOINTS.get(feature)
        if not jids:
            continue
        val = _clamp01(float(w))
        for jid in jids:
            out[jid] = _clamp01(max(float(out.get(jid, 0.0)), val))
    return out


def _merge_joint_id_maps(a: Dict[JointId, float], b: Dict[JointId, float]) -> Dict[JointId, float]:
    out: Dict[JointId, float] = dict(a)
    for k, v in b.items():
        out[k] = _clamp01(max(float(out.get(k, 0.0)), float(v)))
    return out


def _build_morph_targets(level: str, joints: Dict[JointId, float]) -> Dict[str, float]:
    tag = normalize_level(level).replace("-", "_")
    out: Dict[str, float] = {}
    for jid, amount in joints.items():
        out[f"disc_{tag}_{jid}"] = _clamp01(float(amount))
    return out


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    m = _JSON_ONLY_RE.search((s or "").strip())
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


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
        obj = _safe_json_load(raw)
        if obj is None:
            raise RuntimeError("AI did not return valid JSON")
        return ExtractedJson.model_validate(obj)


def to_api_payload(extracted: ExtractedJson, allowed_levels: Optional[List[str]] = None, warnings: Optional[List[str]] = None) -> MorphResponse:
    allowed = None
    if allowed_levels:
        allowed = {normalize_level(x) for x in allowed_levels if isinstance(x, str) and x.strip()}

    discs: List[DiscOut] = []
    morph_targets: Dict[str, float] = {}

    for lvl in extracted.levels:
        nl = normalize_level(lvl.level)
        if allowed is not None and nl not in allowed:
            continue

        semantic = compute_joint_moves(lvl.abnormalities)
        semantic_joints = _semantic_to_joint_ids(semantic)

        fancy_weights = _compute_fancy_feature_weights(lvl.abnormalities)
        fancy_joints = _fancy_weights_to_joint_ids(fancy_weights)

        joints = _merge_joint_id_maps(semantic_joints, fancy_joints)

        if not joints or max(joints.values()) <= 0.0:
            continue

        top_bone, bottom_bone = split_level_to_bones(nl)

        discs.append(
            DiscOut(
                level=nl,
                top_bone=top_bone,
                bottom_bone=bottom_bone,
                joints=joints,
            )
        )

        morph_targets.update(_build_morph_targets(nl, joints))

    meta = dict(extracted.meta) if isinstance(extracted.meta, dict) else {}
    meta["kept_levels"] = [d.level for d in discs]
    meta["joint_map"] = {
        "topright": "joint2",
        "topleft": "joint5",
        "bottomright": "joint6",
        "bottomleft": "joint4",
        "center": "joint7",
    }
    meta["auto_mapped_from_fancy_features"] = {
        "canal_stenosis": ["joint7"],
        "disc_bulge": ["joint7"],
        "foraminal_narrowing": ["joint2", "joint6", "joint5", "joint4"],
        "foraminal_narrowing_left": ["joint5", "joint4"],
        "foraminal_narrowing_right": ["joint2", "joint6"],
    }

    return MorphResponse(
        morph_targets=morph_targets,
        discs=discs,
        global_findings=extracted.global_findings,
        meta=meta,
        warnings=list(warnings or []),
    )


def process_report_to_payload(
    raw_report: str,
    *,
    allowed_levels: Optional[List[str]] = None,
    deid_with_ai: bool = True,
    deid_model: str = "gpt-5-mini",
    deid_api_key: Optional[str] = None,
    extract_model: str = "claude-sonnet-4-5",
    anthropic_api_key: Optional[str] = None,
) -> MorphResponse:
    deid: DeidResult = deidentify_report(raw_report, use_ai=deid_with_ai, model=deid_model, api_key=deid_api_key)
    extracted_text = extract_report(deid.text)
    extracted = ask_spine_model(extracted_text, model=extract_model, api_key=anthropic_api_key)
    return to_api_payload(extracted, allowed_levels=allowed_levels, warnings=deid.warnings)
