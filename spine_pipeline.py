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

_JSON_ONLY_RE = re.compile(r"(?s)\{.*\}\s*$")


def _clamp_delta(x: float) -> float:
    if x < -1.0:
        return -1.0
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


def _sev_to_pos(sev: str) -> float:
    if sev == "none":
        return 0.0
    if sev == "mild":
        return 0.3
    if sev == "moderate":
        return 0.6
    if sev == "severe":
        return 1.0
    return 0.25


def _sev_to_height_loss(sev: str) -> float:
    if sev == "none":
        return 0.0
    if sev == "mild":
        return 0.2
    if sev == "moderate":
        return 0.35
    if sev == "severe":
        return 0.5
    return 0.25


def _vert_from_notes(notes: str) -> str:
    s = (notes or "").lower()
    top_words = ("superior", "upper", "cranial", "cephalad", "top")
    bot_words = ("inferior", "lower", "caudal", "caudad", "bottom")
    has_top = any(w in s for w in top_words)
    has_bot = any(w in s for w in bot_words)
    if has_top and not has_bot:
        return "top"
    if has_bot and not has_top:
        return "bottom"
    return "both"


def _empty_joint_map() -> Dict[JointName, float]:
    return {
        "center": 0.0,
        "topright": 0.0,
        "bottomright": 0.0,
        "topleft": 0.0,
        "bottomleft": 0.0,
    }


def _add_set(d: Dict[JointName, float], key: JointName, value: float) -> None:
    d[key] = _clamp_delta(float(d.get(key, 0.0)) + float(value))


def _apply_side_pair(d: Dict[JointName, float], side: str, value: float, which: str) -> None:
    v = float(value)
    if side == "left":
        if which in ("top", "both"):
            _add_set(d, "topleft", v)
        if which in ("bottom", "both"):
            _add_set(d, "bottomleft", v)
        return

    if side == "right":
        if which in ("top", "both"):
            _add_set(d, "topright", v)
        if which in ("bottom", "both"):
            _add_set(d, "bottomright", v)
        return

    if side == "bilateral":
        if which in ("top", "both"):
            _add_set(d, "topleft", v)
            _add_set(d, "topright", v)
        if which in ("bottom", "both"):
            _add_set(d, "bottomleft", v)
            _add_set(d, "bottomright", v)
        return

    v2 = v * 0.7
    if which in ("top", "both"):
        _add_set(d, "topleft", v2)
        _add_set(d, "topright", v2)
    if which in ("bottom", "both"):
        _add_set(d, "bottomleft", v2)
        _add_set(d, "bottomright", v2)


def compute_joint_moves(abns: List[Abnormality]) -> Dict[JointName, float]:
    joints = _empty_joint_map()

    for a in abns:
        lat = a.laterality or "unknown"
        region = a.region or "unknown"
        which = _vert_from_notes(a.notes)

        if a.type == "disc_height_loss":
            mag = _sev_to_height_loss(a.severity)
            neg = -mag
            _add_set(joints, "center", neg * 0.3)
            _apply_side_pair(joints, lat if lat != "midline" else "unknown", neg, which)
            continue

        sev_w = _sev_to_pos(a.severity)
        size_w = _clamp_delta(float(a.size_mm) / 5.0) if isinstance(a.size_mm, (int, float)) else None

        if a.type in ("annular_bulge", "disc_bulge"):
            w = size_w if size_w is not None else max(sev_w, 0.2)
            if region in ("foraminal", "extraforaminal"):
                _apply_side_pair(joints, lat, w, which)
            elif region == "paracentral":
                _add_set(joints, "center", w * 0.8)
                _apply_side_pair(joints, lat, w * 0.8 if lat in ("left", "right", "bilateral") else w * 0.5, which)
            else:
                _add_set(joints, "center", w)
            continue

        if a.type == "stenosis":
            _add_set(joints, "center", sev_w)
            continue

        if a.type == "foraminal_narrowing":
            _apply_side_pair(joints, lat, sev_w if sev_w > 0 else 0.2, which)
            continue

        if a.type in ("protrusion", "extrusion"):
            w = size_w if size_w is not None else max(sev_w, 0.4)
            if region in ("central", "unknown"):
                _add_set(joints, "center", w)
            elif region == "paracentral":
                _add_set(joints, "center", w * 0.75)
                _apply_side_pair(joints, lat, w, which)
            else:
                _apply_side_pair(joints, lat, w, which)
            continue

        if a.type == "facet_arthropathy":
            _apply_side_pair(joints, lat, max(sev_w, 0.2), which)
            continue

        if a.type == "cord_compression":
            _add_set(joints, "center", max(sev_w, 0.8))
            continue

        if a.type == "nerve_root_impingement":
            _apply_side_pair(joints, lat, max(sev_w, 0.7), which)

    return {k: round(_clamp_delta(v), 4) for k, v in joints.items()}


def _semantic_to_joint_ids(semantic: Dict[JointName, float]) -> Dict[JointId, float]:
    out: Dict[JointId, float] = {}
    for k, v in semantic.items():
        out[JOINT_ID_BY_NAME[k]] = float(v)
    return out


def _build_morph_targets(level: str, joints: Dict[JointId, float]) -> Dict[str, float]:
    tag = normalize_level(level).replace("-", "_")
    out: Dict[str, float] = {}
    for jid, amount in joints.items():
        out[f"{jid}_{tag}"] = round(_clamp_delta(float(amount)), 4)
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


def to_api_payload(extracted: ExtractedJson, warnings: Optional[List[str]] = None) -> MorphResponse:
    discs: List[DiscOut] = []
    morph_targets: Dict[str, float] = {}

    for lvl in extracted.levels:
        nl = normalize_level(lvl.level)

        semantic = compute_joint_moves(lvl.abnormalities)
        joints = _semantic_to_joint_ids(semantic)

        if not joints or max(abs(v) for v in joints.values()) <= 0.0:
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
    deid_with_ai: bool = True,
    deid_model: str = "gpt-5-mini",
    deid_api_key: Optional[str] = None,
    extract_model: str = "claude-sonnet-4-5",
    anthropic_api_key: Optional[str] = None,
) -> MorphResponse:
    deid: DeidResult = deidentify_report(raw_report, use_ai=deid_with_ai, model=deid_model, api_key=deid_api_key)
    extracted_text = extract_report(deid.text)
    extracted = ask_spine_model(extracted_text, model=extract_model, api_key=anthropic_api_key)
    return to_api_payload(extracted, warnings=deid.warnings)
