from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from anthropic import Anthropic, transform_schema

from extract import deidentify_report, extract_report


class LabelFormat(BaseModel):
    answer: list[str]


class LabelRequest(BaseModel):
    text: str


class LabelResponse(BaseModel):
    labels: list[str]


MODEL = os.getenv("LABEL_MODEL", "gpt-5.1")

MODEL_INSTRUCTIONS = """Task: Extract spine levels ONLY where the report explicitly states an abnormal finding.

Abnormal findings include (non-exhaustive): degenerative change, disc bulge/protrusion/extrusion, stenosis, foraminal narrowing, facet arthropathy, fracture, edema, cord/root compression, alignment abnormality, spondylolisthesis, scoliosis, endplate changes, Modic changes, osteophytes.

Rules:
- Return a Python list literal of strings, e.g., ['L4-5', 'T12-L1'].
- Include a level only if an abnormality is explicitly linked to that level.
- Include a single vertebra (e.g., 'L1') ONLY if a vertebral abnormality is explicitly stated at that vertebra (e.g., 'L1 compression fracture').
- Exclude any level explicitly described as normal/negative (e.g., “no bulge,” “no stenosis,” “foramina patent,” “no impingement,” “unremarkable”).
- If abnormalities are mentioned without a specific level, ignore them (do not infer).
- Deduplicate and sort levels in anatomical order (C -> T -> L -> S; increasing numbers).
- If the report is not spine-related or no qualifying abnormalities exist, return [].

Output format:
- Output ONLY the final Python list literal.
- Do NOT output explanations, notes, headings, or extra text.
- The list must contain ALL qualifying levels (not just one).
"""


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


class MorphRequest(BaseModel):
    text: str
    allowed_levels: Optional[List[str]] = None


class MorphResponse(BaseModel):
    morph_targets: Dict[str, float] = Field(default_factory=dict)
    kept_levels: List[str] = Field(default_factory=list)
    cord_compression: bool = False
    nerve_root_impingement: bool = False
    alignment_notes: str = ""
    incidental: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def _normalize_level(level: str) -> str:
    s = (level or "").strip().upper()
    s = s.replace("–", "-").replace("—", "-").replace(" ", "")
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


def _build_morph_targets(level_to_weights: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for level, weights in level_to_weights.items():
        tag = level.replace("-", "_")
        for k, v in weights.items():
            x = float(v)
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0
            out[f"{k}_{tag}"] = x
    return out


_MORPH_SYSTEM = (
    "Return ONLY JSON matching the schema.\n"
    "Extract ONLY abnormal spine levels explicitly described.\n"
    "Do NOT output normal levels.\n"
    "Level format MUST be like T2-3 and T12-L1.\n"
    "For each abnormal level, output abnormalities with type, severity, size_mm when present.\n"
    "If the report explicitly states no cord compression and no nerve root impingement, set those false.\n"
    "Include alignment_notes if described.\n"
    "Include incidental non-spine findings in incidental.\n"
)


def _ask_label_model(text: str) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    client = OpenAI(api_key=api_key)

    for _ in range(3):
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": MODEL_INSTRUCTIONS},
                    {"role": "user", "content": text},
                ],
                response_format=LabelFormat,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("no_parsed")
            ans = parsed.answer
            if isinstance(ans, list) and all(isinstance(x, str) for x in ans):
                return ans
            return []
        except Exception:
            time.sleep(1.0)
    return []


def _ask_morph_model(text: str, *, model: str) -> ExtractedJson:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("missing_anthropic_api_key")

    client = Anthropic(api_key=api_key)
    resp = client.beta.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        betas=["structured-outputs-2025-11-13"],
        system=_MORPH_SYSTEM,
        messages=[{"role": "user", "content": text}],
        output_format={"type": "json_schema", "schema": transform_schema(ExtractedJson)},
    )
    raw = resp.content[0].text
    return ExtractedJson.model_validate_json(raw)


def _to_unreal_payload(extracted: ExtractedJson, allowed_levels: List[str]) -> MorphResponse:
    allowed = {_normalize_level(x) for x in allowed_levels if isinstance(x, str) and x.strip()}
    level_to_weights: Dict[str, Dict[str, float]] = {}
    kept: List[str] = []

    for lvl in extracted.levels:
        nl = _normalize_level(lvl.level)
        if nl not in allowed:
            continue
        weights = _compute_weights(lvl.abnormalities)
        if not weights:
            continue
        level_to_weights[nl] = weights
        kept.append(nl)

    morph_targets = _build_morph_targets(level_to_weights)

    gf = extracted.global_findings
    return MorphResponse(
        morph_targets=morph_targets,
        kept_levels=kept,
        cord_compression=bool(gf.cord_compression),
        nerve_root_impingement=bool(gf.nerve_root_impingement),
        alignment_notes=gf.alignment_notes or "",
        incidental=list(gf.incidental) if isinstance(gf.incidental, list) else [],
        warnings=[],
    )


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/label", response_model=LabelResponse)
def label(req: LabelRequest):
    labels = _ask_label_model(req.text)
    return LabelResponse(labels=labels)


@app.post("/morph", response_model=MorphResponse)
async def morph(req: MorphRequest):
    allowed = req.allowed_levels or ["T2-3", "T12-L1"]
    deid = deidentify_report(req.text)
    extracted_text = extract_report(deid.text)

    model = os.getenv("MORPH_MODEL", "claude-sonnet-4-5")
    try:
        extracted_json = await asyncio.to_thread(_ask_morph_model, extracted_text, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    payload = _to_unreal_payload(extracted_json, allowed)
    return payload


@app.get("/health")
def health():
    return {"ok": True}
