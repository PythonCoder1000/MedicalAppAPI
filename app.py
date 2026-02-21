from __future__ import annotations

import ast
import asyncio
import json
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from extract import deidentify_report, extract_report
from schemas import LabelRequest, LabelResponse, MorphRequest, MorphResponse
from spine_pipeline import process_report_to_payload


APP_TITLE = "Office Ally Medical AI API"
APP_VERSION = "1.0.0"

LABEL_MODEL = os.getenv("LABEL_MODEL", "gpt-5-mini")
DEID_MODEL = os.getenv("DEID_MODEL", "gpt-5-mini")
MORPH_MODEL = os.getenv("MORPH_MODEL", "claude-sonnet-4-5")

LABEL_INSTRUCTIONS = (
    "Extract spine levels ONLY where the report explicitly states an abnormal finding.\n"
    "Return ONLY valid JSON in this exact shape: {\"labels\": [\"T12-L1\", \"L4-5\"]}\n"
    "Rules:\n"
    "- Include a level only if an abnormality is explicitly linked to that level.\n"
    "- Exclude levels explicitly described as normal or negative.\n"
    "- If abnormality has no specific level, ignore it.\n"
    "- Deduplicate and sort in anatomical order.\n"
    "- If no qualifying levels exist, return {\"labels\": []}\n"
)


def _safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_label_output(raw: str) -> List[str]:
    obj = _safe_json_load(raw)
    if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
        return [str(x) for x in obj["labels"] if isinstance(x, str)]
    try:
        parsed = ast.literal_eval(raw.strip())
        if isinstance(parsed, list):
            return [str(x) for x in parsed if isinstance(x, str)]
    except Exception:
        pass
    return []


def _ask_label_model(text: str) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return []

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=LABEL_MODEL,
        instructions=LABEL_INSTRUCTIONS,
        input=text,
        store=False,
    )
    raw = getattr(resp, "output_text", "") or ""
    return _parse_label_output(raw)


app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": APP_TITLE, "version": APP_VERSION}


@app.post("/label", response_model=LabelResponse)
async def label(req: LabelRequest):
    deid = await asyncio.to_thread(deidentify_report, req.text, use_ai=True, model=DEID_MODEL, api_key=None)
    extracted_text = extract_report(deid.text)
    labels = await asyncio.to_thread(_ask_label_model, extracted_text)
    return LabelResponse(labels=labels)


@app.post("/morph", response_model=MorphResponse)
async def morph(req: MorphRequest):
    allowed = req.allowed_levels or ["T2-3", "T12-L1"]
    try:
        return process_report_to_payload(
            req.text,
            allowed_levels=allowed,
            deid_with_ai=req.use_ai_deid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
