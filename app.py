from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import MorphRequest, MorphResponse
from spine_pipeline import process_report_to_payload

APP_TITLE = "Office Ally Medical AI API"
APP_VERSION = "2.1.3"

DEID_MODEL = os.getenv("DEID_MODEL", "gpt-5-mini")
MORPH_MODEL = os.getenv("MORPH_MODEL", "claude-sonnet-4-5")

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


@app.post("/morph", response_model=MorphResponse)
async def morph(req: MorphRequest):
    try:
        return await asyncio.to_thread(
            process_report_to_payload,
            req.text,
            deid_with_ai=req.use_ai_deid,
            deid_model=DEID_MODEL,
            extract_model=MORPH_MODEL,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
