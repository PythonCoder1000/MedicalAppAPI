from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from schemas import MorphRequest, MorphResponse
from spine_pipeline import process_report_to_payload

APP_TITLE = "Office Ally Medical AI API"
APP_VERSION = "4.1.1"

DEID_MODEL = os.getenv("DEID_MODEL", "gpt-5-mini")
MORPH_MODEL = os.getenv("MORPH_MODEL", "claude-sonnet-4-6")

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    print(f"\n=== {request.method} {request.url.path} ===")
    print(f"Headers: {dict(request.headers)}")
    print(f"Body ({len(body)} bytes): {body[:2000]!r}")
    print("=" * 40)
    response = await call_next(request)
    return response


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
