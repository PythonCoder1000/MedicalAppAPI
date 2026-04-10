from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from schemas import MorphRequest, MorphResponse
from spine_pipeline import process_report_to_payload
from utils import APP_TITLE, APP_VERSION, DEID_MODEL, EXTRACT_MODEL

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
            use_deid=req.use_deid,
            deid_model=DEID_MODEL,
            extract_model=EXTRACT_MODEL,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
