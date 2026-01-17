from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import re
import os
import time

class LabelFormat(BaseModel):
    answer: list[str]

class LabelRequest(BaseModel):
    text: str

class LabelResponse(BaseModel):
    labels: list[str]

MODEL = "gpt-5.1"

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

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_model(text: str) -> list[str]:
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
            ans= completion.choices[0].message.parsed
            if ans is None:
                raise Exception
            
            ans = ans.answer
            if isinstance(ans, list) and all(isinstance(x, str) for x in ans):
                return ans
            return []
        except Exception as e:
            time.sleep(1.0)
    return []

def remove_phrase_and_rest_of_line(text: str, phrase: str) -> str:
    pat = re.compile(re.escape(phrase) + r"[^\n]*")
    return pat.sub("", text)

@app.post("/label", response_model=LabelResponse)
def label(req: LabelRequest):
    labels = ask_model(req.text)
    return LabelResponse(labels=labels)

@app.get("/health")
def health():
    return {"ok": True}
