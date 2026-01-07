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

MODEL_INSTRUCTIONS = """Extract only spine locations where the report explicitly describes an abnormal finding (degenerative change, bulge/protrusion/extrusion, stenosis, foraminal narrowing, facet arthropathy, fracture, edema, cord/root compression, alignment abnormality, etc.).
Return a deduplicated Python-style list of strings containing spine levels (e.g., "L4-5", "T12-L1") and single vertebrae only if a vertebral abnormality is explicitly stated (e.g., "L1" for a compression fracture at L1).
Do not include levels described as normal (e.g., “no bulge,” “no stenosis,” “foramina patent,” “no nerve root contact/impingement”).
If an abnormality is described in a region without an exact level, include only the most specific location given (e.g., "lumbar lordosis" is not a level and should be ignored).
If the report is not spine-related, return [].
Output only the final list and nothing else.
Example output: ['L1', 'L2-3']"""

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
    text = remove_phrase_and_rest_of_line(req.text, "PATIENT NAME:")
    text = remove_phrase_and_rest_of_line(text, "Referring Physician:")
    labels = ask_model(text)
    return LabelResponse(labels=labels)

@app.get("/health")
def health():
    return {"ok": True}
