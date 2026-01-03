from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import time

class LabelFormat(BaseModel):
    answer: list[str]

class LabelRequest(BaseModel):
    text: str

class LabelResponse(BaseModel):
    labels: list[str]
    pid: int

MODEL = "gpt-4o-mini"

MODEL_INSTRUCTIONS = """Extract only the anatomical locations that are abnormal from a radiology report.
Return locations as spine levels (preferred) and single vertebrae only when the report explicitly describes an abnormality at that vertebra.
Do not output explanations, only the final list. Example of output: ['L1', 'L2-3'] where L1 is for the vertebrea and L2-3 is for the disc between them.
If the report is not spinal or spinal related, then do not output anything (return an empty list)."""

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
            ans = completion.choices[0].message.parsed.answer
            if isinstance(ans, list) and all(isinstance(x, str) for x in ans):
                return ans
            return []
        except Exception as e:
            time.sleep(1.0)
    return []

@app.post("/label", response_model=LabelResponse)
def label(req: LabelRequest):
    labels = ask_model(req.text)
    print("ENDPOINT_RETURN:", labels)
    return LabelResponse(labels=labels, pid=os.getpid())

@app.get("/health")
def health():
    return {"ok": True}
