import re
from dataclasses import dataclass
from typing import Literal, TypeAlias, Any, Dict, List, Optional, Tuple
import json
from openai import OpenAI

Status: TypeAlias = Literal["ok", "fail"]
HeadingName: TypeAlias = Literal["impression", "indication", "technique", "findings"]


_DEID_SCHEMA_HINT = {
    "type": "object",
    "required": ["deidentified_text", "removed"],
    "properties": {
        "deidentified_text": {"type": "string"},
        "removed": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "value"],
                "properties": {"type": {"type": "string"}, "value": {"type": "string"}},
            },
        },
        "notes": {"type": "string"},
    },
}

_LLM_INSTRUCTIONS = (
    "You are a strict medical text de-identification system.\n"
    "Task: remove or replace any sensitive identifiers (PHI) from the input medical imaging report.\n"
    "Return ONLY valid JSON with keys:\n"
    "  deidentified_text: string\n"
    "  removed: array of {type: string, value: string}\n"
    "No extra keys unless 'notes' is needed.\n"
    "Rules:\n"
    "- Replace patient names with [PATIENT_NAME].\n"
    "- Replace MRN with [MRN], accession with [ACCESSION], DOB with [DOB], dates with [DATE].\n"
    "- Replace addresses with [ADDRESS], phone with [PHONE], email with [EMAIL].\n"
    "- Replace provider names with [PROVIDER] and facilities with [FACILITY].\n"
    "- Do not alter clinical findings.\n"
    "- Do not invent or hallucinate.\n"
)

_JSON_ONLY_RE = re.compile(r"(?s)\{.*\}\s*$")

@dataclass(frozen=True, slots=True)
class ExtractOk:
    status: Literal["ok"]
    sections: dict[HeadingName, str]

@dataclass(frozen=True, slots=True)
class ExtractFail:
    status: Literal["fail"]
    raw_text: str

ExtractResult: TypeAlias = ExtractOk | ExtractFail

@dataclass(frozen=True, slots=True)
class SelectOk:
    status: Literal["ok"]
    selected_text: str
    included_headings: list[HeadingName]
    total_strength: int

@dataclass(frozen=True, slots=True)
class SelectFail:
    status: Literal["fail"]
    reason: Literal["empty_input", "no_sections", "insufficient_strength"]
    raw_text: str
    total_strength: int

SelectResult: TypeAlias = SelectOk | SelectFail

def normalize_text(text: str) -> str:
    return text.casefold()

def heading_strength() -> dict[HeadingName, int]:
    return {"impression": 100, "findings": 90, "technique": 30, "indication": 20}

def _compile_heading_pattern() -> re.Pattern[str]:
    headings: list[str] = ["impression", "indication", "technique", "findings"]
    alternation = "|".join(re.escape(h) for h in headings)
    pattern = rf"(?m)^(?P<h>{alternation})\s*(?:[:\-–—])\s*"
    return re.compile(pattern, flags=re.IGNORECASE)

def extract_headings(text: str) -> ExtractResult:
    pattern = _compile_heading_pattern()
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return ExtractFail(status="fail", raw_text=text)
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        heading = m.group("h").strip().casefold()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections[heading] = body
    typed_sections: dict[HeadingName, str] = {}
    for k, v in sections.items():
        if k == "impression":
            typed_sections["impression"] = v
        elif k == "indication":
            typed_sections["indication"] = v
        elif k == "technique":
            typed_sections["technique"] = v
        elif k == "findings":
            typed_sections["findings"] = v
    if not typed_sections:
        return ExtractFail(status="fail", raw_text=text)
    return ExtractOk(status="ok", sections=typed_sections)

def select_headings_as_text(original_text: str, heading_to_text: dict[HeadingName, str], min_total_strength: int) -> SelectResult:
    if not original_text.strip():
        return SelectFail(status="fail", reason="empty_input", raw_text=original_text, total_strength=0)
    if not heading_to_text:
        return SelectFail(status="fail", reason="no_sections", raw_text=original_text, total_strength=0)
    strength_map = heading_strength()
    ranked: list[tuple[int, HeadingName, str]] = []
    for h, body in heading_to_text.items():
        body_clean = body.strip()
        if not body_clean:
            continue
        ranked.append((strength_map[h], h, body_clean))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    total_strength = 0
    included: list[HeadingName] = []
    parts: list[str] = []
    for s, h, body in ranked:
        if s <= 0:
            continue
        total_strength += s
        included.append(h)
        parts.append(f"{h.upper()}:\n{body}")
    if total_strength < min_total_strength or not parts:
        return SelectFail(status="fail", reason="insufficient_strength", raw_text=original_text, total_strength=total_strength)
    selected_text = "\n\n".join(parts).strip()
    return SelectOk(status="ok", selected_text=selected_text, included_headings=included, total_strength=total_strength)

@dataclass(frozen=True, slots=True)
class DeidResult:
    text: str
    removed: List[Dict[str, Any]]
    warnings: List[str]

_LINE_FIELD_RULES: List[Tuple[str, re.Pattern, str]] = [
    ("PATIENT_NAME_LINE", re.compile(r"(?im)^\s*(patient\s*name|name)\s*[:#].*$"), "[PATIENT_NAME]"),
    ("MRN_LINE", re.compile(r"(?im)^\s*(mrn|medical\s*record\s*number)\s*[:#].*$"), "[MRN]"),
    ("ACCESSION_LINE", re.compile(r"(?im)^\s*(accession|acc)\s*[:#].*$"), "[ACCESSION]"),
    ("DOB_LINE", re.compile(r"(?im)^\s*(dob|d\.o\.b\.|date\s*of\s*birth)\s*[:#].*$"), "[DOB]"),
    ("PHONE_LINE", re.compile(r"(?im)^\s*(phone|tel|telephone)\s*[:#].*$"), "[PHONE]"),
    ("EMAIL_LINE", re.compile(r"(?im)^\s*(email|e-mail)\s*[:#].*$"), "[EMAIL]"),
    ("ADDRESS_LINE", re.compile(r"(?im)^\s*(address|addr)\s*[:#].*$"), "[ADDRESS]"),
    ("ID_LINE", re.compile(r"(?im)^\s*(patient\s*id|id|acct|account|case)\s*[:#].*$"), "[ID]"),
    ("FACILITY_LINE", re.compile(r"(?im)^\s*(facility|institution|hospital|clinic|imaging\s*center)\s*[:#].*$"), "[FACILITY]"),
    ("PROVIDER_LINE", re.compile(r"(?im)^\s*(referring|ordering|attending)\s*(physician|provider|dr\.?)?\s*[:#].*$"), "[PROVIDER]"),
]

_INLINE_RULES: List[Tuple[str, re.Pattern, str]] = [
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[EMAIL]"),
    ("PHONE", re.compile(r"\b(?:\+?1[\s-]?)?(?:\(\s*\d{3}\s*\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b"), "[PHONE]"),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    ("MRN", re.compile(r"\bMRN\s*[:#]?\s*[A-Z0-9-]{5,}\b", re.I), "[MRN]"),
    ("ACCESSION", re.compile(r"\b(?:Accession|ACC)\s*[:#]?\s*[A-Z0-9-]{5,}\b", re.I), "[ACCESSION]"),
    ("DOB", re.compile(r"\b(?:DOB|D\.O\.B\.|Date\s*of\s*Birth)\s*[:#]?\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z]{3,9}\s+\d{1,2},\s+\d{4})\b", re.I), "[DOB]"),
    ("DATE", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})\b"), "[DATE]"),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})\b"), "[ZIP]"),
]

_PROVIDER_GUESS = re.compile(r"(?im)\b(?:Dr\.|Doctor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_FACILITY_GUESS = re.compile(r"(?im)\b(?:Hospital|Medical\s+Center|Clinic|Imaging\s+Center|Radiology)\b(?:\s+[A-Z][A-Za-z&.-]+){0,6}\b")

def _apply_rules(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    removed: List[Dict[str, Any]] = []
    lines = text.splitlines()
    out_lines: List[str] = []
    for line in lines:
        replaced = line
        for typ, rx, repl in _LINE_FIELD_RULES:
            if rx.match(replaced):
                removed.append({"type": typ, "value": replaced})
                replaced = re.sub(r"^.*$", repl, replaced)
                break
        out_lines.append(replaced)
    out = "\n".join(out_lines)

    for typ, rx, repl in _INLINE_RULES:
        for m in list(rx.finditer(out)):
            removed.append({"type": typ, "value": m.group(0)})
        out = rx.sub(repl, out)

    for m in list(_PROVIDER_GUESS.finditer(out)):
        removed.append({"type": "PROVIDER_NAME_GUESS", "value": m.group(0)})
    out = _PROVIDER_GUESS.sub("[PROVIDER]", out)

    for m in list(_FACILITY_GUESS.finditer(out)):
        removed.append({"type": "FACILITY_GUESS", "value": m.group(0)})
    out = _FACILITY_GUESS.sub("[FACILITY]", out)

    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, removed

_LLM_INSTRUCTIONS = (
    "You are a strict medical text de-identification system.\n"
    "Remove or replace any sensitive identifiers (PHI) from the input medical imaging report.\n"
    "Return ONLY valid JSON with keys:\n"
    "  deidentified_text: string\n"
    "  removed: array of {type: string, value: string}\n"
    "Rules:\n"
    "- Replace patient names with [PATIENT_NAME].\n"
    "- Replace MRN with [MRN], accession with [ACCESSION], DOB with [DOB], dates with [DATE].\n"
    "- Replace addresses with [ADDRESS], phone with [PHONE], email with [EMAIL].\n"
    "- Replace provider names with [PROVIDER] and facilities with [FACILITY].\n"
    "- Do not alter clinical findings.\n"
    "- Do not remove entire paragraphs.\n"
    "- Do not invent.\n"
)

_JSON_ONLY_RE = re.compile(r"(?s)\{.*\}\s*$")

def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    s2 = s.strip()
    m = _JSON_ONLY_RE.search(s2)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _validate_llm_payload(obj: Dict[str, Any]) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    if not isinstance(obj, dict):
        return None
    dt = obj.get("deidentified_text")
    rm = obj.get("removed")
    if not isinstance(dt, str) or not isinstance(rm, list):
        return None
    cleaned_removed: List[Dict[str, Any]] = []
    for it in rm:
        if isinstance(it, dict) and isinstance(it.get("type"), str) and isinstance(it.get("value"), str):
            cleaned_removed.append({"type": it["type"], "value": it["value"]})
    return dt, cleaned_removed

def deidentify_report(text: str, *, use_ai: bool = True, model: str = "gpt-5-mini", api_key: Optional[str] = None) -> DeidResult:
    warnings: List[str] = []
    removed_all: List[Dict[str, Any]] = []

    rule_text, rule_removed = _apply_rules(text)
    removed_all.extend(rule_removed)

    if not use_ai or OpenAI is None:
        final_text, post_removed = _apply_rules(rule_text)
        removed_all.extend(post_removed)
        if use_ai and OpenAI is None:
            warnings.append("openai package not available; returned rule-based result only.")
        return DeidResult(text=final_text, removed=removed_all, warnings=warnings)

    try:
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        warnings.append("Failed to initialize OpenAI client; returned rule-based result only.")
        final_text, post_removed = _apply_rules(rule_text)
        removed_all.extend(post_removed)
        return DeidResult(text=final_text, removed=removed_all, warnings=warnings)

    prompt = "INPUT_REPORT:\n" + rule_text
    llm_text: Optional[str] = None
    llm_removed: List[Dict[str, Any]] = []

    try:
        resp = client.responses.create(
            model=model,
            instructions=_LLM_INSTRUCTIONS,
            input=prompt,
            store=False,
        )
        raw = getattr(resp, "output_text", "") or ""
        obj = _safe_json_load(raw)
        if obj is None:
            warnings.append("AI did not return valid JSON; used rule-based result only.")
        else:
            parsed = _validate_llm_payload(obj)
            if parsed is None:
                warnings.append("AI JSON missing required keys/types; used rule-based result only.")
            else:
                llm_text, llm_removed = parsed
    except Exception:
        warnings.append("AI request failed; used rule-based result only.")

    base = llm_text if isinstance(llm_text, str) and llm_text.strip() else rule_text
    removed_all.extend(llm_removed)

    final_text, post_removed = _apply_rules(base)
    removed_all.extend(post_removed)

    return DeidResult(text=final_text, removed=removed_all, warnings=warnings)

def extract_report(text: str) -> str:
    normalized_text = normalize_text(text)
    extracted_text = extract_headings(normalized_text)
    if extracted_text.status == "fail":
        selected: SelectResult = SelectFail(status="fail", reason="no_sections", raw_text=normalized_text, total_strength=0)
    
    else:
        selected = select_headings_as_text(normalized_text, extracted.sections, min_total_strength=90)
    
    final_text = selected.selected_text if selected.status == "ok" else normalized_text
    return final_text

if __name__ == "__main__":
    input_text = "IMPRESSION: None\nINDICATION: Alot\nFINDINGS: L4-5 mild bulge"
    normalized_text = normalize_text(input_text)
    extracted = extract_headings(normalized_text)
    if extracted.status == "fail":
        selected: SelectResult = SelectFail(status="fail", reason="no_sections", raw_text=normalized_text, total_strength=0)
    else:
        selected = select_headings_as_text(normalized_text, extracted.sections, min_total_strength=90)
    final_text = selected.selected_text if selected.status == "ok" else normalized_text
    print(f"Original Input:\n{input_text}\n")
    print(f"Normalized:\n{normalized_text}\n")
    print(f"Extracted:\n{extracted}\n")
    print(f"Selected:\n{selected}\n")
    print(f"Final Text Used:\n{final_text}\n")
