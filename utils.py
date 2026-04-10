from __future__ import annotations

import os
import re
from typing import Dict, List

EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "claude-opus-4-6")
DEID_MODEL = os.getenv("DEID_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

APP_TITLE = "Office Ally Medical AI API"
APP_VERSION = "5.0.0"

MAX_CORNER = 0.5
MAX_SCALER = 0.5

DISC_INDEX: Dict[str, int] = {
    "C1": 0,  "C2": 1,  "C3": 2,  "C4": 3,  "C5": 4,  "C6": 5,  "C7": 6,
    "T1": 7,  "T2": 8,  "T3": 9,  "T4": 10, "T5": 11, "T6": 12,
    "T7": 13, "T8": 14, "T9": 15, "T10": 16, "T11": 17, "T12": 18,
    "L1": 19, "L2": 20, "L3": 21, "L4": 22, "L5": 23,
}

VALID_TYPES = [
    "annular_bulge", "disc_bulge", "protrusion", "extrusion",
    "stenosis", "foraminal_narrowing", "facet_arthropathy", "alignment",
    "fracture", "edema", "cord_compression", "nerve_root_impingement",
    "disc_height_loss", "other",
]

TYPE_ALIASES: Dict[str, str] = {
    "disc_height": "disc_height_loss",
    "disc_space_narrowing": "disc_height_loss",
    "disc_space_narrowing.": "disc_height_loss",
    "disc_space_loss": "disc_height_loss",
    "disc_space_collapse": "disc_height_loss",
    "disc_collapse": "disc_height_loss",
    "loss_of_disc_height": "disc_height_loss",
    "decreased_disc_height": "disc_height_loss",
    "disc_height_reduction": "disc_height_loss",
}

SEVERITY_ALIASES: Dict[str, str] = {
    "minimal": "mild",
    "slight": "mild",
}

FIELD_LABELS = [
    "Patient Name", "Patient ID", "MRN", "Medical Record Number",
    "Accession Number", "Accession", "DOB", "Date of Birth",
    "Gender", "Sex", "Age",
    "Referring Physician", "Referring", "Ordering Physician", "Ordering",
    "Attending Physician", "Attending",
    "Exam Date", "Exam", "Modality", "Report Status", "Report Date",
    "Location", "Facility", "Institution", "Hospital", "Clinic",
    "Phone", "Telephone", "Email", "Address",
    "INDICATION", "TECHNIQUE", "FINDINGS", "IMPRESSION", "COMPARISON",
]

LABEL_BREAK_RE = re.compile(
    r"(" + "|".join(re.escape(l) for l in FIELD_LABELS) + r")\s*[:#]",
    re.IGNORECASE,
)

GENDER_RE = re.compile(r"(?i)\b(?:gender|sex)\s*[:#]?\s*(male|female|m|f)\b")


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def disc_name(level: str) -> str:
    s = (level or "").strip().upper().replace("\u2013", "-").replace("\u2014", "-").replace(" ", "")
    return s.split("-", 1)[0] if "-" in s else s


JOINT_NAMES = ["_1", "_2", "_3", "_4", ""]

JOINT_POSITIONS = {
    "_1": "top_right",
    "_2": "bottom_right",
    "_3": "bottom_left",
    "_4": "top_left",
    "": "center_scaler",
}


def validate_axis(values: List[float], max_val: float = MAX_CORNER) -> List[float]:
    if len(values) != 3:
        return [1.0, 1.0, 1.0]
    return [round(1.0 + clamp(v, -max_val, max_val), 4) for v in values]


def insert_field_breaks(text: str) -> str:
    def _sub(m: re.Match) -> str:
        start = m.start()
        prefix = "" if start == 0 or text[start - 1] == "\n" else "\n"
        return prefix + m.group(1) + ":"
    return LABEL_BREAK_RE.sub(_sub, text)


def detect_gender(text: str) -> str:
    m = GENDER_RE.search(text)
    if m:
        v = m.group(1).lower()
        if v in ("male", "m"):
            return "male"
        if v in ("female", "f"):
            return "female"
    return "unknown"
