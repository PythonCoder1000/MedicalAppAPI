from pathlib import Path
from pypdf import PdfReader
import csv

INPUT_DIR = Path("pure_data")
OUTPUT_CSV = Path("processed_data/data.csv")

DROP_FIRST_N = 10
DROP_LAST_N = 1

def extract_pdf_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""

    parts = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        if len(lines) > DROP_FIRST_N + DROP_LAST_N:
            lines = lines[DROP_FIRST_N:len(lines) - DROP_LAST_N]
        else:
            lines = []

        parts.append("\n".join(lines))

    return "\n\n".join([p for p in parts if p.strip()]).strip()

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    existing_rows_by_id = {}
    existing_fieldnames = []

    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            existing_fieldnames = list(r.fieldnames or [])
            for row in r:
                rid = (row.get("id") or "").strip()
                if rid:
                    existing_rows_by_id[rid] = row

    field_set = set(existing_fieldnames)
    field_set.update(["id", "text", "label"])
    fieldnames = [fn for fn in existing_fieldnames if fn in field_set]
    for fn in ["id", "text", "label"]:
        if fn not in fieldnames:
            fieldnames.append(fn)
    for fn in sorted(field_set):
        if fn not in fieldnames:
            fieldnames.append(fn)

    pdf_paths = sorted([p for p in INPUT_DIR.rglob("*.pdf") if p.is_file()])

    for pdf_path in pdf_paths:
        rid = pdf_path.stem
        text = extract_pdf_text(pdf_path)

        if rid in existing_rows_by_id:
            existing_rows_by_id[rid]["text"] = text
            existing_rows_by_id[rid].setdefault("label", "")
        else:
            row = {k: "" for k in fieldnames}
            row["id"] = rid
            row["text"] = text
            row["label"] = ""
            existing_rows_by_id[rid] = row

    out_rows = [existing_rows_by_id[k] for k in sorted(existing_rows_by_id.keys())]

    tmp_path = OUTPUT_CSV.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    tmp_path.replace(OUTPUT_CSV)

if __name__ == "__main__":
    main()
