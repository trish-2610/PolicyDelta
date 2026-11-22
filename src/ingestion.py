"""
Step 1: Ingestion (reading and cleaning the PDFs)

What this file does:
    1. Reads the list of documents from data/sources.json
    2. Opens each PDF from data/raw, page by page
    3. Drops the Hindi pages (the government PDFs have Hindi first, English after)
    4. Removes headers, footers and other junk lines, and fixes spacing glitches
    5. Keeps the ORIGINAL page number on every page (needed later for citations)
    6. Saves the clean text and prints a results table

Input : data/sources.json, data/raw/<doc_id>.pdf
Output: data/processed/<doc_id>.json        clean pages + document info (used by the next steps)
        data/processed/<doc_id>.txt         same text, easy to read in VS Code
        data/eval/ingestion_results.json    the results table

Run (venv active, inside the project folder):
    python src/ingestion.py
"""

## importing required libraries
import json
import re
import time
from pathlib import Path

from pypdf import PdfReader


## folder paths (all relative to the project folder)
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCES_FILE = ROOT / "data" / "sources.json"
RESULTS_FILE = ROOT / "data" / "eval" / "ingestion_results.json"

## a page is a Hindi page if half or more of its letters are Hindi (Devanagari)
HINDI_PAGE_THRESHOLD = 0.5

## a page with less text than this after cleaning is treated as empty
MIN_PAGE_CHARS = 30

## sanity check: how many numbered sections / rules each legal document must have
EXPECTED_HEADINGS = {
    "dpdp_act_2023": 44,          ## the Act has 44 sections
    "dpdp_rules_2025_final": 23,  ## the final Rules have 23 rules
}

## any Hindi (Devanagari) letter
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

## lines that are gazette headers, footers or cover text, not law text
JUNK_LINES = [
    re.compile(r"^\s*\d+\s+THE GAZETTE OF INDIA\b.*$"),                    ## page header
    re.compile(r"^\s*SEC\.\s*\d+\]\s*THE GAZETTE OF INDIA\b.*$"),          ## page header
    re.compile(r"^\s*xxxGID[HE]xxx\s*$"),                                  ## gazette id
    re.compile(r"^\s*CG-DL-E-\d+-\d+\s*$"),                                ## gazette id
    re.compile(r"^\s*Uploaded by Dte\. of Printing\b.*$", re.I),           ## press footer
    re.compile(r"^\s*and Published by the Controller of Publications\b.*$", re.I),
    re.compile(r"^\s*UPLOADED BY THE MANAGER\b.*$", re.I),
    re.compile(r"^\s*AND PUBLISHED BY THE CONTROLLER\b.*$", re.I),
    re.compile(r"^\s*MGIPMRND\b.*$"),
    re.compile(r"^\s*EXTRAORDINARY\s*$"),                                  ## cover block
    re.compile(r"^\s*PART II\s*[—–-]+\s*Section\b.*$"),
    re.compile(r"^\s*PUBLISHED\s+BY\s+AUTHORITY\s*$"),
    re.compile(r"^\s*Separate paging is given\b.*$"),
    re.compile(r"^\s*No\.\s*\d+\]\s*NEW DELHI\b.*$"),
    re.compile(r"^(?:vlk/kkj\.k|Hkkx|izkf/kdkj|lañ|bl Hkkx|jftLVªh)(?:\s|$).*"),  ## old Hindi font read as gibberish
    re.compile(r"^\s*[•●▪]\s*$"),                                          ## empty bullet
]


def load_sources():
    """This function reads the list of documents (written by hand in data/sources.json)"""

    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


def hindi_share(text):
    """This function returns what share of the letters on a page are Hindi (0.0 to 1.0)"""

    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    hindi_letters = [c for c in letters if DEVANAGARI.match(c)]
    return len(hindi_letters) / len(letters)


def clean_page(raw_text):
    """This function cleans one English page. Returns the clean text and how many junk lines were removed."""

    raw_text = raw_text.replace(" ", " ").replace("\x0c", "")

    kept_lines = []
    junk_count = 0

    ## step 1: remove junk lines (Hindi fragments, headers, footers)
    for line in raw_text.split("\n"):
        if DEVANAGARI.search(line) or any(pattern.match(line) for pattern in JUNK_LINES):
            junk_count += 1
        else:
            kept_lines.append(line.rstrip())

    text = "\n".join(kept_lines)

    ## step 2: fix glitches that come from PDF text extraction
    text = re.sub(r"(\w)\s*-\s*\n\s*(\w)", r"\1-\2", text)       ## "sub -\nsection" -> "sub-section"
    text = re.sub(r"(?<=[A-Za-z]) -(?=[A-Za-z])", "-", text)     ## "sub -section"   -> "sub-section"
    text = re.sub(r"(\d)\s+(st|nd|rd|th)\b", r"\1\2", text)       ## "3 rd January"   -> "3rd January"
    text = re.sub(r"(?m)^(\d{1,2})\.(?=[A-Z])", r"\1. ", text)    ## "17.Appointment" -> "17. Appointment"
    text = re.sub(r"[ \t]+", " ", text)                           ## many spaces      -> one space
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()                ## many blank lines -> one

    return text, junk_count


def count_headings(pages):
    """This function counts numbered headings like '1. ', '2. ', '3. ' ... in order.
    The Act must give 44 and the final Rules 23. If not, the cleaning missed something."""

    numbers_found = set()
    for page in pages:
        for line in page["text"].split("\n"):
            match = re.match(r"^(\d{1,2})\.\s", line)
            if match:
                numbers_found.add(int(match.group(1)))

    count = 0
    while (count + 1) in numbers_found:
        count += 1
    return count


def process_pdf(source):
    """This function reads one PDF, cleans every page and returns the clean document and its numbers"""

    start_time = time.perf_counter()
    pdf_path = RAW_DIR / f"{source['doc_id']}.pdf"
    reader = PdfReader(str(pdf_path))

    pages = []
    hindi_pages = 0
    empty_pages = 0
    junk_lines = 0
    chars_before = 0

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""

        ## drop Hindi pages
        if hindi_share(raw_text) >= HINDI_PAGE_THRESHOLD:
            hindi_pages += 1
            continue

        ## clean English pages
        text, junk_count = clean_page(raw_text)
        junk_lines += junk_count

        ## drop pages that are empty after cleaning
        if len(text) < MIN_PAGE_CHARS:
            empty_pages += 1
            continue

        chars_before += len(raw_text)
        pages.append({"page": page_number, "text": text})   ## original PDF page number is kept

    ## the clean document = document info from sources.json + clean pages
    document = {**source, "pages": pages}

    headings_found = count_headings(pages)
    headings_expected = EXPECTED_HEADINGS.get(source["doc_id"])
    if headings_expected is None:
        check = "-"
    elif headings_found == headings_expected:
        check = "OK"
    else:
        check = "CHECK"

    stats = {
        "doc_id": source["doc_id"],
        "pdf_pages": len(reader.pages),
        "hindi_pages_dropped": hindi_pages,
        "empty_pages_dropped": empty_pages,
        "pages_kept": len(pages),
        "chars_before": chars_before,
        "chars_after": sum(len(p["text"]) for p in pages),
        "junk_lines_removed": junk_lines,
        "pages_with_hindi_left": sum(1 for p in pages if DEVANAGARI.search(p["text"])),
        "headers_left": sum(p["text"].count("THE GAZETTE OF INDIA") for p in pages),
        "headings_found": headings_found,
        "headings_expected": headings_expected,
        "heading_check": check,
        "seconds": round(time.perf_counter() - start_time, 2),
    }
    return document, stats


def save_document(document):
    """This function saves the clean document as .json (for the code) and .txt (for reading)"""

    doc_id = document["doc_id"]

    json_path = PROCESSED_DIR / f"{doc_id}.json"
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    readable = "\n\n".join(f"===== PAGE {p['page']} =====\n{p['text']}" for p in document["pages"])
    txt_path = PROCESSED_DIR / f"{doc_id}.txt"
    txt_path.write_text(readable, encoding="utf-8")


def print_table(all_stats):
    """This function prints the results table"""

    print("\nINGESTION RESULTS")
    print(f"{'doc_id':<33}{'pdf':>4}{'hindi':>6}{'empty':>6}{'kept':>5}{'before':>8}{'after':>8}"
          f"{'junk':>6}{'hin_left':>9}{'hdr_left':>9}{'1..n':>6}{'check':>7}{'sec':>6}")
    for s in all_stats:
        print(f"{s['doc_id']:<33}{s['pdf_pages']:>4}{s['hindi_pages_dropped']:>6}{s['empty_pages_dropped']:>6}"
              f"{s['pages_kept']:>5}{s['chars_before']:>8}{s['chars_after']:>8}{s['junk_lines_removed']:>6}"
              f"{s['pages_with_hindi_left']:>9}{s['headers_left']:>9}{s['headings_found']:>6}"
              f"{s['heading_check']:>7}{s['seconds']:>6}")

    print("\nHow to read it:")
    print("  hin_left and hdr_left should be 0 (no Hindi or headers left after cleaning)")
    print("  check should be OK for the Act (44 sections) and the final Rules (23 rules)")


## main function
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    sources = load_sources()
    print(f"{len(sources)} documents listed in sources.json")

    all_stats = []
    missing = []

    for source in sources:
        pdf_path = RAW_DIR / f"{source['doc_id']}.pdf"
        if not pdf_path.exists():
            missing.append(source["doc_id"])
            continue

        document, stats = process_pdf(source)
        save_document(document)
        all_stats.append(stats)
        print(f"Cleaned {source['doc_id']}: kept {stats['pages_kept']} of {stats['pdf_pages']} pages")

    ## save the results table
    RESULTS_FILE.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print_table(all_stats)

    ## warnings
    if missing:
        print("\nMISSING PDFs (save them in data/raw with exactly this name):")
        for doc_id in missing:
            print(f"  data/raw/{doc_id}.pdf")

    known_ids = {s["doc_id"] for s in sources}
    old_files = [f.name for f in PROCESSED_DIR.glob("*.txt") if f.stem not in known_ids]
    if old_files:
        print("\nWARNING: old files in data/processed that are not in sources.json (delete them):")
        for name in old_files:
            print(f"  {name}")

    print(f"\nSaved: clean documents in {PROCESSED_DIR}")
    print(f"Saved: results table in {RESULTS_FILE}")


if __name__ == "__main__":
    main()
