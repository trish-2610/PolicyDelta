"""
Step 2: Chunking (cut the clean documents into small pieces, in 2 ways, and compare)

Why: a search system cannot hand a whole 20-page document to the LLM. It finds and returns
small pieces (chunks). How we cut decides whether an answer stays together in one piece.

Way A: equal pieces of 1000 characters, each repeating the last 150 characters of the previous one
       (like cutting a ribbon into equal lengths)
Way B: cut at each rule / section heading, so one chunk = one rule
       (like tearing a book at the start of every chapter)
       a rule longer than 1500 characters is split again; a piece shorter than 200 is joined to the one before

Input : data/processed/<doc_id>.json     (clean pages from Step 1)
        data/eval/golden_set.json          (the 50 test questions)
Output: data/processed/chunks_A.jsonl, data/processed/chunks_B.jsonl
        data/eval/chunking_results.json
Run   : python src\\chunking.py
"""

## importing required libraries
import json
import re
from pathlib import Path


## folder paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCES_FILE = ROOT / "data" / "sources.json"
GOLDEN_FILE = ROOT / "data" / "eval" / "golden_set.json"
RESULTS_FILE = ROOT / "data" / "eval" / "chunking_results.json"

## settings for way A
SIZE_A = 1000        ## characters per chunk
OVERLAP_A = 150      ## characters repeated from the previous chunk

## settings for way B
MAX_B = 1500         ## a rule longer than this is split further
MIN_B = 200          ## a piece shorter than this is joined to the one before

## a schedule heading, e.g. "THE SCHEDULE" or "THIRD SCHEDULE"
SCHEDULE = re.compile(r"^(THE SCHEDULE|(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH) SCHEDULE)\b")

## a rule / section heading, e.g. "7. Intimation of personal data breach"
HEADING = re.compile(r"^(\d{1,2})\.\s?[A-Z(]")


def load_documents():
    """This function loads the clean documents from Step 1 (only the ones listed in sources.json)"""

    sources = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    documents = []
    for src in sources:
        doc_file = PROCESSED_DIR / f"{src['doc_id']}.json"
        documents.append(json.loads(doc_file.read_text(encoding="utf-8")))
    return documents


def join_pages(document):
    """This function joins all pages into one text and remembers where each page starts.
    Returns the text and a list like [(start_position, page_number), ...]"""

    text = ""
    page_starts = []
    for page in document["pages"]:
        page_starts.append((len(text), page["page"]))
        text += page["text"] + "\n"
    return text, page_starts


def page_of(page_starts, position):
    """This function tells which PDF page a character position belongs to"""

    page = page_starts[0][1]
    for start, number in page_starts:
        if start <= position:
            page = number
    return page


def way_a(text):
    """Way A: equal pieces with overlap. Returns a list of (start, end, heading)"""

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + SIZE_A, len(text))

        ## do not cut a word in half: move the end back to the last space
        if end < len(text):
            space = text.rfind(" ", start + SIZE_A - 100, end)
            if space > start:
                end = space

        pieces.append((start, end, ""))   ## way A does not know which rule it is in
        if end >= len(text):
            break
        start = max(end - OVERLAP_A, start + 1)
    return pieces


def find_headings(text):
    """This function finds where each rule / section / schedule starts.
    Numbers must come in order (1, then 2, then 3 ...) so a list item like '2. ' inside a rule is not taken as a heading"""

    headings = []
    next_number = 1
    in_schedule = False
    position = 0

    for line in text.split("\n"):
        clean_line = line.strip()
        if SCHEDULE.match(clean_line):
            in_schedule = True
            headings.append((position, clean_line[:80]))
        elif not in_schedule:
            match = HEADING.match(clean_line)
            if match and int(match.group(1)) == next_number:
                headings.append((position, clean_line[:80]))
                next_number += 1
        position += len(line) + 1

    ## the text before the first heading (title, preamble) gets its own label
    if not headings or headings[0][0] > 0:
        headings.insert(0, (0, "start of document"))
    return headings


def split_long(text, start, end, heading):
    """This function splits a rule that is too long, at a line break, keeping the same heading"""

    pieces = []
    while end - start > MAX_B:
        cut = text.rfind("\n", start + MAX_B // 2, start + MAX_B)
        if cut <= start:
            cut = start + MAX_B
        pieces.append((start, cut, heading))
        start = cut
    pieces.append((start, end, heading))
    return pieces


def way_b(text):
    """Way B: one piece per rule / section. Returns a list of (start, end, heading)"""

    headings = find_headings(text)
    pieces = []

    for i, (start, heading) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)

        ## join tiny pieces to the one before (e.g. a schedule title alone on a line)
        previous_is_bare_title = pieces and SCHEDULE.match(pieces[-1][2]) and pieces[-1][1] - pieces[-1][0] < MIN_B
        too_small = pieces and end - start < MIN_B and not SCHEDULE.match(heading)
        if previous_is_bare_title or too_small:
            pieces[-1] = (pieces[-1][0], end, pieces[-1][2])
        else:
            pieces.append((start, end, heading))

    ## split rules that are too long
    final = []
    for start, end, heading in pieces:
        if end - start > MAX_B:
            final.extend(split_long(text, start, end, heading))
        else:
            final.append((start, end, heading))
    return final


def make_chunks(documents, way_name, cut_function):
    """This function cuts every document and returns the chunks with their info (document, pages, heading)"""

    chunks = []
    for document in documents:
        text, page_starts = join_pages(document)
        for number, (start, end, heading) in enumerate(cut_function(text), start=1):
            body = text[start:end].strip()
            if not body:
                continue
            chunks.append({
                "chunk_id": f"{document['doc_id']}_{way_name}_{number:03d}",
                "doc_id": document["doc_id"],
                "title": document["title"],
                "source_type": document["source_type"],
                "page_start": page_of(page_starts, start),
                "page_end": page_of(page_starts, max(start, end - 1)),
                "heading": heading,
                "n_chars": len(body),
                "text": body,
            })
    return chunks


def normalize(text):
    """lower case and single spaces, so line breaks do not break the match"""

    return re.sub(r"\s+", " ", text).strip().lower()


def measure(chunks, evidence_list):
    """This function measures how good one way of cutting is"""

    ## 1. is each answer phrase from the golden set sitting fully inside one chunk of its document?
    kept_whole = 0
    for ev in evidence_list:
        phrase = normalize(ev["evidence"])
        if any(c["doc_id"] == ev["doc_id"] and phrase in normalize(c["text"]) for c in chunks):
            kept_whole += 1

    ## 2. how many chunks start in the middle of a sentence (first letter is small)?
    mid_sentence = sum(1 for c in chunks if c["text"][0].islower())

    ## 3. does every chunk know which rule / section it belongs to?
    with_heading = sum(1 for c in chunks if c["heading"])

    return {
        "chunks": len(chunks),
        "average_size": round(sum(c["n_chars"] for c in chunks) / len(chunks)),
        "answers_kept_whole": f"{kept_whole} of {len(evidence_list)}",
        "chunks_starting_mid_sentence": f"{mid_sentence} of {len(chunks)}",
        "chunks_with_rule_heading": f"{with_heading} of {len(chunks)}",
    }


def save_chunks(chunks, way_name):
    """This function saves the chunks, one per line (jsonl)"""

    out_file = PROCESSED_DIR / f"chunks_{way_name}.jsonl"
    out_file.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")


## main function
def main():
    documents = load_documents()
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))

    ## every answer phrase in the golden set, with the document it belongs to
    evidence_list = [src for q in golden for src in q["sources"]]
    print(f"{len(documents)} documents loaded, {len(evidence_list)} answer phrases to check")

    results = {}
    for way_name, cut_function in [("A", way_a), ("B", way_b)]:
        chunks = make_chunks(documents, way_name, cut_function)
        save_chunks(chunks, way_name)
        results[way_name] = measure(chunks, evidence_list)

    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ## results table
    rows = [
        ("How many chunks", "chunks"),
        ("Average chunk size (characters)", "average_size"),
        ("Answers kept whole in one chunk", "answers_kept_whole"),
        ("Chunks starting mid-sentence", "chunks_starting_mid_sentence"),
        ("Chunks that know their rule heading", "chunks_with_rule_heading"),
    ]
    print("\nCHUNKING RESULTS")
    print(f"{'':<38}{'Way A (equal pieces)':<24}Way B (by rule)")
    for label, key in rows:
        print(f"{label:<38}{str(results['A'][key]):<24}{results['B'][key]}")

    print("\nSaved: data/processed/chunks_A.jsonl, data/processed/chunks_B.jsonl")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
