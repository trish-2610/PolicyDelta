"""
Step 6: Guardrails (security checks that run BEFORE anything is sent to an LLM)

    1. Input check     : reject empty or very long questions
    2. PII masking     : hide Aadhaar, PAN, phone numbers and emails  ->  [AADHAAR], [PAN], [PHONE], [EMAIL]
                         so personal data never leaves the laptop (not to the LLM, not to logs, not to the cache)
    3. Injection check : block questions that try to trick the system
                         ("ignore your instructions", "reveal your system prompt", "you are now DAN" ...)

These are fast, free rules (no LLM). The router in query_processing.py adds a second, smarter check with an LLM.
"""

## importing required libraries
import re


## settings
MIN_CHARS = 3
MAX_CHARS = 1000

## personal data patterns (order matters: email and Aadhaar first, so their digits are not taken as a phone number)
PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),          ## 12 digits, does not start with 0 or 1
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),                            ## e.g. ABCDE1234F
    ("PHONE", re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b")),        ## Indian mobile, 10 digits
]

## phrases used to trick an AI system (written as general patterns, not copies of the test questions)
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|rules?|prompts?|documents?|guidelines|safety)\b",
    r"\b(reveal|print|show|repeat|display|leak)\b.{0,40}\b(system prompt|hidden instructions?|instructions?|prompt)\b",
    r"\brepeat everything\b",
    r"\byou are now\b",
    r"\bfrom now on\b",
    r"\b(pretend|act as|role-?play)\b",
    r"\b(DAN|jailbreak|developer mode)\b",
    r"\bsystem override\b",
    r"\b(no|without|do not give|don't give)\b.{0,15}\bcitations?\b",
]
INJECTION_REGEX = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def check_input(question):
    """Returns (ok, reason). Rejects empty or very long questions."""

    text = (question or "").strip()
    if len(text) < MIN_CHARS:
        return False, "question is empty or too short"
    if len(text) > MAX_CHARS:
        return False, f"question is longer than {MAX_CHARS} characters"
    return True, ""


def mask_pii(question):
    """Replaces personal data with a label. Returns (masked_text, list of what was found)."""

    found = []
    masked = question
    for label, pattern in PII_PATTERNS:
        if pattern.search(masked):
            found.append(label.lower())
            masked = pattern.sub(f"[{label}]", masked)
    return masked, found


def detect_injection(question):
    """Returns (is_attack, the pattern that matched)."""

    for pattern in INJECTION_REGEX:
        match = pattern.search(question)
        if match:
            return True, match.group(0)
    return False, ""