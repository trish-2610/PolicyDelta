"""
Settings shared by all steps: folder paths, model names and API keys.
API keys are read from the .env file (never written in the code, never uploaded to GitHub).
"""

## importing required libraries
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


## folder paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
GOLDEN_FILE = EVAL_DIR / "golden_set.json"

## LLM (free API from Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ROUTER_MODEL = "openai/gpt-oss-20b"     ## small and fast: used for routing and query rewriting
ANSWER_MODEL = "openai/gpt-oss-120b"    ## bigger: used to write the final answers (Step 7)