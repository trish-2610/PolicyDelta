"""
Step 9: the shapes of the API requests and responses (FastAPI checks them automatically)
"""

## importing required libraries
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """What the user sends to /ask"""

    question: str = Field(..., examples=["Within how many hours must a breach be reported to the Board?"])


class Source(BaseModel):
    """One citation: where the answer came from"""

    n: int
    title: str
    pages: str
    rule: str


class AskResponse(BaseModel):
    """What /ask sends back"""

    status: str                     ## answered / not_in_documents / rejected_out_of_domain / blocked_injection ...
    answer: str
    sources: list[Source] = []
    disclaimer: str = ""
    from_cache: bool = False
    fallback_used: bool = False
    pii_masked: list[str] = []
    seconds: float


class UploadResponse(BaseModel):
    """What /upload sends back"""

    status: str                     ## added / rejected
    message: str
    doc_id: str = ""
    pages_kept: int = 0
    chunks_total: int = 0
    seconds: float = 0.0