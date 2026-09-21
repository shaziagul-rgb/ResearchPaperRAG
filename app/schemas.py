from pydantic import BaseModel


class Evidence(BaseModel):
    page: int
    text: str
    score: float
    section: str | None = None


class AnalysisItem(BaseModel):
    label: str
    status: str
    confidence: str | None
    evidence: Evidence | None
    alternatives: list[Evidence]
    why: str
    queries: list[str]


class AnalysisResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    evidence_coverage: int
    analysis: list[AnalysisItem]
    evidence_gaps: list[str]