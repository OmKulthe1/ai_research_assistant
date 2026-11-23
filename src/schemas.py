from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ResearchPlan(BaseModel):
    """Schema for decomposed research objectives."""
    sub_questions: List[str] = Field(
        description="Sub-topics breaking down the quantum research request into targeted domains."
    )
    arxiv_queries: List[str] = Field(
        description="Search queries tailored for ArXiv quant-ph repository."
    )
    web_queries: List[str] = Field(
        description="Search queries tailored for general web and industry tech announcements."
    )

class SourceItem(BaseModel):
    """Schema for retrieved academic papers and web articles."""
    title: str = Field(description="Title of the paper or article.")
    url: str = Field(description="Direct URL or ArXiv entry ID.")
    published: str = Field(description="Publication date or recency indicator.")
    content: str = Field(description="Text snippet, abstract, or summary.")
    source_type: str = Field(description="Category: 'Academic (ArXiv)' or 'Industry Web'.")

class CritiqueResult(BaseModel):
    """Schema for self-critique evaluation gate."""
    passes_gate: bool = Field(description="True if report meets standards for grounding, citations, and coverage.")
    score: int = Field(description="Quality score out of 10.")
    feedback: str = Field(description="Detailed critique notes or missing aspect recommendations.")
    refined_report: Optional[str] = Field(default=None, description="Polished markdown report if improvements were applied.")

class CritiqueAuditResult(BaseModel):
    """Schema for Quality Auditor's evaluation and reflection guidance."""
    passes_audit: bool = Field(description="True if grounding score >= 85 and all sub-questions have cited evidence.")
    grounding_score: int = Field(ge=0, le=100, description="Percentage of factual claims backed by verified [index] citations (0-100).")
    citation_density_score: int = Field(ge=0, le=100, description="Score evaluating coverage across citations [1], [2], etc (0-100).")
    critique_feedback: str = Field(description="Specific actionable critique detailing missing benchmarks, ungrounded claims, or gaps.")
    remedial_queries: List[str] = Field(default_factory=list, description="Targeted new search queries to fill the gaps if audit fails (1-2 queries).")
    refined_report: Optional[str] = Field(default=None, description="Polished report if approved or improved.")

class ReflectionIterationRecord(BaseModel):
    """Record of a single self-correction iteration cycle."""
    iteration: int
    grounding_score: int
    citation_density_score: int
    passes_audit: bool
    feedback: str
    remedial_queries: List[str] = Field(default_factory=list)
    new_sources_retrieved: int = 0
    status: str = "Needs Improvement"

