"""
src/agents/academic_worker.py - Academic Literature Contractor Agent
Specialized role for evaluating academic research tasks, bidding via Contract Net Protocol,
and querying arXiv for peer-reviewed literature.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.agents.base import BaseRoleAgent
from src.protocols import CNPTaskAnnouncement, CNPBidProposal, CNPContractAward
from src.tools.search import search_arxiv
from src.tools.pdf_reader import fetch_arxiv_pdf_text, chunk_document_text


class AcademicBidSchema(BaseModel):
    """Structured LLM output schema for Academic Worker bidding."""
    proposed_queries: List[str] = Field(description="2 targeted search queries optimized for ArXiv.")
    confidence: float = Field(description="Confidence rating between 0.8 and 1.0 indicating domain alignment.")
    rationale: str = Field(description="Brief explanation of why these queries target academic literature.")


class AcademicLiteratureAgent(BaseRoleAgent):
    """
    Contractor Agent specialized in Academic Literature & arXiv preprints.
    Participates in Contract Net Protocol bidding for scientific investigation tasks.
    """

    def __init__(self, model_name: str = "gemini-3.6-flash"):
        super().__init__(
            name="AcademicSpecialist",
            role="Contractor / Academic Literature Specialist",
            model_name=model_name
        )

    def evaluate_cfp_and_bid(self, cfp: CNPTaskAnnouncement) -> CNPBidProposal:
        """
        Contract Net Protocol Phase 2 (PROPOSE):
        Evaluates the Coordinator's CFP announcement and submits a competitive bid with proposed queries.
        """
        prompt = f"""You are an elite Academic Literature Specialist reviewing a research task call-for-proposals (CFP).
Objective: "{cfp.objective}"
Scope: "{cfp.scope_description}"

Task:
Formulate 2 highly targeted academic literature search queries optimized for arXiv.
Assign a self-assessed confidence score between 0.80 and 1.0 based on how well academic literature can address this objective.
Provide a concise rationale for your query choices.
"""
        bid_data: AcademicBidSchema = self.invoke_llm(prompt, structured_schema=AcademicBidSchema)

        return CNPBidProposal(
            task_id=cfp.task_id,
            bidder=self.name,
            role=self.role,
            proposed_queries=bid_data.proposed_queries,
            confidence=min(max(bid_data.confidence, 0.5), 1.0),
            rationale=bid_data.rationale
        )

    def deep_read_paper(self, paper: Dict, max_pages: int = 8) -> List[Dict]:
        """
        Deep Paper Reader:
        Downloads the full-text PDF for an arXiv paper and extracts section chunks.
        """
        url = paper.get("url", "")
        full_text = fetch_arxiv_pdf_text(url, max_pages=max_pages)
        if not full_text:
            return []

        chunks = chunk_document_text(full_text, chunk_size=1200, chunk_overlap=150)
        deep_chunks = []
        # Keep top 3 informative chunks
        for idx, ch in enumerate(chunks[:3], 1):
            deep_chunks.append({
                "title": f"{paper.get('title', 'Paper')} [Full-Text Section {idx}]",
                "url": url,
                "published": paper.get("published", "Recent"),
                "content": ch,
                "source_type": "Academic (Full-Text PDF)",
                "retrieved_by": f"{self.name} (Deep PDF Reader)"
            })
        return deep_chunks

    def execute_contract(
        self,
        award: CNPContractAward,
        max_per_query: int = 2,
        deep_read: bool = False
    ) -> List[Dict]:
        """
        Contract Net Protocol Phase 4 (INFORM / EXECUTE):
        Executes awarded arXiv search queries, collects peer-reviewed sources,
        and optionally deep-reads full-text PDFs for key findings.
        """
        collected_sources: List[Dict] = []
        for q in award.awarded_queries:
            results = search_arxiv(q, max_results=max_per_query)
            for r in results:
                r["retrieved_by"] = self.name
            collected_sources.extend(results)

        # Autonomous Deep PDF Reading on top paper
        if deep_read and collected_sources:
            top_paper = collected_sources[0]
            deep_extracts = self.deep_read_paper(top_paper, max_pages=8)
            collected_sources.extend(deep_extracts)

        return collected_sources
