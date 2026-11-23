"""
src/agents/industry_worker.py - Industry & Market Intelligence Contractor Agent
Specialized role for evaluating commercial research tasks, bidding via Contract Net Protocol,
and querying real-time web intelligence for benchmarks and announcements.
"""

from typing import List, Dict
from pydantic import BaseModel, Field
from src.agents.base import BaseRoleAgent
from src.protocols import CNPTaskAnnouncement, CNPBidProposal, CNPContractAward
from src.tools.search import search_web


class IndustryBidSchema(BaseModel):
    """Structured LLM output schema for Industry Worker bidding."""
    proposed_queries: List[str] = Field(description="2 targeted search queries optimized for live tech news and benchmarks.")
    confidence: float = Field(description="Confidence rating between 0.8 and 1.0 indicating domain alignment.")
    rationale: str = Field(description="Brief explanation of why these queries target industry developments.")


class IndustryIntelligenceAgent(BaseRoleAgent):
    """
    Contractor Agent specialized in Industry Announcements, Hardware Benchmarks, and Tech News.
    Participates in Contract Net Protocol bidding for practical commercial intelligence tasks.
    """

    def __init__(self, model_name: str = "gemini-3.6-flash"):
        super().__init__(
            name="IndustryScout",
            role="Contractor / Industry Intelligence Scout",
            model_name=model_name
        )

    def evaluate_cfp_and_bid(self, cfp: CNPTaskAnnouncement) -> CNPBidProposal:
        """
        Contract Net Protocol Phase 2 (PROPOSE):
        Evaluates the Coordinator's CFP announcement and submits a competitive bid with proposed queries.
        """
        prompt = f"""You are a Senior Industry & Market Intelligence Scout reviewing a research task call-for-proposals (CFP).
Objective: "{cfp.objective}"
Scope: "{cfp.scope_description}"

Task:
Formulate 2 highly targeted web search queries focused on industry breakthroughs, corporate engineering blogs, and benchmark reports.
Assign a self-assessed confidence score between 0.80 and 1.0 based on how well industry intelligence can address this objective.
Provide a concise rationale for your query choices.
"""
        bid_data: IndustryBidSchema = self.invoke_llm(prompt, structured_schema=IndustryBidSchema)

        return CNPBidProposal(
            task_id=cfp.task_id,
            bidder=self.name,
            role=self.role,
            proposed_queries=bid_data.proposed_queries,
            confidence=min(max(bid_data.confidence, 0.5), 1.0),
            rationale=bid_data.rationale
        )

    def execute_contract(self, award: CNPContractAward, max_per_query: int = 2) -> List[Dict]:
        """
        Contract Net Protocol Phase 4 (INFORM / EXECUTE):
        Executes awarded web search queries and collects commercial intelligence.
        """
        collected_sources: List[Dict] = []
        for q in award.awarded_queries:
            results = search_web(q, max_results=max_per_query)
            for r in results:
                r["retrieved_by"] = self.name
            collected_sources.extend(results)
        return collected_sources
