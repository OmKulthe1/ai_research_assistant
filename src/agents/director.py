"""
src/agents/director.py - Research Director Coordinator Agent
Acts as the Manager in the Contract Net Protocol: decomposes objectives,
broadcasts CFPs, evaluates bids, awards contracts, and synthesizes findings.
"""

from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.base import BaseRoleAgent, extract_text
from src.protocols import CNPTaskAnnouncement, CNPBidProposal, CNPContractAward


class DecomposedPlanSchema(BaseModel):
    """Schema for Director's objective decomposition."""
    sub_questions: List[str] = Field(description="3 distinct sub-questions exploring core technical dimensions.")
    academic_scope: str = Field(description="Specific research focus suitable for arXiv literature.")
    industry_scope: str = Field(description="Specific research focus suitable for commercial benchmarks and tech news.")


class DirectorAgent(BaseRoleAgent):
    """
    Coordinator Agent responsible for multi-agent leadership:
    - Task decomposition and issuance of Call for Proposals (CFP).
    - Evaluation of worker bids and awarding of contracts (ACCEPT_PROPOSAL).
    - Comprehensive synthesis of intelligence collected by contractors.
    """

    def __init__(self, model_name: str = "gemini-3.6-flash"):
        super().__init__(
            name="ResearchDirector",
            role="Coordinator / Manager",
            model_name=model_name
        )

    def decompose_objective(self, objective: str) -> DecomposedPlanSchema:
        """Decomposes a broad research goal into domain-specific scopes."""
        prompt = f"""You are a Senior Academic & Technical Research Director.
Decompose the following user research objective into distinct, targeted inquiry areas.

Objective: "{objective}"

Task:
1. Formulate 3 distinct sub-questions covering core technical, architectural, or methodology areas.
2. Define an academic research scope tailored for peer-reviewed papers.
3. Define an industry intelligence scope tailored for practical announcements, benchmarks, and production systems.
"""
        return self.invoke_llm(prompt, structured_schema=DecomposedPlanSchema)

    def announce_tasks(self, objective: str, decomposition: DecomposedPlanSchema) -> List[CNPTaskAnnouncement]:
        """
        Contract Net Protocol Phase 1 (CFP Announcement):
        Broadcasts specialized task specifications to candidate contractors.
        """
        academic_task = CNPTaskAnnouncement(
            task_id="TASK-ACAD-01",
            task_type="academic_literature",
            objective=objective,
            scope_description=decomposition.academic_scope
        )

        industry_task = CNPTaskAnnouncement(
            task_id="TASK-IND-02",
            task_type="industry_intelligence",
            objective=objective,
            scope_description=decomposition.industry_scope
        )

        return [academic_task, industry_task]

    def evaluate_bids_and_award(self, bids: List[CNPBidProposal]) -> List[CNPContractAward]:
        """
        Contract Net Protocol Phase 3 (ACCEPT_PROPOSAL Awarding):
        Evaluates submitted contractor bids and awards contracts based on competence and proposal quality.
        """
        awards: List[CNPContractAward] = []

        for bid in bids:
            # High confidence threshold evaluation
            is_accepted = bid.confidence >= 0.6 and len(bid.proposed_queries) > 0
            awards.append(
                CNPContractAward(
                    task_id=bid.task_id,
                    contractor=bid.bidder,
                    accepted=is_accepted,
                    awarded_queries=bid.proposed_queries,
                    instructions=f"Proceed with targeted multi-query retrieval for {bid.role}. Focus on high-relevance evidence."
                )
            )

        return awards

    def synthesize_report(self, objective: str, sources: List[Dict]) -> str:
        """
        Synthesizes intelligence delivered by all contractors into an academic-grade report with citations.
        """
        context_text = ""
        for idx, s in enumerate(sources, 1):
            agent_tag = s.get("retrieved_by", "Worker")
            context_text += (
                f"\n[{idx}] Title: {s.get('title', 'Untitled')}\n"
                f"    Type: {s.get('source_type', 'Source')} | Retrieved By: {agent_tag} | Date: {s.get('published', 'N/A')}\n"
                f"    URL: {s.get('url', '#')}\n"
                f"    Content: {s.get('content', '')}\n"
            )

        prompt = f"""You are a Lead Research Director and Technical Writer.
Generate a structured, academic-grade research synthesis report based strictly on the retrieved context below.

User Objective: "{objective}"

Retrieved Multi-Agent Context Sources:
{context_text}

Report Structure Requirements:
# Autonomous Research Synthesis Report: [Topic Title based on Objective]

## Executive Summary
(Synthesize high-level breakthrough state-of-the-art developments)

## 1. Key Technical & Conceptual Breakthroughs
(Detailed breakdown of primary innovations, methodologies, architectures, or frameworks)

## 2. Industry Milestones vs. Academic Research
(Key advancements from industry leaders alongside peer-reviewed paper findings)

## 3. Open Challenges & Technical Roadblocks
(Current limitations, bottlenecks, or future outlook)

## 4. References & Citation Index
(List every cited source matching index [1], [2], etc., formatted with clickable URLs and publication dates)

CRITICAL INSTRUCTION: Every factual claim, benchmark, or statistic MUST be explicitly backed by a citation index like [1] or [2].
Do NOT make up any ungrounded citations.
"""
        messages = [
            SystemMessage(content="You produce rigorous, fact-grounded technical research summaries."),
            HumanMessage(content=prompt)
        ]

        response = self.invoke_llm(messages)
        return extract_text(response.content)

    def incorporate_remedial_evidence(
        self,
        objective: str,
        previous_draft: str,
        feedback: str,
        new_sources: List[Dict],
        all_sources: List[Dict]
    ) -> str:
        """
        Director integrates remedial evidence and addresses Auditor critique
        in the self-correction reflection loop.
        """
        remedial_context = ""
        for idx, s in enumerate(new_sources, len(all_sources) - len(new_sources) + 1):
            agent_tag = s.get("retrieved_by", "Worker")
            remedial_context += (
                f"\n[{idx}] Title: {s.get('title', 'Untitled')}\n"
                f"    Type: {s.get('source_type', 'Source')} | Retrieved By: {agent_tag}\n"
                f"    URL: {s.get('url', '#')}\n"
                f"    Content: {s.get('content', '')}\n"
            )

        prompt = f"""You are the Lead Research Director refining a research report in an autonomous self-correction cycle.

Research Objective: "{objective}"

Auditor Peer-Review Feedback:
"{feedback}"

Newly Retrieved Remedial Evidence:
{remedial_context if remedial_context else "No new sources; refine existing draft based on critique."}

Previous Draft Report:
{previous_draft}

Refinement Instructions:
1. Address all weaknesses, missing metrics, and ungrounded statements flagged in the Auditor Feedback.
2. Seamlessly weave in the newly retrieved remedial sources, attributing them with their citation indices.
3. Preserve existing accurate citations [1], [2], etc., and expand the References index.
4. Output ONLY the updated, comprehensive Markdown report.
"""
        messages = [
            SystemMessage(content="You are a decisive Research Director executing remedial report refinement based on peer critique."),
            HumanMessage(content=prompt)
        ]
        response = self.invoke_llm(messages)
        return extract_text(response.content)
