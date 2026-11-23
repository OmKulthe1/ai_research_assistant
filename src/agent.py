"""
src/agent.py - Role-Based Multi-Agent Orchestrator with Contract Net Protocol (CNP)
Coordinates specialized agents (Director, Academic Specialist, Industry Scout, Quality Auditor)
using FIPA-ACL performatives and dual-tier memory.
"""

from typing import List, Dict, Tuple, Optional, Any
import uuid
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import get_llm, FALLBACK_MODELS
from src.schemas import ResearchPlan
from src.tools.rag import filter_top_k_sources
from src.memory import ShortTermMemory, LongTermMemory
from src.protocols import (
    ACLMessage,
    ACLPerformative,
    CNPTaskAnnouncement,
    CNPBidProposal,
    CNPContractAward
)
from src.agents import (
    DirectorAgent,
    AcademicLiteratureAgent,
    IndustryIntelligenceAgent,
    QualityCriticAgent,
    extract_text
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
_extract_text = extract_text


class AutonomousResearchAgent:
    """
    Multi-Agent Research Coordinator implementing Role-Based Agent Teams
    and the Contract Net Protocol (CNP) for task allocation and negotiation.
    """

    def __init__(self, model_name: str = "gemini-3.6-flash", session_id: Optional[str] = None):
        self.model_name = model_name
        self.current_session_id = session_id

        # Instantiate Role-Based Agent Team
        self.director = DirectorAgent(model_name=model_name)
        self.academic_worker = AcademicLiteratureAgent(model_name=model_name)
        self.industry_worker = IndustryIntelligenceAgent(model_name=model_name)
        self.critic_worker = QualityCriticAgent(model_name=model_name)

        # Communication Log for Contract Net Protocol (FIPA-ACL Messages)
        self.cnp_message_log: List[ACLMessage] = []
        self.reflection_history: List[Dict] = []

        # Dual-Tier Memory Systems
        self.short_term_memory = ShortTermMemory(window_size=6)
        self.long_term_memory = LongTermMemory()

        if session_id:
            self.load_session(session_id)

    def _log_acl_message(
        self,
        performative: ACLPerformative,
        sender: str,
        receiver: str,
        content: Any,
        reply_with: Optional[str] = None
    ) -> ACLMessage:
        """Appends a standardized FIPA-ACL message to the multi-agent communication trace."""
        msg = ACLMessage(
            performative=performative,
            sender=sender,
            receiver=receiver,
            content=content,
            reply_with=reply_with
        )
        self.cnp_message_log.append(msg)
        return msg

    def plan(self, objective: str) -> Tuple[ResearchPlan, List[CNPContractAward]]:
        """
        Executes Contract Net Protocol Phases 1 through 3:
        1. Task Announcement (CFP)
        2. Worker Bidding (PROPOSE)
        3. Contract Awarding (ACCEPT_PROPOSAL)
        """
        # Decompose objective
        decomp = self.director.decompose_objective(objective)

        # Phase 1: Director issues CFP Task Announcements
        announcements = self.director.announce_tasks(objective, decomp)
        acad_task = announcements[0]
        ind_task = announcements[1]

        self._log_acl_message(
            performative=ACLPerformative.CFP,
            sender=self.director.name,
            receiver=self.academic_worker.name,
            content=acad_task.dict()
        )
        self._log_acl_message(
            performative=ACLPerformative.CFP,
            sender=self.director.name,
            receiver=self.industry_worker.name,
            content=ind_task.dict()
        )

        # Phase 2: Workers evaluate CFP and submit bids (PROPOSE)
        acad_bid: CNPBidProposal = self.academic_worker.evaluate_cfp_and_bid(acad_task)
        self._log_acl_message(
            performative=ACLPerformative.PROPOSE,
            sender=self.academic_worker.name,
            receiver=self.director.name,
            content=acad_bid.dict()
        )

        ind_bid: CNPBidProposal = self.industry_worker.evaluate_cfp_and_bid(ind_task)
        self._log_acl_message(
            performative=ACLPerformative.PROPOSE,
            sender=self.industry_worker.name,
            receiver=self.director.name,
            content=ind_bid.dict()
        )

        # Phase 3: Director evaluates proposals and awards contracts (ACCEPT_PROPOSAL)
        awards = self.director.evaluate_bids_and_award([acad_bid, ind_bid])
        for award in awards:
            self._log_acl_message(
                performative=ACLPerformative.ACCEPT_PROPOSAL if award.accepted else ACLPerformative.REJECT_PROPOSAL,
                sender=self.director.name,
                receiver=award.contractor,
                content=award.dict()
            )

        # Assemble unified research plan
        plan = ResearchPlan(
            sub_questions=decomp.sub_questions,
            arxiv_queries=acad_bid.proposed_queries,
            web_queries=ind_bid.proposed_queries
        )

        return plan, awards

    def fetch_sources(
        self,
        awards: List[CNPContractAward],
        max_per_query: int = 2,
        deep_read: bool = False,
        custom_sources: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Contract Net Protocol Phase 4 (EXECUTE & INFORM):
        Awarded contractor agents execute searches in parallel domains and report findings.
        Optionally triggers deep full-text PDF parsing and ingests user-uploaded documents.
        """
        all_sources: List[Dict] = []

        # Ingest custom uploaded documents if provided
        if custom_sources:
            all_sources.extend(custom_sources)
            self._log_acl_message(
                performative=ACLPerformative.INFORM,
                sender="DeepDocReader",
                receiver=self.director.name,
                content=f"Ingested {len(custom_sources)} custom document sections into shared evidence base."
            )

        for award in awards:
            if not award.accepted:
                continue

            if award.contractor == self.academic_worker.name:
                acad_sources = self.academic_worker.execute_contract(
                    award,
                    max_per_query=max_per_query,
                    deep_read=deep_read
                )
                all_sources.extend(acad_sources)
                self._log_acl_message(
                    performative=ACLPerformative.INFORM,
                    sender=self.academic_worker.name,
                    receiver=self.director.name,
                    content=f"Delivered {len(acad_sources)} academic literature sources{' (including full-text PDF sections)' if deep_read else ''}."
                )

            elif award.contractor == self.industry_worker.name:
                ind_sources = self.industry_worker.execute_contract(award, max_per_query=max_per_query)
                all_sources.extend(ind_sources)
                self._log_acl_message(
                    performative=ACLPerformative.INFORM,
                    sender=self.industry_worker.name,
                    receiver=self.director.name,
                    content=f"Delivered {len(ind_sources)} industry news and benchmark sources."
                )

        return all_sources

    def synthesize(self, objective: str, sources: List[Dict]) -> str:
        """Director synthesizes multi-agent intelligence into a cited draft report."""
        draft = self.director.synthesize_report(objective, sources)
        self._log_acl_message(
            performative=ACLPerformative.INFORM,
            sender=self.director.name,
            receiver="AllAgents",
            content=f"Synthesized draft report ({len(draft)} characters) based on {len(sources)} sources."
        )
        return draft

    def critique_and_refine(self, report: str, objective: str, sources: List[Dict]) -> str:
        """Quality Auditor conducts rigorous peer review and citation verification."""
        self._log_acl_message(
            performative=ACLPerformative.REQUEST,
            sender=self.director.name,
            receiver=self.critic_worker.name,
            content="Audit report for grounding, citation fidelity, and technical depth."
        )

        final_report = self.critic_worker.audit_and_refine(report, objective, sources)

        self._log_acl_message(
            performative=ACLPerformative.INFORM,
            sender=self.critic_worker.name,
            receiver=self.director.name,
            content="Peer review completed. Citations verified and final report polished."
        )
        return final_report

    def prepare_plan_for_approval(self, objective: str) -> Dict[str, Any]:
        """
        Human-in-the-Loop Gate (HITL Phase 1):
        Decomposes objective and solicits bids from contractors (CFP -> PROPOSE).
        Pauses before contract award to allow human inspection and modification.
        """
        self.cnp_message_log = []
        self.reflection_history = []

        decomp = self.director.decompose_objective(objective)
        announcements = self.director.announce_tasks(objective, decomp)
        acad_task = announcements[0]
        ind_task = announcements[1]

        self._log_acl_message(
            performative=ACLPerformative.CFP,
            sender=self.director.name,
            receiver=self.academic_worker.name,
            content=acad_task.dict()
        )
        self._log_acl_message(
            performative=ACLPerformative.CFP,
            sender=self.director.name,
            receiver=self.industry_worker.name,
            content=ind_task.dict()
        )

        acad_bid: CNPBidProposal = self.academic_worker.evaluate_cfp_and_bid(acad_task)
        self._log_acl_message(
            performative=ACLPerformative.PROPOSE,
            sender=self.academic_worker.name,
            receiver=self.director.name,
            content=acad_bid.dict()
        )

        ind_bid: CNPBidProposal = self.industry_worker.evaluate_cfp_and_bid(ind_task)
        self._log_acl_message(
            performative=ACLPerformative.PROPOSE,
            sender=self.industry_worker.name,
            receiver=self.director.name,
            content=ind_bid.dict()
        )

        return {
            "objective": objective,
            "sub_questions": decomp.sub_questions,
            "academic_scope": decomp.academic_scope,
            "industry_scope": decomp.industry_scope,
            "arxiv_queries": acad_bid.proposed_queries,
            "web_queries": ind_bid.proposed_queries,
            "acad_confidence": acad_bid.confidence,
            "ind_confidence": ind_bid.confidence,
            "acad_rationale": acad_bid.rationale,
            "ind_rationale": ind_bid.rationale
        }

    def execute_from_approved_plan(
        self,
        objective: str,
        approved_plan: ResearchPlan,
        progress_callback=None,
        max_reflection_loops: int = 2,
        deep_read: bool = False,
        custom_sources: Optional[List[Dict]] = None
    ) -> Tuple[ResearchPlan, List[Dict], str, str]:
        """
        Human-in-the-Loop Gate (HITL Phase 2):
        Takes the human-authorized queries, issues contract awards (ACCEPT_PROPOSAL),
        executes distributed retrieval, and runs the self-correction reflection loop.
        """
        # Phase 3: Contract Award with Approved Queries
        acad_award = CNPContractAward(
            task_id="TASK-ACAD-AUTH",
            contractor=self.academic_worker.name,
            accepted=True,
            awarded_queries=approved_plan.arxiv_queries,
            instructions="Execute authorized academic queries."
        )
        ind_award = CNPContractAward(
            task_id="TASK-IND-AUTH",
            contractor=self.industry_worker.name,
            accepted=True,
            awarded_queries=approved_plan.web_queries,
            instructions="Execute authorized industry queries."
        )

        self._log_acl_message(
            performative=ACLPerformative.ACCEPT_PROPOSAL,
            sender=self.director.name,
            receiver=self.academic_worker.name,
            content=f"Human authorized {len(approved_plan.arxiv_queries)} arXiv queries: {approved_plan.arxiv_queries}"
        )
        self._log_acl_message(
            performative=ACLPerformative.ACCEPT_PROPOSAL,
            sender=self.director.name,
            receiver=self.industry_worker.name,
            content=f"Human authorized {len(approved_plan.web_queries)} Web queries: {approved_plan.web_queries}"
        )

        # Step 2: Distributed Information Retrieval
        if progress_callback:
            progress_callback(f"Awarded contractors executing searches{' and Deep PDF reading' if deep_read else ''}...", 2)
        raw_sources = self.fetch_sources(
            [acad_award, ind_award],
            deep_read=deep_read,
            custom_sources=custom_sources
        )
        filtered_sources = filter_top_k_sources(objective, raw_sources, top_k=7 if (custom_sources or deep_read) else 6)

        # Step 3: Director Synthesis
        if progress_callback:
            progress_callback(f"Director synthesizing intelligence from {len(filtered_sources)} verified sources...", 3)
        current_draft = self.synthesize(objective, filtered_sources)

        # Step 4: Autonomous Self-Correction & Reflection Loop
        all_sources = list(filtered_sources)
        final_report = current_draft

        for loop_idx in range(1, max_reflection_loops + 1):
            if progress_callback:
                progress_callback(f"Quality Auditor running audit loop {loop_idx}/{max_reflection_loops}...", 4)

            self._log_acl_message(
                performative=ACLPerformative.REQUEST,
                sender=self.director.name,
                receiver=self.critic_worker.name,
                content=f"Audit loop {loop_idx}: evaluate grounding, citations, and identify gaps."
            )

            audit_result = self.critic_worker.audit_and_evaluate(
                report=current_draft,
                objective=objective,
                sources=all_sources,
                sub_questions=approved_plan.sub_questions
            )

            record = {
                "iteration": loop_idx,
                "grounding_score": audit_result.grounding_score,
                "citation_density_score": audit_result.citation_density_score,
                "passes_audit": audit_result.passes_audit,
                "feedback": audit_result.critique_feedback,
                "remedial_queries": audit_result.remedial_queries,
                "new_sources_retrieved": 0,
                "status": "Approved" if audit_result.passes_audit else ("Remediated" if loop_idx < max_reflection_loops else "Max Loops Reached")
            }

            if audit_result.passes_audit:
                self._log_acl_message(
                    performative=ACLPerformative.INFORM,
                    sender=self.critic_worker.name,
                    receiver=self.director.name,
                    content=f"Audit PASSED at iteration {loop_idx} (Grounding: {audit_result.grounding_score}%, Density: {audit_result.citation_density_score}%). Report approved."
                )
                final_report = audit_result.refined_report or current_draft
                self.reflection_history.append(record)
                break
            else:
                self._log_acl_message(
                    performative=ACLPerformative.INFORM,
                    sender=self.critic_worker.name,
                    receiver=self.director.name,
                    content=f"Audit FAILED at iteration {loop_idx} (Grounding: {audit_result.grounding_score}%). Feedback: {audit_result.critique_feedback}"
                )

                if loop_idx < max_reflection_loops and audit_result.remedial_queries:
                    if progress_callback:
                        progress_callback(f"Self-Correction: Dispatching remedial retrieval for {len(audit_result.remedial_queries)} identified gap(s)...", 4)

                    remedial_raw = []
                    for q in audit_result.remedial_queries[:2]:
                        q_award = CNPContractAward(
                            task_id=f"REMEDIAL-{loop_idx}",
                            contractor=self.industry_worker.name,
                            accepted=True,
                            awarded_queries=[q],
                            instructions="Retrieve missing evidence to resolve audit gaps"
                        )
                        self._log_acl_message(
                            performative=ACLPerformative.ACCEPT_PROPOSAL,
                            sender=self.director.name,
                            receiver=self.industry_worker.name,
                            content=f"Remedial contract awarded for gap query: '{q}'"
                        )
                        results = self.industry_worker.execute_contract(q_award, max_per_query=2)
                        remedial_raw.extend(results)

                    existing_urls = {s.get("url") for s in all_sources}
                    new_unique = [s for s in remedial_raw if s.get("url") not in existing_urls]
                    all_sources.extend(new_unique)
                    record["new_sources_retrieved"] = len(new_unique)
                    self.reflection_history.append(record)

                    if progress_callback:
                        progress_callback(f"Director integrating {len(new_unique)} remedial sources into report...", 4)

                    current_draft = self.director.incorporate_remedial_evidence(
                        objective=objective,
                        previous_draft=current_draft,
                        feedback=audit_result.critique_feedback,
                        new_sources=new_unique,
                        all_sources=all_sources
                    )
                    final_report = current_draft
                else:
                    final_report = audit_result.refined_report or current_draft
                    self.reflection_history.append(record)
                    break

        # Long-Term Memory Persistence
        session_id = self.current_session_id or f"sess_{uuid.uuid4().hex[:8]}"
        self.current_session_id = session_id
        topic_title = objective[:60] + ("..." if len(objective) > 60 else "")

        self.long_term_memory.save_session(
            session_id=session_id,
            objective=objective,
            topic_title=topic_title,
            model_name=self.model_name,
            plan=approved_plan,
            sources=all_sources,
            final_report=final_report,
            chat_history=self.short_term_memory.get_history(),
            reflection_history=self.reflection_history
        )

        return approved_plan, all_sources, current_draft, final_report

    def run_pipeline(
        self,
        objective: str,
        progress_callback=None,
        max_reflection_loops: int = 2,
        deep_read: bool = False,
        custom_sources: Optional[List[Dict]] = None
    ) -> Tuple[ResearchPlan, List[Dict], str, str]:
        """
        Executes the full Role-Based Multi-Agent lifecycle with Contract Net Protocol (CNP)
        under Full Autonomy mode.
        """
        if progress_callback:
            progress_callback("Broadcasting CFP task announcements and negotiating worker bids...", 1)
        plan_data = self.prepare_plan_for_approval(objective)
        initial_plan = ResearchPlan(
            sub_questions=plan_data["sub_questions"],
            arxiv_queries=plan_data["arxiv_queries"],
            web_queries=plan_data["web_queries"]
        )
        return self.execute_from_approved_plan(
            objective=objective,
            approved_plan=initial_plan,
            progress_callback=progress_callback,
            max_reflection_loops=max_reflection_loops,
            deep_read=deep_read,
            custom_sources=custom_sources
        )


    def load_session(self, session_id: str) -> Optional[Dict]:
        """Loads a prior session from long-term memory and hydrates working memory."""
        session_data = self.long_term_memory.get_session(session_id)
        if session_data:
            self.current_session_id = session_id
            self.short_term_memory.load_from_list(session_data.get("chat_history", []))
            self.reflection_history = session_data.get("reflection_history", [])
        return session_data
        if session_data:
            self.current_session_id = session_id
            self.short_term_memory.load_from_list(session_data.get("chat_history", []))
        return session_data

    def chat_followup(self, user_question: str, report_context: str, sources: List[Dict]) -> str:
        """
        Multi-turn interactive Q&A utilizing Short-Term Working Memory
        grounded in the synthesized research report and cited sources.
        """
        context_sources = ""
        for idx, s in enumerate(sources, 1):
            agent_tag = s.get("retrieved_by", "Worker")
            context_sources += f"[{idx}] {s.get('title', 'Untitled')} ({s.get('source_type', 'Web')}) - by {agent_tag}\n    {s.get('content', '')[:250]}...\n"

        working_dialogue = self.short_term_memory.format_for_prompt()

        prompt = f"""You are an expert scientific and technical research director answering a user's follow-up questions.

Synthesized Research Report Context:
{report_context}

Available Cited Sources:
{context_sources}

Short-Term Working Memory (Recent Conversation):
{working_dialogue}

User Follow-Up Question:
\"{user_question}\"

Instructions:
1. Provide a direct, rigorous, fact-based answer referring specifically to the sections and citation tags [1], [2], etc., where applicable.
2. Maintain conversational continuity with prior questions from working memory.
3. Be concise, structured, and insightful.
"""
        messages = [
            SystemMessage(content="You are a rigorous technical research director with working conversational memory."),
            HumanMessage(content=prompt)
        ]

        # Record user turn
        self.short_term_memory.add_turn("user", user_question)

        response = self.director.invoke_llm(messages)
        answer = extract_text(response.content)

        # Record assistant turn
        self.short_term_memory.add_turn("assistant", answer)

        # Offload to long-term memory
        if self.current_session_id:
            self.long_term_memory.update_chat_history(
                self.current_session_id,
                self.short_term_memory.get_history()
            )

        return answer


# Backward compatibility alias
QuantumResearchAgent = AutonomousResearchAgent
