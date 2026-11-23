"""
src/protocols.py - Agent Communication Protocols & Contract Net Protocol (CNP) Schemas
Implements FIPA-ACL style performatives and structured messages for multi-agent coordination.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import datetime


class ACLPerformative(str, Enum):
    """FIPA-ACL communication performatives defining speech-act intentions."""
    CFP = "cfp"                       # Call for Proposals / Task Announcement
    PROPOSE = "propose"               # Bid / Proposal submitted by contractor
    ACCEPT_PROPOSAL = "accept-proposal"  # Manager awards contract to winning bid
    REJECT_PROPOSAL = "reject-proposal"  # Manager declines bid
    INFORM = "inform"                 # Share result, evidence, or belief
    REQUEST = "request"               # Direct request for action
    QUERY_IF = "query-if"             # Verification inquiry


class ACLMessage(BaseModel):
    """
    Standard Agent Communication Language (ACL) message envelope.
    Enables heterogeneous role-based agents to coordinate across decoupled boundaries.
    """
    performative: ACLPerformative
    sender: str
    receiver: str
    content: Any
    ontology: str = "research-coordination"
    reply_with: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%H:%M:%S")
    )


class CNPTaskAnnouncement(BaseModel):
    """Task specification broadcasted by Coordinator in Phase 1 (CFP)."""
    task_id: str
    task_type: str  # 'academic_literature' | 'industry_intelligence'
    objective: str
    scope_description: str


class CNPBidProposal(BaseModel):
    """Proposal bid submitted by Worker Agent in Phase 2 (PROPOSE)."""
    task_id: str
    bidder: str
    role: str
    proposed_queries: List[str] = Field(description="Targeted search queries for this contractor's specialty.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0 based on domain fit.")
    rationale: str = Field(description="Justification for chosen query vectors.")


class CNPContractAward(BaseModel):
    """Contract award issued by Coordinator in Phase 3 (ACCEPT_PROPOSAL)."""
    task_id: str
    contractor: str
    accepted: bool
    awarded_queries: List[str]
    instructions: str
