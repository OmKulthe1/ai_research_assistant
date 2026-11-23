"""
src/agents - Role-Based Multi-Agent Team with Contract Net Protocol (CNP)
"""

from src.agents.base import BaseRoleAgent, extract_text
from src.agents.director import DirectorAgent, DecomposedPlanSchema
from src.agents.academic_worker import AcademicLiteratureAgent
from src.agents.industry_worker import IndustryIntelligenceAgent
from src.agents.critic_worker import QualityCriticAgent

__all__ = [
    "BaseRoleAgent",
    "extract_text",
    "DirectorAgent",
    "DecomposedPlanSchema",
    "AcademicLiteratureAgent",
    "IndustryIntelligenceAgent",
    "QualityCriticAgent",
]
