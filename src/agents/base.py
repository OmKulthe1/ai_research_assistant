"""
src/agents/base.py - Base class for Role-Based Autonomous Agents
Provides common LLM invocation, fallback routing, and ACL message envelope helpers.
"""

from typing import Any, Optional
import logging
from src.config import get_llm, FALLBACK_MODELS
from src.protocols import ACLMessage, ACLPerformative

logger = logging.getLogger(__name__)


def extract_text(content: Any) -> str:
    """Safely extracts normalized text from model content responses."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
        return "\n".join(parts) if parts else str(content)
    elif hasattr(content, "text") and isinstance(content.text, str):
        return content.text
    return str(content)


class BaseRoleAgent:
    """Base class for all role-based agents in the multi-agent system."""

    def __init__(self, name: str, role: str, model_name: str = "gemini-3.6-flash"):
        self.name = name
        self.role = role
        self.model_name = model_name

    def invoke_llm(self, input_data: Any, structured_schema: Optional[Any] = None) -> Any:
        """
        Executes LLM call with automatic multi-model fallback cascade on 429 / resource exhaustion.
        """
        models_to_try = [self.model_name] + [m for m in FALLBACK_MODELS if m != self.model_name]
        last_error = None

        for model in models_to_try:
            try:
                llm_instance = get_llm(model_name=model)
                if structured_schema:
                    structured_llm = llm_instance.with_structured_output(structured_schema)
                    return structured_llm.invoke(input_data)
                else:
                    return llm_instance.invoke(input_data)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    logger.warning(f"[{self.name}] Model '{model}' hit quota limit. Falling back to next engine...")
                    last_error = e
                    continue
                raise e

        raise last_error

    def create_message(
        self,
        performative: ACLPerformative,
        receiver: str,
        content: Any,
        reply_with: Optional[str] = None
    ) -> ACLMessage:
        """Helper to construct a standardized ACL message."""
        return ACLMessage(
            performative=performative,
            sender=self.name,
            receiver=receiver,
            content=content,
            reply_with=reply_with
        )
