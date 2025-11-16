"""Core package initialization."""
from .schemas import State, IntentClassification, ProductMetadata
from .config import settings
from .prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    PRODUCT_METADATA_EXTRACTION_PROMPT,
    FAQ_ASSISTANT_SYSTEM_PROMPT,
    PRODUCT_ASSISTANT_SYSTEM_PROMPT,
    DEFAULT_CLARIFICATION_MESSAGE,
)

__all__ = [
    "State",
    "IntentClassification",
    "ProductMetadata",
    "settings",
    "INTENT_CLASSIFICATION_PROMPT",
    "PRODUCT_METADATA_EXTRACTION_PROMPT",
    "FAQ_ASSISTANT_SYSTEM_PROMPT",
    "PRODUCT_ASSISTANT_SYSTEM_PROMPT",
    "DEFAULT_CLARIFICATION_MESSAGE",
]
