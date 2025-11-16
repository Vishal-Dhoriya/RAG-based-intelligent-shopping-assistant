"""Pydantic schemas for state and structured outputs."""
from typing import Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class IntentClassification(BaseModel):
    """Classify user intent as FAQ or Product search."""
    
    intent_type: Literal["faq", "product"] = Field(
        description="Type of query: faq for questions, product for shopping"
    )
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Brief explanation of classification")


class ProductMetadata(BaseModel):
    """Extract product search metadata."""
    
    search_query: str = Field(description="Refined search query for embedding")
    articleType: Optional[str] = Field(None, description="Shirts, Jeans, Watches, etc.")
    gender: Optional[str] = Field(None, description="Men, Women, Boys, Girls, Unisex")
    baseColour: Optional[str] = Field(None, description="Color preference")
    usage: Optional[str] = Field(None, description="Casual, Formal, Ethnic, Sports")
    season: Optional[str] = Field(None, description="Summer, Winter, Fall, Spring")
    
    can_search: bool = Field(
        description="True if we have enough info to search (even without all filters)"
    )
    needs_clarification: bool = Field(
        description="True only if query is too vague to search at all"
    )
    clarification_question: Optional[str] = Field(
        None,
        description="Question to ask user if needs_clarification=True"
    )


class State(TypedDict, total=False):
    """Graph state with messages and extracted metadata."""
    messages: Annotated[list, add_messages]
    intent: Optional[IntentClassification]
    product_metadata: Optional[ProductMetadata]

