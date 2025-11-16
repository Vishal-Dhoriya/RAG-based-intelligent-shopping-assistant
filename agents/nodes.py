"""Graph nodes and routing functions."""
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core import (
    State,
    IntentClassification,
    ProductMetadata,
    settings,
    INTENT_CLASSIFICATION_PROMPT,
    PRODUCT_METADATA_EXTRACTION_PROMPT,
    FAQ_ASSISTANT_SYSTEM_PROMPT,
    PRODUCT_ASSISTANT_SYSTEM_PROMPT,
    DEFAULT_CLARIFICATION_MESSAGE,
)
from services import search_faq_tool, search_products_tool


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    api_key=settings.GOOGLE_API_KEY
)


def classify_intent(state: State) -> dict:
    """
    Classify user intent as FAQ or Product search.
    Uses full conversation history for context-aware classification.
    """
    structured_llm = llm.with_structured_output(IntentClassification)
    
    # Filter messages: only keep HumanMessage and final AIMessage responses
    # Exclude tool calls and tool messages
    conversation_history = []
    for msg in state['messages']:
        if isinstance(msg, HumanMessage):
            conversation_history.append(msg)
        elif isinstance(msg, AIMessage):
            # Handle both string content and list content
            content_text = None
            if isinstance(msg.content, str):
                content_text = msg.content.strip()
            elif isinstance(msg.content, list):
                # Extract text from list of content parts
                text_parts = [part.get('text', '') if isinstance(part, dict) else str(part) for part in msg.content]
                content_text = ' '.join(text_parts).strip()
            
            # Only include if we have actual text content
            if content_text:
                # Create a new AIMessage with string content
                conversation_history.append(AIMessage(content=content_text))
    
    # Add the classification prompt
    messages = conversation_history + [
        HumanMessage(content=INTENT_CLASSIFICATION_PROMPT.format(
            user_message=state['messages'][-1].content
        ))
    ]
    
    intent = structured_llm.invoke(messages)
    
    return {"intent": intent}

def extract_product_metadata(state: State) -> dict:
    """
    Extract product search metadata from user message.
    Uses full conversation history for context-aware extraction.
    """
    structured_llm = llm.with_structured_output(ProductMetadata)
    
    # Filter messages: only keep HumanMessage and final AIMessage responses
    # Exclude tool calls and tool messages
    conversation_history = []
    for msg in state['messages']:
        if isinstance(msg, HumanMessage):
            conversation_history.append(msg)
        elif isinstance(msg, AIMessage):
            # Handle both string content and list content
            content_text = None
            if isinstance(msg.content, str):
                content_text = msg.content.strip()
            elif isinstance(msg.content, list):
                # Extract text from list of content parts
                text_parts = [part.get('text', '') if isinstance(part, dict) else str(part) for part in msg.content]
                content_text = ' '.join(text_parts).strip()
            
            # Only include if we have actual text content
            if content_text:
                # Create a new AIMessage with string content
                conversation_history.append(AIMessage(content=content_text))
    
    # Ensure we have at least the current user message
    if not conversation_history:
        conversation_history.append(state['messages'][-1])
    
    # Add the extraction prompt
    messages = conversation_history + [
        HumanMessage(content=PRODUCT_METADATA_EXTRACTION_PROMPT.format(
            user_message=state['messages'][-1].content
        ))
    ]
    
    metadata = structured_llm.invoke(messages)
    
    return {"product_metadata": metadata}


def ask_clarification(state: State) -> dict:
    """Ask user for clarification when query is too vague."""
    
    metadata = state['product_metadata']
    question = metadata.clarification_question or DEFAULT_CLARIFICATION_MESSAGE
    
    return {"messages": [AIMessage(content=question)]}


def faq_assistant(state: State) -> dict:
    """
    Assistant for FAQ queries.
    Uses search_faq_tool to find answers.
    """
    llm_with_tools = llm.bind_tools([search_faq_tool])
    sys_msg = SystemMessage(content=FAQ_ASSISTANT_SYSTEM_PROMPT)
    
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def product_assistant(state: State) -> dict:
    """
    Assistant for product search.
    Uses search_products_tool with extracted metadata.
    """
    llm_with_tools = llm.bind_tools([search_products_tool])
    
    metadata = state.get('product_metadata')
    
    # Build context for assistant
    context = ""
    search_query = ""
    if metadata:
        search_query = metadata.search_query
        context = f"""
User is looking for:
- Search query: {metadata.search_query}
- Article type: {metadata.articleType or 'any'}
- Gender: {metadata.gender or 'any'}
- Color: {metadata.baseColour or 'any'}
- Usage: {metadata.usage or 'any'}
"""
    
    sys_msg = SystemMessage(content=PRODUCT_ASSISTANT_SYSTEM_PROMPT.format(
        context=context,
        search_query=search_query
    ))
    
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Routing functions
def route_by_intent(state: State) -> Literal["faq_assistant", "extract_product_metadata"]:
    """Route based on FAQ vs Product intent."""
    if state['intent'].intent_type == "faq":
        return "faq_assistant"
    return "extract_product_metadata"


def route_by_metadata(state: State) -> Literal["ask_clarification", "product_assistant"]:
    """Route based on whether clarification is needed."""
    metadata = state.get('product_metadata')
    if metadata and metadata.needs_clarification:
        return "ask_clarification"
    return "product_assistant"
