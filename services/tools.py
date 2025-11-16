"""Search tools for FAQ and Products."""
from typing import List, Dict, Any, Optional

from core.config import settings
from .vector_store import get_vector_store


def search_faq_tool(query: str) -> List[Dict[str, Any]]:
    """
    Search FAQ knowledge base.
    
    Args:
        query: User's question
        
    Returns:
        List of FAQ entries with questions and answers
    """
    vector_store = get_vector_store()
    
    # Embed query
    query_vec = vector_store.embed_query(query)
    
    # Search
    distances, results = vector_store.search(
        vector_store.faq_index,
        vector_store.faq_metadata,
        query_vec,
        k=settings.FAQ_SEARCH_K
    )
    
    # Add similarity scores
    for i, result in enumerate(results):
        result['similarity_score'] = float(distances[i])
    
    return results


def search_products_tool(
    query: str = "general product search",
    articleType: Optional[str] = None,
    gender: Optional[str] = None,
    baseColour: Optional[str] = None,
    usage: Optional[str] = None,
    season: Optional[str] = None,
    k: int = 8
) -> Dict[str, Any]:
    """
    Search product catalog with metadata filters.
    Returns adaptive results based on count.
    
    Args:
        query: Search query (describes what user is looking for)
        articleType: Product type filter (e.g., 'Shirts', 'Jeans', 'Watches')
        gender: Gender filter (e.g., 'Men', 'Women', 'Boys', 'Girls')
        baseColour: Color filter (e.g., 'Blue', 'Red', 'Black')
        usage: Usage filter (e.g., 'Casual', 'Formal', 'Sports')
        season: Season filter (e.g., 'Summer', 'Winter')
        k: Number of results to return (default 8)
        
    Returns:
        Dict with count, results, and optional available_filters
        
    Example:
        search_products_tool(query="blue shirts", gender="Men", usage="Casual")
    """
    vector_store = get_vector_store()
    
    # Embed query
    query_vec = vector_store.embed_query(query)
    
    # Build filters (exclude None values)
    filters = {}
    if articleType: filters['articleType'] = articleType
    if gender: filters['gender'] = gender
    if baseColour: filters['baseColour'] = baseColour
    if usage: filters['usage'] = usage
    if season: filters['season'] = season
    
    # Search with post-filtering
    distances, results = vector_store.search(
        vector_store.product_index,
        vector_store.product_metadata,
        query_vec,
        k=k,
        filters=filters if filters else None
    )
    
    # Add similarity scores
    for i, result in enumerate(results):
        result['similarity_score'] = float(distances[i]) if i < len(distances) else 0.0
    
    # Adaptive result handling
    count = len(results)
    
    # Analyze missing filters from results
    available_filters = {}
    if count >= 10 and results:  # Only if we have many results
        if not gender:
            genders = list(set([r.get('gender') for r in results[:20] if r.get('gender')]))
            if len(genders) > 1:
                available_filters['gender'] = genders
        
        if not baseColour:
            colors = list(set([r.get('baseColour') for r in results[:20] if r.get('baseColour')]))
            if len(colors) > 3:
                available_filters['baseColour'] = colors[:5]  # Top 5 colors
        
        if not usage:
            usages = list(set([r.get('usage') for r in results[:20] if r.get('usage')]))
            if len(usages) > 1:
                available_filters['usage'] = usages
    
    return {
        "count": count,
        "results": results[:k],  # Return top k
        "available_filters": available_filters if available_filters else None
    }


def get_tools():
    """Get list of all search tools."""
    return [search_faq_tool, search_products_tool]
