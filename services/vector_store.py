"""Vector store management for FAISS indices"""
import sys
import numpy as np
import joblib
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional, Tuple
from core.config import settings


class VectorStore:
    """Manages FAISS vector stores for FAQ and Products"""
    
    def __init__(self):
        print("Loading vector stores...")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Load FAQ and Product indices
        self.faq_index = faiss.read_index(str(settings.FAQ_INDEX_PATH))
        self.faq_metadata = joblib.load(str(settings.FAQ_METADATA_PATH))
        print(f"FAQ store loaded: {self.faq_index.ntotal} vectors")
        
        self.product_index = faiss.read_index(str(settings.PRODUCT_INDEX_PATH))
        self.product_metadata = joblib.load(str(settings.PRODUCT_METADATA_PATH))
        print(f"Product store loaded: {self.product_index.ntotal} vectors")
    
    def embed_query(self, text: str) -> np.ndarray:
        """Convert text to vector embedding"""
        return self.embedding_model.encode([text])[0]
    
    def search(
        self,
        index: faiss.Index,
        metadata: Dict,
        query_vec: np.ndarray,
        k: int = settings.DEFAULT_SEARCH_K,
        filters: Optional[Dict[str, str]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Search FAISS index with optional metadata filtering
        Returns: (distances, filtered_results)
        """
        if filters:
            # Retrieve more results first, then filter
            search_k = min(k * 10, index.ntotal)
            distances, indices = index.search(
                np.array([query_vec], dtype=np.float32), search_k
            )
            
            # Extract metadata
            metadata_list = metadata.get('metadata_list', metadata)
            id_to_metadata = metadata.get('id_to_metadata', {})
            
            # Collect candidates
            candidates = []
            candidate_distances = []
            for dist, idx in zip(distances[0], indices[0]):
                idx = int(idx)
                item = self._get_metadata_item(metadata_list, id_to_metadata, idx)
                
                if item:
                    candidates.append(item)
                    candidate_distances.append(dist)
            
            # Filter by metadata (case-insensitive: Men == men)
            filtered_results = []
            filtered_distances = []
            for item, dist in zip(candidates, candidate_distances):
                if self._matches_filters(item, filters):
                    filtered_results.append(item)
                    filtered_distances.append(dist)
                    if len(filtered_results) >= k:
                        break
            
            return np.array(filtered_distances), filtered_results
        else:
            # No filters - direct search
            distances, indices = index.search(
                np.array([query_vec], dtype=np.float32), k
            )
            
            metadata_list = metadata.get('metadata_list', metadata)
            id_to_metadata = metadata.get('id_to_metadata', {})
            
            results = []
            for idx in indices[0]:
                idx = int(idx)
                item = self._get_metadata_item(metadata_list, id_to_metadata, idx)
                if item:
                    results.append(item)
            
            return distances[0], results
    
    def _get_metadata_item(
        self,
        metadata_list: Any,
        id_to_metadata: Dict,
        idx: int
    ) -> Optional[Dict[str, Any]]:
        """Get metadata item by index."""
        # Try metadata_list first
        if isinstance(metadata_list, list) and idx < len(metadata_list):
            return metadata_list[idx]
        # Fallback to id_to_metadata
        elif isinstance(metadata_list, dict) and idx in id_to_metadata:
            return id_to_metadata[idx]
        elif idx in id_to_metadata:
            return id_to_metadata[idx]
        return None
    
    def _matches_filters(self, item: Dict[str, Any], filters: Dict[str, str]) -> bool:
        """Check if item matches all filters (case-insensitive)."""
        return all(
            str(item.get(key, '')).lower() == str(value).lower()
            for key, value in filters.items()
        )


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
