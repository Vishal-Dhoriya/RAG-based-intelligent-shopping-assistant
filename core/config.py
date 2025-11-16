"""Application configuration and settings"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API keys and model configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    LLM_MODEL: str = "gemini-2.0-flash"
    
    # Embedding model for vector search
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Data paths - FAISS indices stored here
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDICES_DIR: Path = DATA_DIR / "indices"
    
    FAQ_INDEX_PATH: Path = INDICES_DIR / "faq.index"
    FAQ_METADATA_PATH: Path = INDICES_DIR / "faq.metadata"
    PRODUCT_INDEX_PATH: Path = INDICES_DIR / "products.index"
    PRODUCT_METADATA_PATH: Path = INDICES_DIR / "products.metadata"
    
    # Search settings
    DEFAULT_SEARCH_K: int = 11
    FAQ_SEARCH_K: int = 3


settings = Settings()
