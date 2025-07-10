"""
Configuration management for Agentic RAG Solution
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Agentic RAG system"""
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "astellas")
    MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "astellas-web")
    VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "astellas_vector_index")
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multi-qa-mpnet-base-cos-v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-pro")
    
    # Search Configuration
    VECTOR_SEARCH_LIMIT = int(os.getenv("VECTOR_SEARCH_LIMIT", "5"))
    VECTOR_SEARCH_CANDIDATES = int(os.getenv("VECTOR_SEARCH_CANDIDATES", "50"))
    WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
    
    # Evaluation Configuration
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "7.0"))
    MIN_ANSWERABLE_CHUNKS = int(os.getenv("MIN_ANSWERABLE_CHUNKS", "1"))
    MIN_HIGH_QUALITY_CHUNKS = int(os.getenv("MIN_HIGH_QUALITY_CHUNKS", "2"))
    
    # Application Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "10"))
    
    # Streamlit Configuration
    STREAMLIT_THEME = os.getenv("STREAMLIT_THEME", "light")
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Required configurations
        required_configs = [
            ("MONGODB_URI", cls.MONGODB_URI),
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
            ("TAVILY_API_KEY", cls.TAVILY_API_KEY)
        ]
        
        for config_name, config_value in required_configs:
            if not config_value:
                validation_results["errors"].append(f"{config_name} is required but not set")
                validation_results["valid"] = False
        
        # Optional configurations with defaults
        optional_configs = [
            ("MONGODB_DB_NAME", cls.MONGODB_DB_NAME, "astellas"),
            ("MONGODB_COLLECTION_NAME", cls.MONGODB_COLLECTION_NAME, "astellas-web"),
            ("EMBEDDING_MODEL", cls.EMBEDDING_MODEL, "multi-qa-mpnet-base-cos-v1")
        ]
        
        for config_name, config_value, default_value in optional_configs:
            if config_value == default_value:
                validation_results["warnings"].append(f"{config_name} using default value: {default_value}")
        
        return validation_results
    
    @classmethod
    def get_mongodb_config(cls) -> Dict[str, str]:
        """Get MongoDB configuration"""
        return {
            "uri": cls.MONGODB_URI,
            "db_name": cls.MONGODB_DB_NAME,
            "collection_name": cls.MONGODB_COLLECTION_NAME,
            "index_name": cls.VECTOR_INDEX_NAME
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration"""
        return {
            "vector_limit": cls.VECTOR_SEARCH_LIMIT,
            "vector_candidates": cls.VECTOR_SEARCH_CANDIDATES,
            "web_max_results": cls.WEB_SEARCH_MAX_RESULTS,
            "relevance_threshold": cls.RELEVANCE_THRESHOLD,
            "min_answerable_chunks": cls.MIN_ANSWERABLE_CHUNKS,
            "min_high_quality_chunks": cls.MIN_HIGH_QUALITY_CHUNKS
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, str]:
        """Get model configuration"""
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_model": cls.LLM_MODEL
        }

# Configuration validation on import
if __name__ == "__main__":
    validation = Config.validate()
    
    if validation["valid"]:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration errors found:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    if validation["warnings"]:
        print("⚠️  Configuration warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    print("\n📊 Current Configuration:")
    print(f"  - MongoDB DB: {Config.MONGODB_DB_NAME}")
    print(f"  - Collection: {Config.MONGODB_COLLECTION_NAME}")
    print(f"  - Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"  - LLM Model: {Config.LLM_MODEL}")
    print(f"  - Vector Search Limit: {Config.VECTOR_SEARCH_LIMIT}")
    print(f"  - Relevance Threshold: {Config.RELEVANCE_THRESHOLD}")
