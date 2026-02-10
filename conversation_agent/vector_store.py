import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
import logging
from .config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    VECTOR_STORE_PATH,
    KNOWLEDGE_BASE_PATH,
    TOP_K_RESULTS
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for OwnQuesta knowledge base using OpenAI embeddings"""
    
    def __init__(self):
        # Initialize OpenAI client only when API key is provided
        if OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client: %s", e)
                self.client = None
        else:
            logger.warning("OPENAI_API_KEY not set; vector creation disabled until configured.")
            self.client = None

        self.documents: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.vector_store_path = Path(VECTOR_STORE_PATH)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing vector store; if none and OpenAI client available, create one
        if not self.load_vector_store():
            if self.client:
                self.create_vector_store()
            else:
                logger.info("No existing vector store found and no OpenAI client available. Skipping creation.")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts using OpenAI API"""
        if not self.client:
            raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY to enable embeddings generation.")

        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)
    
    def load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load the OwnQuesta knowledge base from JSON"""
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_vector_store(self):
        """Create vector store from knowledge base"""
        logger.info("Creating new vector store from knowledge base...")
        
        # Load knowledge base
        self.documents = self.load_knowledge_base()
        
        # Create text representations for embedding
        texts = []
        for doc in self.documents:
            # Combine title and content for better context
            text = f"Title: {doc['title']}\n\nContent: {doc['content']}\n\nCategory: {doc['category']}\n\nTags: {', '.join(doc['tags'])}"
            texts.append(text)
        
        # Create embeddings
        self.embeddings = self.create_embeddings(texts)
        
        # Save vector store
        self.save_vector_store()
        logger.info("Vector store created with %d documents", len(self.documents))
    
    def save_vector_store(self):
        """Save vector store to disk"""
        vector_store_file = self.vector_store_path / "vector_store.pkl"
        with open(vector_store_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
    
    def load_vector_store(self) -> bool:
        """Load vector store from disk"""
        vector_store_file = self.vector_store_path / "vector_store.pkl"
        if vector_store_file.exists():
            try:
                with open(vector_store_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.embeddings = data['embeddings']
                logger.info("Vector store loaded with %d documents", len(self.documents))
                return True
            except Exception as e:
                logger.exception("Error loading vector store: %s", e)
                return False
        return False
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity"""
        # Create embedding for query
        if self.embeddings is None:
            logger.debug("No embeddings available for search. Returning empty results.")
            return []

        query_embedding = self.create_embeddings([query])[0]
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Return relevant documents with scores
        results = []
        for idx, score in top_results:
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(score)
            results.append(doc)
        
        return results
    
    def rebuild_vector_store(self):
        """Rebuild vector store from scratch"""
        self.create_vector_store()


# Singleton instance
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Get or create vector store singleton"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
