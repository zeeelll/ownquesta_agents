import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from .config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    VECTOR_STORE_PATH,
    KNOWLEDGE_BASE_PATH,
    TOP_K_RESULTS
)


class VectorStore:
    """Vector store for OwnQuesta knowledge base using OpenAI embeddings"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        self.vector_store_path = Path(VECTOR_STORE_PATH)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing vector store, otherwise create new one
        if not self.load_vector_store():
            self.create_vector_store()
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts using OpenAI API"""
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
        print("Creating new vector store from knowledge base...")
        
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
        print(f"Vector store created with {len(self.documents)} documents")
    
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
                print(f"Vector store loaded with {len(self.documents)} documents")
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return False
        return False
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity"""
        # Create embedding for query
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
