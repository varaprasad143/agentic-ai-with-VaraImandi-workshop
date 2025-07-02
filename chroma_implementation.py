import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any
import uuid
from chromadb.utils import embedding_functions

class ChromaVectorDB:
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        self.collections = {}
    
    def create_collection(self, 
                         name: str, 
                         embedding_function=None,
                         metadata: Dict[str, Any] = None):
        """Create a new collection"""
        if embedding_function is None:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        try:
            collection = self.client.create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata=metadata or {}
            )
            self.collections[name] = collection
            print(f"Created collection: {name}")
            return collection
        except Exception as e:
            print(f"Collection {name} might already exist. Getting existing collection.")
            return self.get_collection(name)
    
    def get_collection(self, name: str):
        """Get an existing collection"""
        if name not in self.collections:
            self.collections[name] = self.client.get_collection(name)
        return self.collections[name]
    
    def add_documents(self, 
                     collection_name: str,
                     documents: List[str],
                     metadatas: List[Dict] = None,
                     ids: List[str] = None):
        """Add documents to a collection"""
        collection = self.get_collection(collection_name)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Add documents
        collection.add(
            documents=documents,
            metadatas=metadatas or [{} for _ in documents],
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to {collection_name}")
        return ids
    
    def add_embeddings(self,
                      collection_name: str,
                      embeddings: List[List[float]],
                      metadatas: List[Dict] = None,
                      ids: List[str] = None):
        """Add pre-computed embeddings to a collection"""
        collection = self.get_collection(collection_name)
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]
        
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in embeddings],
            ids=ids
        )
        
        print(f"Added {len(embeddings)} embeddings to {collection_name}")
        return ids
    
    def query(self,
             collection_name: str,
             query_texts: List[str] = None,
             query_embeddings: List[List[float]] = None,
             n_results: int = 10,
             where: Dict = None,
             include: List[str] = None):
        """Query a collection"""
        collection = self.get_collection(collection_name)
        
        # Default include parameters
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        results = collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include
        )
        
        return results
    
    def update_documents(self,
                        collection_name: str,
                        ids: List[str],
                        documents: List[str] = None,
                        metadatas: List[Dict] = None):
        """Update existing documents"""
        collection = self.get_collection(collection_name)
        
        collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Updated {len(ids)} documents in {collection_name}")
    
    def delete_documents(self, collection_name: str, ids: List[str]):
        """Delete documents from a collection"""
        collection = self.get_collection(collection_name)
        collection.delete(ids=ids)
        print(f"Deleted {len(ids)} documents from {collection_name}")
    
    def get_collection_info(self, collection_name: str):
        """Get information about a collection"""
        collection = self.get_collection(collection_name)
        count = collection.count()
        
        return {
            "name": collection_name,
            "count": count,
            "metadata": collection.metadata
        }
    
    def list_collections(self):
        """List all collections"""
        collections = self.client.list_collections()
        return [col.name for col in collections]

if __name__ == "__main__":
    # Initialize vector database
    vdb = ChromaVectorDB()
    
    # Create a collection for documents
    collection_name = "knowledge_base"
    vdb.create_collection(
        name=collection_name,
        metadata={"description": "Knowledge base for AI applications"}
    )
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Reinforcement learning trains agents through interaction with an environment."
    ]
    
    # Add documents with metadata
    metadatas = [
        {"topic": "machine_learning", "difficulty": "beginner"},
        {"topic": "deep_learning", "difficulty": "intermediate"},
        {"topic": "nlp", "difficulty": "beginner"},
        {"topic": "computer_vision", "difficulty": "intermediate"},
        {"topic": "reinforcement_learning", "difficulty": "advanced"}
    ]
    
    doc_ids = vdb.add_documents(
        collection_name=collection_name,
        documents=documents,
        metadatas=metadatas
    )
    
    # Query the collection
    query_results = vdb.query(
        collection_name=collection_name,
        query_texts=["What is neural network learning?"],
        n_results=3
    )
    
    print("\nQuery Results:")
    for i, (doc, metadata, distance) in enumerate(zip(
        query_results['documents'][0],
        query_results['metadatas'][0],
        query_results['distances'][0]
    )):
        print(f"{i+1}. Distance: {distance:.4f}")
        print(f"   Topic: {metadata['topic']}")
        print(f"   Document: {doc}")
        print()
    
    # Filtered query
    filtered_results = vdb.query(
        collection_name=collection_name,
        query_texts=["learning algorithms"],
        n_results=5,
        where={"difficulty": "beginner"}
    )
    
    print("Filtered Results (beginner level):")
    for doc, metadata in zip(
        filtered_results['documents'][0],
        filtered_results['metadatas'][0]
    ):
        print(f"- {metadata['topic']}: {doc}") 