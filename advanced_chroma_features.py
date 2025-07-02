from chromadb.utils import embedding_functions
import chromadb
from chroma_implementation import ChromaVectorDB
from typing import List

class AdvancedChromaFeatures:
    def __init__(self, openai_api_key: str = None):
        self.vdb = ChromaVectorDB()
        self.openai_api_key = openai_api_key
    
    def setup_custom_embedding_function(self):
        """Setup custom embedding function using OpenAI or local sentence transformer"""
        if self.openai_api_key:
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name="text-embedding-ada-002"
            )
            return openai_ef
        else:
            # Use default sentence transformer
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            return sentence_transformer_ef
    
    def create_multi_modal_collection(self):
        """Create collection with custom embedding function"""
        embedding_function = self.setup_custom_embedding_function()
        
        collection = self.vdb.create_collection(
            name="multi_modal_docs",
            embedding_function=embedding_function,
            metadata={
                "description": "Multi-modal document collection",
                "embedding_model": "text-embedding-ada-002" if self.openai_api_key else "all-MiniLM-L6-v2"
            }
        )
        
        return collection
    
    def batch_operations(self, collection_name: str, batch_size: int = 100):
        """Demonstrate batch operations for large datasets"""
        # Generate sample data
        large_dataset = []
        large_metadata = []
        
        for i in range(1000):
            large_dataset.append(f"This is document number {i} with some content about topic {i % 10}")
            large_metadata.append({
                "doc_id": i,
                "topic_id": i % 10,
                "batch": i // batch_size
            })
        
        # Process in batches
        all_ids = []
        for i in range(0, len(large_dataset), batch_size):
            batch_docs = large_dataset[i:i+batch_size]
            batch_meta = large_metadata[i:i+batch_size]
            
            batch_ids = self.vdb.add_documents(
                collection_name=collection_name,
                documents=batch_docs,
                metadatas=batch_meta
            )
            all_ids.extend(batch_ids)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(large_dataset)-1)//batch_size + 1}")
        
        return all_ids
    
    def complex_filtering(self, collection_name: str):
        """Demonstrate complex filtering capabilities"""
        # Multiple filter conditions
        filters = [
            {"topic_id": {"$in": [1, 2, 3]}},  # Topic ID in list
            {"doc_id": {"$gte": 100}},          # Document ID >= 100
            {"batch": {"$lt": 5}}               # Batch < 5
        ]
        
        results = []
        for i, filter_condition in enumerate(filters):
            result = self.vdb.query(
                collection_name=collection_name,
                query_texts=["document content"],
                n_results=10,
                where=filter_condition
            )
            results.append(result)
            print(f"Filter {i+1} returned {len(result['documents'][0])} results")
        
        return results
    
    def similarity_search_with_score_threshold(self, 
                                              collection_name: str,
                                              query: str,
                                              threshold: float = 0.7):
        """Perform similarity search with score threshold"""
        # Get more results than needed
        results = self.vdb.query(
            collection_name=collection_name,
            query_texts=[query],
            n_results=50  # Get more results to filter
        )
        
        # Filter by similarity threshold
        filtered_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Convert distance to similarity (assuming cosine distance)
            similarity = 1 - dist
            if similarity >= threshold:
                filtered_results['documents'][0].append(doc)
                filtered_results['metadatas'][0].append(meta)
                filtered_results['distances'][0].append(dist)
        
        print(f"Found {len(filtered_results['documents'][0])} results above threshold {threshold}")
        return filtered_results

if __name__ == "__main__":
    # Initialize advanced features
    advanced = AdvancedChromaFeatures()
    
    # Create collection with custom embedding
    collection = advanced.create_multi_modal_collection()
    
    # Perform batch operations
    print("Performing batch operations...")
    ids = advanced.batch_operations("multi_modal_docs")
    
    # Complex filtering
    print("\nTesting complex filtering...")
    filter_results = advanced.complex_filtering("multi_modal_docs")
    
    # Similarity search with threshold
    print("\nSimilarity search with threshold...")
    threshold_results = advanced.similarity_search_with_score_threshold(
        "multi_modal_docs",
        "document about topic",
        threshold=0.8
    ) 