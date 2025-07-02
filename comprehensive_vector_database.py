# comprehensive_vector_database.py
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import pickle
import sqlite3
from pathlib import Path
from abc import ABC, abstractmethod
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import heapq
import hashlib
from collections import defaultdict

# Advanced imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VectorDocument:
    """Represents a document with its vector embedding and metadata"""
    id: str
    content: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'vector': self.vector.tolist(),
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VectorDocument':
        return cls(
            id=data['id'],
            content=data['content'],
            vector=np.array(data['vector']),
            metadata=data['metadata'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

@dataclass
class SearchResult:
    """Represents a search result with similarity score"""
    document: VectorDocument
    similarity: float
    rank: int
    
class VectorIndex(ABC):
    """Abstract base class for vector indexes"""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        pass
    
    @abstractmethod
    def remove_vector(self, vector_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        pass

class FlatIndex(VectorIndex):
    """Simple flat index using brute force search"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension))
        self.ids = []
        self.id_to_index = {}
        
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        start_idx = len(self.vectors)
        self.vectors = np.vstack([self.vectors, vectors]) if len(self.vectors) > 0 else vectors
        
        for i, vector_id in enumerate(ids):
            self.ids.append(vector_id)
            self.id_to_index[vector_id] = start_idx + i
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if len(self.vectors) == 0:
            return []
        
        similarities = cosine_similarity(
            query_vector.reshape(1, -1), 
            self.vectors
        )[0]
        
        top_k_indices = np.argsort(similarities)[::-1][:k]
        results = [
            (self.ids[idx], similarities[idx])
            for idx in top_k_indices
        ]
        
        return results
    
    def remove_vector(self, vector_id: str) -> bool:
        if vector_id not in self.id_to_index:
            return False
        
        idx = self.id_to_index[vector_id]
        
        # Remove from vectors array
        self.vectors = np.delete(self.vectors, idx, axis=0)
        
        # Remove from ids list
        self.ids.pop(idx)
        
        # Update id_to_index mapping
        del self.id_to_index[vector_id]
        for vid, vidx in self.id_to_index.items():
            if vidx > idx:
                self.id_to_index[vid] = vidx - 1
        
        return True
    
    def get_stats(self) -> Dict:
        return {
            'type': 'FlatIndex',
            'dimension': self.dimension,
            'total_vectors': len(self.vectors),
            'memory_usage_mb': self.vectors.nbytes / (1024 * 1024)
        }

class LSHIndex(VectorIndex):
    """Locality Sensitive Hashing index for approximate search"""
    
    def __init__(self, dimension: int, n_projections: int = 10, n_tables: int = 5):
        self.dimension = dimension
        self.n_projections = n_projections
        self.n_tables = n_tables
        
        # Create random projection matrices
        self.projections = [
            GaussianRandomProjection(n_components=n_projections, random_state=i)
            for i in range(n_tables)
        ]
        
        # Hash tables
        self.hash_tables = [defaultdict(list) for _ in range(n_tables)]
        self.vectors = {}
        self.fitted = False
    
    def _hash_vector(self, vector: np.ndarray) -> List[str]:
        hashes = []
        for i, projection in enumerate(self.projections):
            if not self.fitted:
                # Fit the projection on the first vector
                projection.fit(vector.reshape(1, -1))
            
            projected = projection.transform(vector.reshape(1, -1))[0]
            hash_value = ''.join(['1' if x > 0 else '0' for x in projected])
            hashes.append(hash_value)
        
        return hashes
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        if not self.fitted:
            # Fit all projections on the first batch
            for projection in self.projections:
                projection.fit(vectors)
            self.fitted = True
        
        for vector, vector_id in zip(vectors, ids):
            self.vectors[vector_id] = vector
            hashes = self._hash_vector(vector)
            
            for i, hash_value in enumerate(hashes):
                self.hash_tables[i][hash_value].append(vector_id)
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self.fitted:
            return []
        
        # Get candidate vectors from hash tables
        candidates = set()
        query_hashes = self._hash_vector(query_vector)
        
        for i, query_hash in enumerate(query_hashes):
            candidates.update(self.hash_tables[i][query_hash])
        
        # Calculate similarities for candidates
        similarities = []
        for candidate_id in candidates:
            candidate_vector = self.vectors[candidate_id]
            similarity = cosine_similarity(
                query_vector.reshape(1, -1),
                candidate_vector.reshape(1, -1)
            )[0][0]
            similarities.append((candidate_id, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def remove_vector(self, vector_id: str) -> bool:
        if vector_id not in self.vectors:
            return False
        
        vector = self.vectors[vector_id]
        hashes = self._hash_vector(vector)
        
        # Remove from hash tables
        for i, hash_value in enumerate(hashes):
            if vector_id in self.hash_tables[i][hash_value]:
                self.hash_tables[i][hash_value].remove(vector_id)
        
        # Remove from vectors
        del self.vectors[vector_id]
        return True
    
    def get_stats(self) -> Dict:
        total_buckets = sum(len(table) for table in self.hash_tables)
        avg_bucket_size = np.mean([
            len(bucket) for table in self.hash_tables 
            for bucket in table.values()
        ]) if total_buckets > 0 else 0
        
        return {
            'type': 'LSHIndex',
            'dimension': self.dimension,
            'total_vectors': len(self.vectors),
            'n_projections': self.n_projections,
            'n_tables': self.n_tables,
            'total_buckets': total_buckets,
            'avg_bucket_size': avg_bucket_size
        }

class IVFIndex(VectorIndex):
    """Inverted File index using clustering"""
    
    def __init__(self, dimension: int, n_clusters: int = 8):
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = [[] for _ in range(n_clusters)]
        self.vectors = {}
        self.fitted = False
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        if not self.fitted:
            self.kmeans.fit(vectors)
            self.fitted = True
        
        cluster_assignments = self.kmeans.predict(vectors)
        
        for vector, vector_id, cluster_id in zip(vectors, ids, cluster_assignments):
            self.vectors[vector_id] = vector
            self.clusters[cluster_id].append(vector_id)
    
    def search(self, query_vector: np.ndarray, k: int, n_probe: int = 5) -> List[Tuple[str, float]]:
        if not self.fitted:
            return []
        
        # Find closest clusters
        cluster_distances = euclidean_distances(
            query_vector.reshape(1, -1),
            self.kmeans.cluster_centers_
        )[0]
        
        closest_clusters = np.argsort(cluster_distances)[:n_probe]
        
        # Search in closest clusters
        candidates = []
        for cluster_id in closest_clusters:
            candidates.extend(self.clusters[cluster_id])
        
        # Calculate similarities
        similarities = []
        for candidate_id in candidates:
            candidate_vector = self.vectors[candidate_id]
            similarity = cosine_similarity(
                query_vector.reshape(1, -1),
                candidate_vector.reshape(1, -1)
            )[0][0]
            similarities.append((candidate_id, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def remove_vector(self, vector_id: str) -> bool:
        if vector_id not in self.vectors:
            return False
        
        # Find which cluster contains this vector
        for cluster in self.clusters:
            if vector_id in cluster:
                cluster.remove(vector_id)
                break
        
        del self.vectors[vector_id]
        return True
    
    def get_stats(self) -> Dict:
        cluster_sizes = [len(cluster) for cluster in self.clusters]
        return {
            'type': 'IVFIndex',
            'dimension': self.dimension,
            'total_vectors': len(self.vectors),
            'n_clusters': self.n_clusters,
            'avg_cluster_size': np.mean(cluster_sizes),
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0
        }

class ComprehensiveVectorDatabase:
    """Production-ready vector database with multiple index types"""
    
    def __init__(self, 
                 db_path: str = "./vector_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_type: str = "flat",
                 enable_persistence: bool = True):
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize index
        self.index_type = index_type
        self.index = self._create_index(index_type)
        
        # Document storage
        self.documents = {}
        
        # Metadata database
        self.enable_persistence = enable_persistence
        if enable_persistence:
            self.metadata_db_path = self.db_path / "metadata.db"
            self._init_metadata_db()
        
        # Performance tracking
        self.stats = {
            'total_documents': 0,
            'total_searches': 0,
            'avg_search_time': 0,
            'last_updated': datetime.now()
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized VectorDatabase with {index_type} index, dimension {self.dimension}")
    
    def _create_index(self, index_type: str) -> VectorIndex:
        """Create the appropriate index type"""
        if index_type == "flat":
            return FlatIndex(self.dimension)
        elif index_type == "lsh":
            return LSHIndex(self.dimension)
        elif index_type == "ivf":
            return IVFIndex(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def _init_metadata_db(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                timestamp TEXT,
                vector_norm REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                results_count INTEGER,
                search_time REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_documents(self, 
                     texts: List[str], 
                     metadatas: Optional[List[Dict]] = None,
                     ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to the database"""
        
        with self.lock:
            # Generate IDs if not provided
            if ids is None:
                ids = [self._generate_id(text) for text in texts]
            
            # Generate metadata if not provided
            if metadatas is None:
                metadatas = [{} for _ in texts]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            vectors = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create document objects
            documents = []
            for text, metadata, doc_id, vector in zip(texts, metadatas, ids, vectors):
                doc = VectorDocument(
                    id=doc_id,
                    content=text,
                    vector=vector,
                    metadata=metadata
                )
                documents.append(doc)
                self.documents[doc_id] = doc
            
            # Add to index
            self.index.add_vectors(vectors, ids)
            
            # Persist to database
            if self.enable_persistence:
                self._persist_documents(documents)
            
            # Update stats
            self.stats['total_documents'] += len(documents)
            self.stats['last_updated'] = datetime.now()
            
            logger.info(f"Added {len(documents)} documents to database")
            return ids
    
    def search(self, 
              query: str, 
              k: int = 5,
              filter_metadata: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar documents"""
        
        start_time = time.time()
        
        with self.lock:
            # Generate query embedding
            query_vector = self.embedding_model.encode([query])[0]
            
            # Search in index
            raw_results = self.index.search(query_vector, k * 2)  # Get more for filtering
            
            # Apply metadata filtering
            filtered_results = []
            for doc_id, similarity in raw_results:
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    
                    # Apply metadata filter
                    if filter_metadata:
                        if not self._matches_filter(doc.metadata, filter_metadata):
                            continue
                    
                    filtered_results.append(
                        SearchResult(
                            document=doc,
                            similarity=similarity,
                            rank=len(filtered_results) + 1
                        )
                    )
                    
                    if len(filtered_results) >= k:
                        break
            
            search_time = time.time() - start_time
            
            # Update stats
            self.stats['total_searches'] += 1
            self.stats['avg_search_time'] = (
                (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time) /
                self.stats['total_searches']
            )
            
            # Log search
            if self.enable_persistence:
                self._log_search(query, len(filtered_results), search_time)
            
            logger.info(f"Search completed in {search_time:.3f}s, found {len(filtered_results)} results")
            return filtered_results
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the database"""
        
        with self.lock:
            if doc_id not in self.documents:
                return False
            
            # Remove from index
            success = self.index.remove_vector(doc_id)
            
            if success:
                # Remove from documents
                del self.documents[doc_id]
                
                # Remove from persistent storage
                if self.enable_persistence:
                    self._delete_document_from_db(doc_id)
                
                self.stats['total_documents'] -= 1
                self.stats['last_updated'] = datetime.now()
                
                logger.info(f"Deleted document {doc_id}")
            
            return success
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Retrieve a document by ID"""
        return self.documents.get(doc_id)
    
    def get_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        index_stats = self.index.get_stats()
        
        return {
            'database_stats': self.stats,
            'index_stats': index_stats,
            'model_info': {
                'name': self.embedding_model._modules['0'].auto_model.name_or_path,
                'dimension': self.dimension
            }
        }
    
    def export_data(self, filepath: str) -> None:
        """Export all data to a file"""
        export_data = {
            'documents': [doc.to_dict() for doc in self.documents.values()],
            'stats': self.stats,
            'index_type': self.index_type,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.documents)} documents to {filepath}")
    
    def import_data(self, filepath: str) -> None:
        """Import data from a file"""
        with open(filepath, 'r') as f:
            import_data = json.load(f)
        
        documents = [VectorDocument.from_dict(doc_data) for doc_data in import_data['documents']]
        
        # Add to database
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id for doc in documents]
        
        self.add_documents(texts, metadatas, ids)
        
        logger.info(f"Imported {len(documents)} documents from {filepath}")
    
    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for a document"""
        return hashlib.md5(f"{text}_{datetime.now().isoformat()}".encode()).hexdigest()
    
    def _matches_filter(self, metadata: Dict, filter_metadata: Dict) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def _persist_documents(self, documents: List[VectorDocument]):
        """Persist documents to SQLite database"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        for doc in documents:
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, content, metadata, timestamp, vector_norm)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                doc.id,
                doc.content,
                json.dumps(doc.metadata),
                doc.timestamp.isoformat(),
                np.linalg.norm(doc.vector)
            ))
        
        conn.commit()
        conn.close()
    
    def _log_search(self, query: str, results_count: int, search_time: float):
        """Log search to database"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_logs 
            (query, results_count, search_time, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (
            query,
            results_count,
            search_time,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _delete_document_from_db(self, doc_id: str):
        """Delete document from SQLite database"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
        
        conn.commit()
        conn.close()

# 🧪 Comprehensive Test Suite
def test_comprehensive_vector_database():
    """Test all components of the vector database system"""
    
    print("🚀 Testing Comprehensive Vector Database System\n")
    print("=" * 80)
    
    # Test data
    sample_documents = [
        "Artificial intelligence is transforming the world of technology",
        "Machine learning algorithms can learn from data patterns",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "The weather is sunny and beautiful today",
        "Climate change is affecting global weather patterns",
        "Renewable energy sources are becoming more popular",
        "Electric vehicles are the future of transportation",
        "Sustainable development is crucial for our planet"
    ]
    
    metadatas = [
        {"category": "AI", "topic": "general"},
        {"category": "AI", "topic": "ML"},
        {"category": "AI", "topic": "DL"},
        {"category": "AI", "topic": "NLP"},
        {"category": "AI", "topic": "CV"},
        {"category": "weather", "topic": "current"},
        {"category": "weather", "topic": "climate"},
        {"category": "energy", "topic": "renewable"},
        {"category": "transport", "topic": "electric"},
        {"category": "environment", "topic": "sustainability"}
    ]
    
    # Test different index types
    index_types = ["flat", "lsh", "ivf"]
    
    for index_type in index_types:
        print(f"\n🔧 Testing {index_type.upper()} Index")
        print("-" * 50)
        
        try:
            # Initialize database
            db = ComprehensiveVectorDatabase(
                db_path=f"./test_db_{index_type}",
                index_type=index_type,
                enable_persistence=True
            )
            
            print(f"✅ {index_type.upper()} database initialized")
            
            # Add documents
            doc_ids = db.add_documents(sample_documents, metadatas)
            print(f"✅ Added {len(doc_ids)} documents")
            
            # Test search
            query = "machine learning and artificial intelligence"
            results = db.search(query, k=3)
            
            print(f"\n🔍 Search Results for: '{query}'")
            for result in results:
                print(f"  Rank {result.rank}: {result.document.content[:50]}... (similarity: {result.similarity:.3f})")
            
            # Test filtered search
            ai_results = db.search(query, k=5, filter_metadata={"category": "AI"})
            print(f"\n🎯 Filtered search (AI category): {len(ai_results)} results")
            
            # Test document retrieval
            doc = db.get_document(doc_ids[0])
            print(f"\n📄 Retrieved document: {doc.content[:30]}...")
            
            # Test deletion
            success = db.delete_document(doc_ids[0])
            print(f"\n🗑️ Document deletion: {'Success' if success else 'Failed'}")
            
            # Get statistics
            stats = db.get_stats()
            print(f"\n📊 Database Statistics:")
            print(f"  Total documents: {stats['database_stats']['total_documents']}")
            print(f"  Total searches: {stats['database_stats']['total_searches']}")
            print(f"  Average search time: {stats['database_stats']['avg_search_time']:.3f}s")
            print(f"  Index type: {stats['index_stats']['type']}")
            
        except Exception as e:
            print(f"❌ {index_type.upper()} test failed: {e}")
            continue
    
    print("\n🎉 Vector Database System testing completed!")
    return True

if __name__ == "__main__":
    test_comprehensive_vector_database()