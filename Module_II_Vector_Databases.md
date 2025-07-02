# Module II: Vector Databases and Semantic Search

## 📚 Learning Objectives

By the end of this module, you will:
- ✅ Master vector database fundamentals and embedding concepts
- ✅ Implement production-ready vector storage solutions
- ✅ Build efficient similarity search systems with optimized indexing
- ✅ Design robust embedding pipelines for diverse data types
- ✅ Deploy scalable vector databases with monitoring and performance optimization
- ✅ Create hands-on projects demonstrating real-world semantic search capabilities

---

## 🛠️ Prerequisites and Complete Setup Guide

### System Requirements
- **Python**: 3.9 or higher (3.11 recommended)
- **RAM**: 16GB minimum (32GB recommended for large datasets)
- **Storage**: 25GB free disk space
- **GPU**: Optional but recommended for large-scale embeddings
- **Internet**: Stable connection for model downloads

### 🚀 Step-by-Step Environment Setup

#### Step 1: Create and Activate Virtual Environment
```bash
# Create a dedicated virtual environment
python -m venv vector_db_env

# Activate the environment
# On macOS/Linux:
source vector_db_env/bin/activate
# On Windows:
vector_db_env\Scripts\activate

# Verify activation
which python
```

#### Step 2: Install Core Dependencies
```bash
# Update pip first
pip install --upgrade pip

# Vector databases
pip install chromadb==0.4.22
pip install pinecone-client==2.2.4
pip install weaviate-client==3.25.3
pip install qdrant-client==1.7.0
pip install faiss-cpu==1.7.4  # or faiss-gpu for GPU support

# Embedding models and frameworks
pip install sentence-transformers==2.2.2
pip install transformers==4.36.2
pip install torch==2.1.2
pip install openai==1.6.1
pip install cohere==4.37

# Data processing and utilities
pip install numpy==1.24.3
pip install pandas==2.1.4
pip install scikit-learn==1.3.2
pip install tiktoken==0.5.2

# Document processing
pip install pypdf==3.17.4
pip install python-docx==0.8.11
pip install beautifulsoup4==4.12.2
pip install unstructured==0.11.8
pip install langchain==0.1.0
pip install langchain-community==0.0.10

# Visualization and monitoring
pip install plotly==5.17.0
pip install matplotlib==3.8.2
pip install streamlit==1.29.0
pip install seaborn==0.13.0

# Performance and utilities
pip install python-dotenv==1.0.0
pip install pydantic==2.5.2
pip install requests==2.31.0
pip install tqdm==4.66.1
```

#### Step 3: Environment Configuration
Create a `.env` file in your project root:
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Vector Database Services
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env_here
WEAVIATE_URL=your_weaviate_url_here
WEAVIATE_API_KEY=your_weaviate_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_key_here

# Application Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_RESULTS=10
```

#### Step 4: Verify Installation
```python
# test_vector_setup.py - Run this to verify your setup
import sys
import os
import numpy as np
from datetime import datetime

def test_vector_db_installation():
    """Comprehensive installation test for vector database components"""
    print("🔍 Testing Vector Database Setup...\n")
    print(f"Python version: {sys.version}")
    print(f"Test time: {datetime.now()}\n")
    
    # Test core imports
    tests = [
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("openai", "OpenAI"),
        ("pinecone", "Pinecone"),
        ("weaviate", "Weaviate"),
        ("qdrant_client", "Qdrant"),
        ("plotly", "Plotly"),
        ("streamlit", "Streamlit")
    ]
    
    results = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"✅ {name}: Installed")
            results.append(True)
        except ImportError as e:
            print(f"❌ {name}: Not installed - {e}")
            results.append(False)
    
    # Test vector operations
    print("\n🧮 Testing Vector Operations:")
    try:
        # Test basic vector operations
        vectors = np.random.random((1000, 384)).astype('float32')
        query = np.random.random((1, 384)).astype('float32')
        
        # Test similarity computation
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query, vectors)
        print(f"✅ Vector similarity computation: {similarities.shape}")
        
        # Test FAISS if available
        try:
            import faiss
            index = faiss.IndexFlatIP(384)
            index.add(vectors)
            distances, indices = index.search(query, 5)
            print(f"✅ FAISS indexing and search: Found {len(indices[0])} results")
        except Exception as e:
            print(f"⚠️ FAISS test failed: {e}")
        
        # Test embedding model if available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            test_embeddings = model.encode(["Hello world", "Test sentence"])
            print(f"✅ Sentence Transformers: Generated embeddings shape {test_embeddings.shape}")
        except Exception as e:
            print(f"⚠️ Sentence Transformers test failed: {e}")
            
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        results.append(False)
    else:
        results.append(True)
    
    # Test environment variables
    print("\n🔑 Environment Variables:")
    env_vars = ["OPENAI_API_KEY", "COHERE_API_KEY", "PINECONE_API_KEY", "HUGGINGFACE_API_TOKEN"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set (length: {len(value)})")
        else:
            print(f"⚠️ {var}: Not set")
    
    # Summary
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Setup Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Vector database setup verification complete! Ready to proceed.")
        return True
    else:
        print("⚠️ Some components failed. Please check the installation.")
        return False

if __name__ == "__main__":
    test_vector_db_installation()
```

#### Step 5: Create Project Structure
```bash
# Create organized project structure
mkdir -p vector_db_project/{
data/{raw,processed,embeddings,indexes},
src/{models,utils,databases,search},
notebooks,
tests,
configs,
logs,
outputs,
benchmarks
}

# Create essential files
touch vector_db_project/{__init__.py,main.py,requirements.txt}
touch vector_db_project/src/{__init__.py,models/__init__.py,utils/__init__.py,databases/__init__.py,search/__init__.py}
```

---

## 1. Understanding Vector Databases: The Foundation of Semantic Search

### The Magic of Embeddings: Teaching Machines to Understand Meaning

Imagine if you could teach a computer to understand that "dog" and "puppy" are related, or that "Paris" and "France" have a connection. This is exactly what embeddings do! They transform words, sentences, images, and other data into numerical vectors that capture semantic meaning in a high-dimensional space.

#### 🎯 Hands-On Exercise 1: Building Your First Embedding System

Let's start with a comprehensive, production-ready embedding system:

```python
# advanced_embedding_system.py
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path

# Core imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
class EmbeddingMetrics:
    """Comprehensive metrics for embedding quality assessment"""
    model_name: str
    dimension: int
    total_embeddings: int
    avg_magnitude: float
    std_magnitude: float
    sparsity_ratio: float
    timestamp: datetime
    
class AdvancedEmbeddingSystem:
    """Production-ready embedding system with comprehensive features"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str = "./embedding_cache",
                 enable_gpu: bool = True):
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize the embedding model
        try:
            device = "cuda" if enable_gpu else "cpu"
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model {model_name} with dimension {self.dimension} on {device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Initialize storage
        self.embeddings_cache = {}
        self.metadata_cache = {}
        
        # Performance tracking
        self.embedding_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def encode_texts(self, 
                    texts: List[str], 
                    batch_size: int = 32,
                    show_progress: bool = True,
                    normalize: bool = True) -> np.ndarray:
        """Encode texts into embeddings with caching and optimization"""
        
        start_time = datetime.now()
        
        # Check cache first
        cached_embeddings = []
        texts_to_encode = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embeddings_cache:
                cached_embeddings.append((i, self.embeddings_cache[cache_key]))
                self.cache_hits += 1
            else:
                texts_to_encode.append(text)
                cache_indices.append(i)
                self.cache_misses += 1
        
        # Encode new texts
        new_embeddings = []
        if texts_to_encode:
            logger.info(f"Encoding {len(texts_to_encode)} new texts...")
            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize
            )
            
            # Cache new embeddings
            for text, embedding in zip(texts_to_encode, new_embeddings):
                cache_key = self._get_cache_key(text)
                self.embeddings_cache[cache_key] = embedding
        
        # Combine cached and new embeddings
        all_embeddings = np.zeros((len(texts), self.dimension))
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place new embeddings
        for cache_idx, embedding in zip(cache_indices, new_embeddings):
            all_embeddings[cache_idx] = embedding
        
        # Track performance
        encoding_time = (datetime.now() - start_time).total_seconds()
        self.embedding_times.append(encoding_time)
        
        logger.info(f"Encoded {len(texts)} texts in {encoding_time:.2f}s")
        logger.info(f"Cache stats - Hits: {self.cache_hits}, Misses: {self.cache_misses}")
        
        return all_embeddings
    
    def find_similar(self, 
                    query: str, 
                    corpus_texts: List[str],
                    top_k: int = 5,
                    similarity_metric: str = "cosine") -> List[Tuple[str, float, int]]:
        """Find most similar texts to query"""
        
        # Encode query and corpus
        query_embedding = self.encode_texts([query])[0]
        corpus_embeddings = self.encode_texts(corpus_texts)
        
        # Calculate similarities
        if similarity_metric == "cosine":
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                corpus_embeddings
            )[0]
        elif similarity_metric == "euclidean":
            distances = euclidean_distances(
                query_embedding.reshape(1, -1), 
                corpus_embeddings
            )[0]
            similarities = 1 / (1 + distances)  # Convert distance to similarity
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (corpus_texts[idx], similarities[idx], idx)
            for idx in top_indices
        ]
        
        return results
    
    def analyze_embedding_space(self, 
                               texts: List[str],
                               labels: Optional[List[str]] = None) -> Dict:
        """Comprehensive analysis of embedding space"""
        
        embeddings = self.encode_texts(texts)
        
        # Basic statistics
        magnitudes = np.linalg.norm(embeddings, axis=1)
        sparsity = np.mean(embeddings == 0)
        
        analysis = {
            "total_embeddings": len(embeddings),
            "dimension": self.dimension,
            "magnitude_stats": {
                "mean": np.mean(magnitudes),
                "std": np.std(magnitudes),
                "min": np.min(magnitudes),
                "max": np.max(magnitudes)
            },
            "sparsity_ratio": sparsity,
            "model_name": self.model_name
        }
        
        # Dimensionality reduction for visualization
        if len(embeddings) > 1:
            # PCA
            pca = PCA(n_components=min(50, len(embeddings)-1))
            pca_embeddings = pca.fit_transform(embeddings)
            analysis["pca_explained_variance"] = pca.explained_variance_ratio_[:10].tolist()
            
            # t-SNE for 2D visualization (if enough samples)
            if len(embeddings) >= 10:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                tsne_embeddings = tsne.fit_transform(embeddings)
                analysis["tsne_embeddings"] = tsne_embeddings.tolist()
        
        return analysis
    
    def visualize_embeddings(self, 
                           texts: List[str],
                           labels: Optional[List[str]] = None,
                           method: str = "tsne",
                           save_path: Optional[str] = None) -> go.Figure:
        """Create interactive visualization of embeddings"""
        
        embeddings = self.encode_texts(texts)
        
        if method == "pca":
            reducer = PCA(n_components=2)
            reduced_embeddings = reducer.fit_transform(embeddings)
            title = f"PCA Visualization of {len(texts)} Embeddings"
        elif method == "tsne":
            perplexity = min(30, len(embeddings) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            reduced_embeddings = reducer.fit_transform(embeddings)
            title = f"t-SNE Visualization of {len(texts)} Embeddings"
        else:
            raise ValueError(f"Unsupported visualization method: {method}")
        
        # Create interactive plot
        fig = go.Figure()
        
        if labels:
            unique_labels = list(set(labels))
            colors = px.colors.qualitative.Set3[:len(unique_labels)]
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                fig.add_trace(go.Scatter(
                    x=reduced_embeddings[mask, 0],
                    y=reduced_embeddings[mask, 1],
                    mode='markers',
                    name=label,
                    text=[texts[j] for j in range(len(texts)) if mask[j]],
                    hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>',
                    marker=dict(color=colors[i % len(colors)], size=8)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                mode='markers',
                text=texts,
                hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>',
                marker=dict(color='blue', size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2",
            hovermode='closest',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def benchmark_similarity_metrics(self, 
                                   query: str,
                                   corpus: List[str],
                                   ground_truth_indices: List[int]) -> Dict:
        """Benchmark different similarity metrics"""
        
        query_embedding = self.encode_texts([query])[0]
        corpus_embeddings = self.encode_texts(corpus)
        
        metrics = {}
        
        # Cosine similarity
        cosine_sims = cosine_similarity(
            query_embedding.reshape(1, -1), 
            corpus_embeddings
        )[0]
        
        # Euclidean distance
        euclidean_dists = euclidean_distances(
            query_embedding.reshape(1, -1), 
            corpus_embeddings
        )[0]
        
        # Dot product
        dot_products = np.dot(corpus_embeddings, query_embedding)
        
        # Calculate precision@k for each metric
        for k in [1, 3, 5, 10]:
            if k <= len(corpus):
                # Cosine
                top_k_cosine = np.argsort(cosine_sims)[::-1][:k]
                precision_cosine = len(set(top_k_cosine) & set(ground_truth_indices)) / k
                
                # Euclidean (lower is better, so we sort ascending)
                top_k_euclidean = np.argsort(euclidean_dists)[:k]
                precision_euclidean = len(set(top_k_euclidean) & set(ground_truth_indices)) / k
                
                # Dot product
                top_k_dot = np.argsort(dot_products)[::-1][:k]
                precision_dot = len(set(top_k_dot) & set(ground_truth_indices)) / k
                
                metrics[f"precision@{k}"] = {
                    "cosine": precision_cosine,
                    "euclidean": precision_euclidean,
                    "dot_product": precision_dot
                }
        
        return metrics
    
    def save_embeddings(self, 
                       texts: List[str], 
                       embeddings: np.ndarray,
                       metadata: Optional[Dict] = None,
                       filename: str = "embeddings.npz") -> str:
        """Save embeddings and metadata to disk"""
        
        filepath = self.cache_dir / filename
        
        save_data = {
            "embeddings": embeddings,
            "texts": texts,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            save_data["metadata"] = metadata
        
        np.savez_compressed(filepath, **save_data)
        logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")
        
        return str(filepath)
    
    def load_embeddings(self, filename: str) -> Tuple[List[str], np.ndarray, Dict]:
        """Load embeddings and metadata from disk"""
        
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        texts = data["texts"].tolist()
        embeddings = data["embeddings"]
        metadata = {
            "model_name": str(data["model_name"]),
            "dimension": int(data["dimension"]),
            "timestamp": str(data["timestamp"])
        }
        
        if "metadata" in data:
            metadata.update(data["metadata"].item())
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")
        
        return texts, embeddings, metadata
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0
        
        return {
            "cache_stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": cache_hit_rate
            },
            "timing_stats": {
                "total_encoding_calls": len(self.embedding_times),
                "avg_encoding_time": np.mean(self.embedding_times) if self.embedding_times else 0,
                "total_encoding_time": sum(self.embedding_times)
            },
            "model_info": {
                "name": self.model_name,
                "dimension": self.dimension
            }
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

# 🧪 Comprehensive Test Suite
def test_advanced_embedding_system():
    """Comprehensive test of the advanced embedding system"""
    
    print("🚀 Testing Advanced Embedding System\n")
    print("=" * 70)
    
    # Initialize system
    try:
        embedding_system = AdvancedEmbeddingSystem(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            enable_gpu=False  # Set to True if you have GPU
        )
        print("✅ Embedding system initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # Test data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Python is a popular programming language",
        "JavaScript is used for web development",
        "The weather is sunny today",
        "It's a beautiful day with clear skies",
        "Database systems store and retrieve data efficiently",
        "Vector databases are optimized for similarity search"
    ]
    
    labels = [
        "animals", "animals", "AI", "AI", "programming", 
        "programming", "weather", "weather", "databases", "databases"
    ]
    
    # Test 1: Basic embedding generation
    print("\n🧮 Test 1: Basic Embedding Generation")
    try:
        embeddings = embedding_system.encode_texts(sample_texts)
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        print(f"✅ Embedding dimension: {embedding_system.dimension}")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False
    
    # Test 2: Similarity search
    print("\n🔍 Test 2: Similarity Search")
    try:
        query = "artificial intelligence and machine learning"
        similar_results = embedding_system.find_similar(query, sample_texts, top_k=3)
        
        print(f"Query: '{query}'")
        print("Most similar texts:")
        for i, (text, similarity, idx) in enumerate(similar_results, 1):
            print(f"  {i}. {text} (similarity: {similarity:.3f})")
    except Exception as e:
        print(f"❌ Similarity search failed: {e}")
        return False
    
    # Test 3: Embedding space analysis
    print("\n📊 Test 3: Embedding Space Analysis")
    try:
        analysis = embedding_system.analyze_embedding_space(sample_texts, labels)
        print(f"✅ Total embeddings: {analysis['total_embeddings']}")
        print(f"✅ Average magnitude: {analysis['magnitude_stats']['mean']:.3f}")
        print(f"✅ Sparsity ratio: {analysis['sparsity_ratio']:.3f}")
        if "pca_explained_variance" in analysis:
            print(f"✅ PCA explained variance (first 3): {analysis['pca_explained_variance'][:3]}")
    except Exception as e:
        print(f"❌ Embedding analysis failed: {e}")
        return False
    
    # Test 4: Caching performance
    print("\n⚡ Test 4: Caching Performance")
    try:
        # First encoding (cache miss)
        start_time = datetime.now()
        embeddings1 = embedding_system.encode_texts(sample_texts[:3])
        first_time = (datetime.now() - start_time).total_seconds()
        
        # Second encoding (cache hit)
        start_time = datetime.now()
        embeddings2 = embedding_system.encode_texts(sample_texts[:3])
        second_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ First encoding: {first_time:.3f}s")
        print(f"✅ Second encoding (cached): {second_time:.3f}s")
        print(f"✅ Speedup: {first_time/second_time:.1f}x")
        
        # Verify embeddings are identical
        assert np.allclose(embeddings1, embeddings2), "Cached embeddings don't match!"
        print("✅ Cache consistency verified")
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False
    
    # Test 5: Save and load embeddings
    print("\n💾 Test 5: Save and Load Embeddings")
    try:
        # Save embeddings
        filepath = embedding_system.save_embeddings(
            sample_texts, 
            embeddings,
            metadata={"test_run": True, "version": "1.0"}
        )
        
        # Load embeddings
        loaded_texts, loaded_embeddings, loaded_metadata = embedding_system.load_embeddings(
            "embeddings.npz"
        )
        
        # Verify
        assert loaded_texts == sample_texts, "Loaded texts don't match!"
        assert np.allclose(embeddings, loaded_embeddings), "Loaded embeddings don't match!"
        assert loaded_metadata["model_name"] == embedding_system.model_name
        
        print("✅ Save and load functionality verified")
        print(f"✅ Loaded metadata: {loaded_metadata['test_run']}")
        
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        return False
    
    # Test 6: Performance statistics
    print("\n📈 Test 6: Performance Statistics")
    try:
        stats = embedding_system.get_performance_stats()
        print(f"✅ Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
        print(f"✅ Average encoding time: {stats['timing_stats']['avg_encoding_time']:.3f}s")
        print(f"✅ Total encoding calls: {stats['timing_stats']['total_encoding_calls']}")
    except Exception as e:
        print(f"❌ Performance stats failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Advanced Embedding System is ready for production.")
    return True

if __name__ == "__main__":
    test_advanced_embedding_system()
```

#### Vector Space Visualization: Seeing Meaning in High Dimensions

<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background with gradient -->
  <defs>
    <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f8f9fa;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#e9ecef;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="#00000020"/>
    </filter>
  </defs>
  
  <rect width="800" height="600" fill="url(#bgGradient)"/>
  
  <!-- Title -->
  <text x="400" y="40" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">High-Dimensional Vector Space</text>
  <text x="400" y="65" text-anchor="middle" font-size="14" fill="#6c757d">Semantic Relationships in Embedding Space</text>
  
  <!-- 3D Coordinate system -->
  <g transform="translate(100,150)">
    <!-- X axis -->
    <line x1="0" y1="300" x2="400" y2="300" stroke="#495057" stroke-width="3"/>
    <polygon points="400,300 390,295 390,305" fill="#495057"/>
    <text x="420" y="305" font-size="14" fill="#495057">Dimension 1</text>
    
    <!-- Y axis -->
    <line x1="0" y1="300" x2="0" y2="50" stroke="#495057" stroke-width="3"/>
    <polygon points="0,50 -5,60 5,60" fill="#495057"/>
    <text x="-80" y="30" font-size="14" fill="#495057">Dimension 2</text>
    
    <!-- Z axis (perspective) -->
    <line x1="0" y1="300" x2="150" y2="200" stroke="#495057" stroke-width="3"/>
    <polygon points="150,200 145,205 140,195" fill="#495057"/>
    <text x="160" y="195" font-size="14" fill="#495057">Dimension 3</text>
  </g>
  
  <!-- Semantic clusters -->
  <!-- Animals cluster -->
  <g id="animals-cluster">
    <circle cx="200" cy="200" r="12" fill="#3498db" opacity="0.8" filter="url(#shadow)"/>
    <circle cx="220" cy="190" r="10" fill="#3498db" opacity="0.7"/>
    <circle cx="210" cy="220" r="11" fill="#3498db" opacity="0.75"/>
    <circle cx="185" cy="210" r="9" fill="#3498db" opacity="0.7"/>
    <text x="230" y="205" font-size="12" font-weight="bold" fill="#2980b9">Animals</text>
    <text x="230" y="220" font-size="10" fill="#34495e">dog, cat, bird</text>
  </g>
  
  <!-- Technology cluster -->
  <g id="tech-cluster">
    <circle cx="500" cy="150" r="12" fill="#e74c3c" opacity="0.8" filter="url(#shadow)"/>
    <circle cx="520" cy="140" r="10" fill="#e74c3c" opacity="0.7"/>
    <circle cx="510" cy="170" r="11" fill="#e74c3c" opacity="0.75"/>
    <circle cx="485" cy="160" r="9" fill="#e74c3c" opacity="0.7"/>
    <text x="530" y="155" font-size="12" font-weight="bold" fill="#c0392b">Technology</text>
    <text x="530" y="170" font-size="10" fill="#34495e">AI, ML, code</text>
  </g>
  
  <!-- Food cluster -->
  <g id="food-cluster">
    <circle cx="350" cy="350" r="12" fill="#27ae60" opacity="0.8" filter="url(#shadow)"/>
    <circle cx="370" cy="340" r="10" fill="#27ae60" opacity="0.7"/>
    <circle cx="360" cy="370" r="11" fill="#27ae60" opacity="0.75"/>
    <circle cx="335" cy="360" r="9" fill="#27ae60" opacity="0.7"/>
    <text x="380" y="355" font-size="12" font-weight="bold" fill="#229954">Food</text>
    <text x="380" y="370" font-size="10" fill="#34495e">pizza, pasta, bread</text>
  </g>
  
  <!-- Distance illustrations -->
  <line x1="200" y1="200" x2="500" y2="150" stroke="#95a5a6" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="350" y="170" text-anchor="middle" font-size="11" fill="#7f8c8d">Large semantic distance</text>
  
  <line x1="200" y1="200" x2="220" y2="190" stroke="#2ecc71" stroke-width="3"/>
  <text x="210" y="180" text-anchor="middle" font-size="11" fill="#27ae60">Small distance</text>
  
  <!-- Similarity regions -->
  <ellipse cx="205" cy="205" rx="40" ry="30" fill="#3498db" opacity="0.1" stroke="#3498db" stroke-width="2" stroke-dasharray="5,5"/>
  <ellipse cx="505" cy="155" rx="40" ry="30" fill="#e74c3c" opacity="0.1" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5"/>
  <ellipse cx="355" cy="355" rx="40" ry="30" fill="#27ae60" opacity="0.1" stroke="#27ae60" stroke-width="2" stroke-dasharray="5,5"/>
  
  <!-- Legend -->
  <g transform="translate(50,480)">
    <rect x="0" y="0" width="700" height="100" fill="white" opacity="0.9" stroke="#dee2e6" rx="5"/>
    <text x="20" y="25" font-size="14" font-weight="bold" fill="#2c3e50">Key Concepts:</text>
    
    <circle cx="30" cy="45" r="6" fill="#3498db"/>
    <text x="45" y="50" font-size="12" fill="#34495e">Similar concepts cluster together</text>
    
    <line x1="250" y1="45" x2="280" y2="45" stroke="#2ecc71" stroke-width="3"/>
    <text x="290" y="50" font-size="12" fill="#34495e">Short distance = High similarity</text>
    
    <line x1="480" y1="45" x2="510" y2="45" stroke="#95a5a6" stroke-width="2" stroke-dasharray="4,2"/>
    <text x="520" y="50" font-size="12" fill="#34495e">Long distance = Low similarity</text>
    
    <text x="20" y="75" font-size="11" fill="#6c757d">Each point represents a word/document as a high-dimensional vector</text>
    <text x="20" y="90" font-size="11" fill="#6c757d">Visualization shows 3D projection of typically 384+ dimensional space</text>
  </g>
</svg>

### 1.2 Vector Database Architecture: Building the Foundation for Semantic Search

Vector databases are the backbone of modern AI applications, designed to handle the unique challenges of high-dimensional vector storage and retrieval. Let's dive deep into their architecture and build a comprehensive understanding through hands-on implementation.

#### 🏗️ Hands-On Exercise 2: Building a Complete Vector Database System

Let's create a production-ready vector database implementation that demonstrates all core concepts:

```python
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
import faiss
import chromadb
from chromadb.config import Settings

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
```

#### Vector Database Architecture Diagram

<svg width="900" height="700" xmlns="http://www.w3.org/2000/svg">
  <!-- Background with gradient -->
  <defs>
    <linearGradient id="architectureBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f8f9fa;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#e9ecef;stop-opacity:1" />
    </linearGradient>
    <filter id="architectureShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="3" dy="3" stdDeviation="4" flood-color="#00000030"/>
    </filter>
  </defs>
  
  <rect width="900" height="700" fill="url(#architectureBg)"/>
  
  <!-- Title -->
  <text x="450" y="40" text-anchor="middle" font-size="26" font-weight="bold" fill="#2c3e50">Vector Database Architecture</text>
  <text x="450" y="65" text-anchor="middle" font-size="16" fill="#6c757d">Production-Ready System Components</text>
  
  <!-- Application Layer -->
  <rect x="50" y="100" width="800" height="70" fill="#6f42c1" opacity="0.15" stroke="#6f42c1" stroke-width="3" rx="15" filter="url(#architectureShadow)"/>
  <text x="450" y="125" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Application Layer</text>
  <text x="450" y="150" text-anchor="middle" font-size="14" fill="#495057">REST API • GraphQL • SDK • Client Libraries</text>
  
  <!-- Query Processing Layer -->
  <rect x="50" y="200" width="800" height="90" fill="#fd7e14" opacity="0.15" stroke="#fd7e14" stroke-width="3" rx="15" filter="url(#architectureShadow)"/>
  <text x="450" y="225" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Query Processing Layer</text>
  
  <!-- Query components -->
  <rect x="80" y="245" width="180" height="35" fill="#fd7e14" opacity="0.3" stroke="#e55a00" rx="8"/>
  <text x="170" y="267" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Query Parser</text>
  
  <rect x="280" y="245" width="180" height="35" fill="#fd7e14" opacity="0.3" stroke="#e55a00" rx="8"/>
  <text x="370" y="267" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Filter Engine</text>
  
  <rect x="480" y="245" width="180" height="35" fill="#fd7e14" opacity="0.3" stroke="#e55a00" rx="8"/>
  <text x="570" y="267" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Result Ranker</text>
  
  <rect x="680" y="245" width="140" height="35" fill="#fd7e14" opacity="0.3" stroke="#e55a00" rx="8"/>
  <text x="750" y="267" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Cache Manager</text>
  
  <!-- Vector Index Layer -->
  <rect x="50" y="320" width="800" height="120" fill="#dc3545" opacity="0.15" stroke="#dc3545" stroke-width="3" rx="15" filter="url(#architectureShadow)"/>
  <text x="450" y="345" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Vector Index Layer</text>
  
  <!-- Index Types -->
  <rect x="80" y="365" width="160" height="60" fill="#dc3545" opacity="0.25" stroke="#c82333" rx="10"/>
  <text x="160" y="385" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">HNSW</text>
  <text x="160" y="405" text-anchor="middle" font-size="11" fill="#495057">Hierarchical</text>
  <text x="160" y="418" text-anchor="middle" font-size="11" fill="#495057">Small World</text>
  
  <rect x="260" y="365" width="160" height="60" fill="#dc3545" opacity="0.25" stroke="#c82333" rx="10"/>
  <text x="340" y="385" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">IVF</text>
  <text x="340" y="405" text-anchor="middle" font-size="11" fill="#495057">Inverted File</text>
  <text x="340" y="418" text-anchor="middle" font-size="11" fill="#495057">Clustering</text>
  
  <rect x="440" y="365" width="160" height="60" fill="#dc3545" opacity="0.25" stroke="#c82333" rx="10"/>
  <text x="520" y="385" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">LSH</text>
  <text x="520" y="405" text-anchor="middle" font-size="11" fill="#495057">Locality Sensitive</text>
  <text x="520" y="418" text-anchor="middle" font-size="11" fill="#495057">Hashing</text>
  
  <rect x="620" y="365" width="160" height="60" fill="#dc3545" opacity="0.25" stroke="#c82333" rx="10"/>
  <text x="700" y="385" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">Flat</text>
  <text x="700" y="405" text-anchor="middle" font-size="11" fill="#495057">Brute Force</text>
  <text x="700" y="418" text-anchor="middle" font-size="11" fill="#495057">Exact Search</text>
  
  <!-- Storage Layer -->
  <rect x="50" y="470" width="800" height="100" fill="#198754" opacity="0.15" stroke="#198754" stroke-width="3" rx="15" filter="url(#architectureShadow)"/>
  <text x="450" y="495" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Storage Layer</text>
  
  <!-- Storage components -->
  <rect x="80" y="515" width="200" height="40" fill="#198754" opacity="0.3" stroke="#146c43" rx="8"/>
  <text x="180" y="540" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Vector Storage</text>
  
  <rect x="300" y="515" width="200" height="40" fill="#198754" opacity="0.3" stroke="#146c43" rx="8"/>
  <text x="400" y="540" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Metadata DB</text>
  
  <rect x="520" y="515" width="200" height="40" fill="#198754" opacity="0.3" stroke="#146c43" rx="8"/>
  <text x="620" y="540" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Backup & Recovery</text>
  
  <!-- Infrastructure Layer -->
  <rect x="50" y="600" width="800" height="70" fill="#6c757d" opacity="0.15" stroke="#6c757d" stroke-width="3" rx="15" filter="url(#architectureShadow)"/>
  <text x="450" y="625" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">Infrastructure Layer</text>
  <text x="450" y="650" text-anchor="middle" font-size="14" fill="#495057">Monitoring • Logging • Security • Load Balancing</text>
  
  <!-- Data flow arrows -->
  <g stroke="#495057" stroke-width="3" fill="#495057">
    <!-- App to Query -->
    <line x1="450" y1="170" x2="450" y2="200"/>
    <polygon points="450,200 445,190 455,190"/>
    
    <!-- Query to Index -->
    <line x1="450" y1="290" x2="450" y2="320"/>
    <polygon points="450,320 445,310 455,310"/>
    
    <!-- Index to Storage -->
    <line x1="450" y1="440" x2="450" y2="470"/>
    <polygon points="450,470 445,460 455,460"/>
    
    <!-- Storage to Infrastructure -->
    <line x1="450" y1="570" x2="450" y2="600"/>
    <polygon points="450,600 445,590 455,590"/>
  </g>
  
  <!-- Performance indicators -->
  <g transform="translate(720,100)">
    <rect x="0" y="0" width="150" height="120" fill="white" opacity="0.95" stroke="#dee2e6" rx="8"/>
    <text x="75" y="20" text-anchor="middle" font-size="12" font-weight="bold" fill="#2c3e50">Performance</text>
    
    <text x="10" y="40" font-size="10" fill="#495057">• Sub-ms search</text>
    <text x="10" y="55" font-size="10" fill="#495057">• Horizontal scaling</text>
    <text x="10" y="70" font-size="10" fill="#495057">• 99.9% availability</text>
    <text x="10" y="85" font-size="10" fill="#495057">• Real-time updates</text>
    <text x="10" y="100" font-size="10" fill="#495057">• ACID compliance</text>
  </g>
</svg>

### 1.3 Similarity Metrics

Different similarity metrics serve different purposes in vector search:

```python
# similarity_metrics.py
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

class SimilarityMetrics:
    def __init__(self):
        # Sample vectors for demonstration
        self.vector_a = np.array([1, 2, 3, 4, 5])
        self.vector_b = np.array([2, 3, 4, 5, 6])
        self.vector_c = np.array([5, 4, 3, 2, 1])
    
    def cosine_similarity(self, v1, v2):
        """Calculate cosine similarity"""
        dot_product = np.dot(v1, v2)
        norm_a = np.linalg.norm(v1)
        norm_b = np.linalg.norm(v2)
        return dot_product / (norm_a * norm_b)
    
    def euclidean_distance(self, v1, v2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((v1 - v2) ** 2))
    
    def manhattan_distance(self, v1, v2):
        """Calculate Manhattan distance"""
        return np.sum(np.abs(v1 - v2))
    
    def dot_product_similarity(self, v1, v2):
        """Calculate dot product similarity"""
        return np.dot(v1, v2)
    
    def compare_all_metrics(self):
        """Compare all similarity metrics"""
        vectors = {
            'A': self.vector_a,
            'B': self.vector_b,
            'C': self.vector_c
        }
        
        print("Vector Comparison Results:")
        print("=" * 50)
        
        for name1, vec1 in vectors.items():
            for name2, vec2 in vectors.items():
                if name1 < name2:  # Avoid duplicate comparisons
                    print(f"\n{name1} vs {name2}:")
                    print(f"  Cosine Similarity: {self.cosine_similarity(vec1, vec2):.4f}")
                    print(f"  Euclidean Distance: {self.euclidean_distance(vec1, vec2):.4f}")
                    print(f"  Manhattan Distance: {self.manhattan_distance(vec1, vec2):.4f}")
                    print(f"  Dot Product: {self.dot_product_similarity(vec1, vec2):.4f}")
    
    def demonstrate_metric_properties(self):
        """Demonstrate when to use different metrics"""
        # Normalized vectors (unit length)
        norm_a = self.vector_a / np.linalg.norm(self.vector_a)
        norm_b = self.vector_b / np.linalg.norm(self.vector_b)
        
        # Vectors with different magnitudes but same direction
        scaled_a = self.vector_a * 10
        
        print("\nMetric Properties Demonstration:")
        print("=" * 40)
        
        print("\n1. Cosine similarity is magnitude-invariant:")
        print(f"   Original vectors: {self.cosine_similarity(self.vector_a, self.vector_b):.4f}")
        print(f"   Scaled vector: {self.cosine_similarity(scaled_a, self.vector_b):.4f}")
        
        print("\n2. Euclidean distance is magnitude-sensitive:")
        print(f"   Original vectors: {self.euclidean_distance(self.vector_a, self.vector_b):.4f}")
        print(f"   Scaled vector: {self.euclidean_distance(scaled_a, self.vector_b):.4f}")
        
        print("\n3. For normalized vectors, cosine similarity and dot product are equivalent:")
        print(f"   Cosine similarity: {self.cosine_similarity(norm_a, norm_b):.4f}")
        print(f"   Dot product: {self.dot_product_similarity(norm_a, norm_b):.4f}")

# Usage
metrics = SimilarityMetrics()
metrics.compare_all_metrics()
metrics.demonstrate_metric_properties()
```

---

## 2. Vector Database Implementation

### 2.1 Choosing the Right Vector Database

Different vector databases excel in different scenarios:

| Database | Best For | Strengths | Limitations |
|----------|----------|-----------|-------------|
| **Pinecone** | Production apps | Managed service, scalability | Cost, vendor lock-in |
| **Weaviate** | Hybrid search | GraphQL, multi-modal | Complexity |
| **Chroma** | Development/prototyping | Simple API, local development | Limited scale |
| **Qdrant** | High performance | Rust-based, filtering | Newer ecosystem |
| **Milvus** | Large scale | Distributed, open source | Setup complexity |
| **FAISS** | Research/custom | Facebook-backed, flexible | No built-in persistence |

### 2.2 Implementing with Chroma

Chroma is excellent for development and small to medium-scale applications:

```python
# chroma_implementation.py
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any
import uuid

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

# Usage example
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
```

### 2.3 Advanced Chroma Features

```python
# advanced_chroma_features.py
from chromadb.utils import embedding_functions
import openai
from typing import List

class AdvancedChromaFeatures:
    def __init__(self, openai_api_key: str = None):
        self.vdb = ChromaVectorDB()
        self.openai_api_key = openai_api_key
    
    def setup_custom_embedding_function(self):
        """Setup custom embedding function using OpenAI"""
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

# Usage example
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
```

---

## 3. Vector Indexing Strategies

### 3.1 Understanding Index Types

Different indexing strategies optimize for different use cases:

#### HNSW (Hierarchical Navigable Small World)

<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="600" height="400" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="300" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">HNSW Index Structure</text>
  
  <!-- Layer 2 (top) -->
  <text x="50" y="70" font-size="14" font-weight="bold" fill="#e74c3c">Layer 2</text>
  <circle cx="150" cy="80" r="15" fill="#e74c3c" opacity="0.7"/>
  <circle cx="350" cy="80" r="15" fill="#e74c3c" opacity="0.7"/>
  <circle cx="500" cy="80" r="15" fill="#e74c3c" opacity="0.7"/>
  <line x1="165" y1="80" x2="335" y2="80" stroke="#e74c3c" stroke-width="2"/>
  <line x1="365" y1="80" x2="485" y2="80" stroke="#e74c3c" stroke-width="2"/>
  
  <!-- Layer 1 (middle) -->
  <text x="50" y="170" font-size="14" font-weight="bold" fill="#f39c12">Layer 1</text>
  <circle cx="100" cy="180" r="12" fill="#f39c12" opacity="0.7"/>
  <circle cx="200" cy="180" r="12" fill="#f39c12" opacity="0.7"/>
  <circle cx="300" cy="180" r="12" fill="#f39c12" opacity="0.7"/>
  <circle cx="400" cy="180" r="12" fill="#f39c12" opacity="0.7"/>
  <circle cx="500" cy="180" r="12" fill="#f39c12" opacity="0.7"/>
  
  <!-- Connections in Layer 1 -->
  <line x1="112" y1="180" x2="188" y2="180" stroke="#f39c12" stroke-width="2"/>
  <line x1="212" y1="180" x2="288" y2="180" stroke="#f39c12" stroke-width="2"/>
  <line x1="312" y1="180" x2="388" y2="180" stroke="#f39c12" stroke-width="2"/>
  <line x1="412" y1="180" x2="488" y2="180" stroke="#f39c12" stroke-width="2"/>
  
  <!-- Layer 0 (bottom) -->
  <text x="50" y="270" font-size="14" font-weight="bold" fill="#27ae60">Layer 0</text>
  <circle cx="80" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="130" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="180" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="230" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="280" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="330" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="380" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="430" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="480" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  <circle cx="530" cy="280" r="10" fill="#27ae60" opacity="0.7"/>
  
  <!-- Dense connections in Layer 0 -->
  <line x1="90" y1="280" x2="120" y2="280" stroke="#27ae60" stroke-width="1"/>
  <line x1="140" y1="280" x2="170" y2="280" stroke="#27ae60" stroke-width="1"/>
  <line x1="190" y1="280" x2="220" y2="280" stroke="#27ae60" stroke-width="1"/>
  <line x1="240" y1="280" x2="270" y2="280" stroke="#27ae60" stroke-width="1"/>
  <line x1="290" y1="280" x2="320" y2="280" stroke="#27ae60" stroke-width="1"/>
  <line x1="340" y1="280" x2="370" y2="280" stroke="#27ae60" stroke-width="1"/>
  <line x1="390" y1="280" x2="420" y2="280" stroke="#27ae60" stroke-width="1"/>
  <line x1="440" y1="280" x2="470" y2="280" stroke="#27ae60" stroke-width="1"/>
  <line x1="490" y1="280" x2="520" y2="280" stroke="#27ae60" stroke-width="1"/>
  
  <!-- Vertical connections between layers -->
  <line x1="150" y1="95" x2="200" y2="168" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="350" y1="95" x2="300" y2="168" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="500" y1="95" x2="500" y2="168" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3"/>
  
  <line x1="200" y1="192" x2="180" y2="270" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="300" y1="192" x2="280" y2="270" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="500" y1="192" x2="480" y2="270" stroke="#95a5a6" stroke-width="1" stroke-dasharray="3,3"/>
  
  <!-- Search path illustration -->
  <text x="50" y="350" font-size="12" fill="#2c3e50">Search starts at top layer and navigates down</text>
  <path d="M 150 80 Q 250 130 300 180 Q 320 230 280 280" stroke="#e74c3c" stroke-width="3" fill="none" stroke-dasharray="5,5"/>
  <text x="200" y="370" font-size="12" fill="#e74c3c">Example search path</text>
</svg>

```python
# hnsw_implementation.py
import numpy as np
import heapq
from typing import List, Tuple, Dict
import random

class HNSWIndex:
    def __init__(self, dimension: int, max_connections: int = 16, ef_construction: int = 200):
        self.dimension = dimension
        self.max_connections = max_connections  # M parameter
        self.ef_construction = ef_construction  # efConstruction parameter
        self.max_level = 0
        self.entry_point = None
        
        # Storage for vectors and graph structure
        self.vectors = {}  # id -> vector
        self.levels = {}   # id -> level
        self.graph = {}    # level -> {id -> [connected_ids]}
    
    def _get_random_level(self) -> int:
        """Generate random level for new node"""
        level = 0
        while random.random() < 0.5 and level < 16:
            level += 1
        return level
    
    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance between vectors"""
        return np.linalg.norm(vec1 - vec2)
    
    def _search_layer(self, query: np.ndarray, entry_points: List[int], 
                     num_closest: int, level: int) -> List[Tuple[float, int]]:
        """Search for closest points in a specific layer"""
        visited = set()
        candidates = []
        w = []
        
        # Initialize with entry points
        for ep in entry_points:
            if ep in self.vectors:
                dist = self._distance(query, self.vectors[ep])
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(w, (-dist, ep))  # Max heap for w
                visited.add(ep)
        
        while candidates:
            current_dist, current = heapq.heappop(candidates)
            
            # If current is farther than the farthest in w, stop
            if w and current_dist > -w[0][0]:
                break
            
            # Check neighbors
            if level in self.graph and current in self.graph[level]:
                for neighbor in self.graph[level][current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist = self._distance(query, self.vectors[neighbor])
                        
                        if len(w) < num_closest:
                            heapq.heappush(candidates, (dist, neighbor))
                            heapq.heappush(w, (-dist, neighbor))
                        elif dist < -w[0][0]:
                            heapq.heappush(candidates, (dist, neighbor))
                            heapq.heappop(w)
                            heapq.heappush(w, (-dist, neighbor))
        
        # Convert max heap to min heap and return
        return [(abs(dist), node_id) for dist, node_id in w]
    
    def add(self, vector: np.ndarray, node_id: int):
        """Add a vector to the index"""
        level = self._get_random_level()
        self.vectors[node_id] = vector
        self.levels[node_id] = level
        
        # Initialize graph structure for this node
        for lev in range(level + 1):
            if lev not in self.graph:
                self.graph[lev] = {}
            self.graph[lev][node_id] = []
        
        # If this is the first node or highest level, update entry point
        if self.entry_point is None or level > self.max_level:
            self.entry_point = node_id
            self.max_level = level
            return
        
        # Search for closest points and create connections
        entry_points = [self.entry_point]
        
        # Search from top to level+1
        for lev in range(self.max_level, level, -1):
            entry_points = [ep[1] for ep in self._search_layer(vector, entry_points, 1, lev)]
        
        # Search and connect from level down to 0
        for lev in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, entry_points, self.ef_construction, lev)
            
            # Select connections
            max_conn = self.max_connections if lev > 0 else self.max_connections * 2
            connections = self._select_neighbors(candidates, max_conn)
            
            # Add bidirectional connections
            for _, neighbor_id in connections:
                self.graph[lev][node_id].append(neighbor_id)
                self.graph[lev][neighbor_id].append(node_id)
            
            entry_points = [conn[1] for conn in connections]
    
    def _select_neighbors(self, candidates: List[Tuple[float, int]], 
                         max_connections: int) -> List[Tuple[float, int]]:
        """Select best neighbors (simple implementation)"""
        candidates.sort(key=lambda x: x[0])
        return candidates[:max_connections]
    
    def search(self, query: np.ndarray, k: int, ef: int = None) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors"""
        if self.entry_point is None:
            return []
        
        if ef is None:
            ef = max(self.ef_construction, k)
        
        entry_points = [self.entry_point]
        
        # Search from top level down to level 1
        for level in range(self.max_level, 0, -1):
            entry_points = [ep[1] for ep in self._search_layer(query, entry_points, 1, level)]
        
        # Search level 0 with ef
        candidates = self._search_layer(query, entry_points, ef, 0)
        
        # Return top k
        candidates.sort(key=lambda x: x[0])
        return candidates[:k]

# Usage example
if __name__ == "__main__":
    # Create HNSW index
    dimension = 128
    index = HNSWIndex(dimension=dimension)
    
    # Generate sample data
    num_vectors = 1000
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    
    # Add vectors to index
    print("Building HNSW index...")
    for i, vector in enumerate(vectors):
        index.add(vector, i)
        if (i + 1) % 100 == 0:
            print(f"Added {i + 1}/{num_vectors} vectors")
    
    # Search
    query = np.random.random(dimension).astype(np.float32)
    results = index.search(query, k=10)
    
    print(f"\nTop 10 nearest neighbors:")
    for i, (distance, node_id) in enumerate(results):
        print(f"{i+1}. Node {node_id}: distance = {distance:.4f}")
    
    # Verify with brute force
    print("\nVerifying with brute force search...")
    brute_force_results = []
    for i, vector in enumerate(vectors):
        dist = np.linalg.norm(query - vector)
        brute_force_results.append((dist, i))
    
    brute_force_results.sort(key=lambda x: x[0])
    
    print("Brute force top 10:")
    for i, (distance, node_id) in enumerate(brute_force_results[:10]):
        print(f"{i+1}. Node {node_id}: distance = {distance:.4f}")
```

### 3.2 IVF (Inverted File) Index

```python
# ivf_implementation.py
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict

class IVFIndex:
    def __init__(self, dimension: int, n_clusters: int = 8):
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.is_trained = False
        
        # Storage
        self.cluster_vectors = {}  # cluster_id -> list of (vector_id, vector)
        self.vector_to_cluster = {}  # vector_id -> cluster_id
        self.centroids = None
    
    def train(self, training_vectors: np.ndarray):
        """Train the index with clustering"""
        print(f"Training IVF index with {len(training_vectors)} vectors...")
        self.kmeans.fit(training_vectors)
        self.centroids = self.kmeans.cluster_centers_
        self.is_trained = True
        
        # Initialize cluster storage
        for i in range(self.n_clusters):
            self.cluster_vectors[i] = []
        
        print(f"Training completed. Created {self.n_clusters} clusters.")
    
    def add(self, vector: np.ndarray, vector_id: int):
        """Add a vector to the index"""
        if not self.is_trained:
            raise ValueError("Index must be trained before adding vectors")
        
        # Find closest cluster
        cluster_id = self.kmeans.predict([vector])[0]
        
        # Add to cluster
        self.cluster_vectors[cluster_id].append((vector_id, vector))
        self.vector_to_cluster[vector_id] = cluster_id
    
    def search(self, query: np.ndarray, k: int, n_probe: int = 1) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors"""
        if not self.is_trained:
            raise ValueError("Index must be trained before searching")
        
        # Find closest clusters to query
        cluster_distances = []
        for i, centroid in enumerate(self.centroids):
            dist = np.linalg.norm(query - centroid)
            cluster_distances.append((dist, i))
        
        cluster_distances.sort(key=lambda x: x[0])
        
        # Search in top n_probe clusters
        candidates = []
        for _, cluster_id in cluster_distances[:n_probe]:
            for vector_id, vector in self.cluster_vectors[cluster_id]:
                dist = np.linalg.norm(query - vector)
                candidates.append((dist, vector_id))
        
        # Return top k
        candidates.sort(key=lambda x: x[0])
        return candidates[:k]
    
    def get_cluster_info(self) -> Dict:
        """Get information about clusters"""
        cluster_sizes = {}
        for cluster_id, vectors in self.cluster_vectors.items():
            cluster_sizes[cluster_id] = len(vectors)
        
        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": cluster_sizes,
            "total_vectors": sum(cluster_sizes.values()),
            "avg_cluster_size": np.mean(list(cluster_sizes.values())),
            "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0
        }

# Performance comparison
class IndexComparison:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vectors = None
        self.query_vectors = None
    
    def generate_data(self, n_vectors: int = 10000, n_queries: int = 100):
        """Generate test data"""
        print(f"Generating {n_vectors} vectors and {n_queries} queries...")
        self.vectors = np.random.random((n_vectors, self.dimension)).astype(np.float32)
        self.query_vectors = np.random.random((n_queries, self.dimension)).astype(np.float32)
    
    def benchmark_brute_force(self, k: int = 10) -> Tuple[float, List]:
        """Benchmark brute force search"""
        import time
        
        start_time = time.time()
        all_results = []
        
        for query in self.query_vectors:
            distances = []
            for i, vector in enumerate(self.vectors):
                dist = np.linalg.norm(query - vector)
                distances.append((dist, i))
            
            distances.sort(key=lambda x: x[0])
            all_results.append(distances[:k])
        
        end_time = time.time()
        return end_time - start_time, all_results
    
    def benchmark_ivf(self, k: int = 10, n_clusters: int = 8, n_probe: int = 1) -> Tuple[float, List]:
        """Benchmark IVF index"""
        import time
        
        # Build index
        build_start = time.time()
        index = IVFIndex(self.dimension, n_clusters)
        index.train(self.vectors[:1000])  # Train on subset
        
        for i, vector in enumerate(self.vectors):
            index.add(vector, i)
        
        build_time = time.time() - build_start
        
        # Search
        search_start = time.time()
        all_results = []
        
        for query in self.query_vectors:
            results = index.search(query, k, n_probe)
            all_results.append(results)
        
        search_time = time.time() - search_start
        total_time = build_time + search_time
        
        print(f"IVF build time: {build_time:.4f}s, search time: {search_time:.4f}s")
        return total_time, all_results
    
    def calculate_recall(self, ground_truth: List, results: List, k: int = 10) -> float:
        """Calculate recall@k"""
        total_recall = 0
        
        for gt, res in zip(ground_truth, results):
            gt_ids = set([item[1] for item in gt[:k]])
            res_ids = set([item[1] for item in res[:k]])
            recall = len(gt_ids.intersection(res_ids)) / len(gt_ids)
            total_recall += recall
        
        return total_recall / len(ground_truth)

# Usage example
if __name__ == "__main__":
    # Create comparison
    comparison = IndexComparison(dimension=64)  # Smaller dimension for faster demo
    comparison.generate_data(n_vectors=5000, n_queries=50)
    
    print("\nBenchmarking search methods...")
    
    # Brute force (ground truth)
    bf_time, bf_results = comparison.benchmark_brute_force(k=10)
    print(f"Brute force time: {bf_time:.4f}s")
    
    # IVF with different parameters
    ivf_configs = [
        {"n_clusters": 50, "n_probe": 1},
        {"n_clusters": 50, "n_probe": 5},
        {"n_clusters": 100, "n_probe": 1},
        {"n_clusters": 100, "n_probe": 5}
    ]
    
    for config in ivf_configs:
        ivf_time, ivf_results = comparison.benchmark_ivf(
            k=10, 
            n_clusters=config["n_clusters"], 
            n_probe=config["n_probe"]
        )
        
        recall = comparison.calculate_recall(bf_results, ivf_results, k=10)
        speedup = bf_time / ivf_time
        
        print(f"IVF (clusters={config['n_clusters']}, probe={config['n_probe']}):")
        print(f"  Time: {ivf_time:.4f}s, Speedup: {speedup:.2f}x, Recall@10: {recall:.3f}")
```

---

## 4. Embedding Strategies

### 4.1 Text Embedding Pipelines

```python
# text_embedding_pipeline.py
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from typing import List, Dict, Any
import tiktoken
from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbeddingPipeline:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def load_sentence_transformer(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load a sentence transformer model"""
        if model_name not in self.models:
            print(f"Loading SentenceTransformer: {model_name}")
            self.models[model_name] = SentenceTransformer(model_name)
        return self.models[model_name]
    
    def load_openai_embeddings(self, api_key: str):
        """Setup OpenAI embeddings"""
        openai.api_key = api_key
        self.tokenizers['openai'] = tiktoken.encoding_for_model("text-embedding-ada-002")
    
    def embed_with_sentence_transformer(self, 
                                       texts: List[str], 
                                       model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """Generate embeddings using SentenceTransformer"""
        model = self.load_sentence_transformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def embed_with_openai(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        
        for text in texts:
            # Check token count
            if 'openai' in self.tokenizers:
                tokens = self.tokenizers['openai'].encode(text)
                if len(tokens) > 8191:  # Max tokens for ada-002
                    text = self.tokenizers['openai'].decode(tokens[:8191])
            
            response = openai.Embedding.create(
                input=text,
                model=model
            )
            embeddings.append(response['data'][0]['embedding'])
        
        return embeddings
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]], 
                         embedding_model: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
        """Process documents with chunking and embedding"""
        processed_docs = []
        
        for doc in documents:
            text = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Generate embeddings for chunks
            if embedding_model.startswith('text-embedding'):
                embeddings = self.embed_with_openai(chunks, embedding_model)
            else:
                embeddings = self.embed_with_sentence_transformer(chunks, embedding_model)
            
            # Create processed documents
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                processed_doc = {
                    'id': f"{doc.get('id', 'unknown')}_{i}",
                    'content': chunk,
                    'embedding': embedding,
                    'metadata': {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_doc_id': doc.get('id', 'unknown')
                    }
                }
                processed_docs.append(processed_doc)
        
        return processed_docs

# Usage example
if __name__ == "__main__":
    pipeline = TextEmbeddingPipeline()
    
    # Sample documents
    documents = [
        {
            'id': 'doc1',
            'content': "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience, without being explicitly programmed. The field draws from various disciplines including computer science, statistics, and mathematics.",
            'metadata': {'source': 'textbook', 'chapter': 1}
        },
        {
            'id': 'doc2', 
            'content': "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain, consisting of interconnected nodes that process information.",
            'metadata': {'source': 'research_paper', 'year': 2023}
        }
    ]
    
    # Process documents
    processed = pipeline.process_documents(documents)
    
    print(f"Processed {len(documents)} documents into {len(processed)} chunks")
     for doc in processed:
         print(f"Chunk {doc['id']}: {len(doc['content'])} chars, embedding dim: {len(doc['embedding'])}")
```

### 4.2 Multi-Modal Embeddings

```python
# multimodal_embeddings.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import List, Union, Dict, Any

class MultiModalEmbedding:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text"""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def embed_images(self, images: List[Union[Image.Image, str]]) -> np.ndarray:
        """Generate embeddings for images"""
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                # Load from URL or path
                if img.startswith('http'):
                    response = requests.get(img)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(img)
            processed_images.append(img)
        
        inputs = self.processor(images=processed_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def calculate_similarity(self, text_embeddings: np.ndarray, 
                           image_embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity between text and image embeddings"""
        return np.dot(text_embeddings, image_embeddings.T)
    
    def search_images_by_text(self, query_text: str, 
                             image_embeddings: np.ndarray,
                             image_metadata: List[Dict],
                             top_k: int = 5) -> List[Dict]:
        """Search images using text query"""
        text_embedding = self.embed_text([query_text])
        similarities = self.calculate_similarity(text_embedding, image_embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = {
                'similarity': float(similarities[idx]),
                'metadata': image_metadata[idx],
                'index': int(idx)
            }
            results.append(result)
        
        return results

# Usage example
if __name__ == "__main__":
    # Initialize multi-modal embedding
    mm_embed = MultiModalEmbedding()
    
    # Sample texts and images
    texts = [
        "a cat sitting on a chair",
        "a dog running in the park", 
        "a beautiful sunset over mountains",
        "a person coding on a laptop"
    ]
    
    # For demo, we'll use placeholder image metadata
    image_metadata = [
        {'filename': 'cat.jpg', 'description': 'Orange cat on wooden chair'},
        {'filename': 'dog.jpg', 'description': 'Golden retriever in grass'},
        {'filename': 'sunset.jpg', 'description': 'Mountain landscape at dusk'},
        {'filename': 'coding.jpg', 'description': 'Developer working on computer'}
    ]
    
    # Generate text embeddings
    text_embeddings = mm_embed.embed_text(texts)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Simulate image embeddings (in practice, you'd load actual images)
    # For demo, we'll use the text embeddings as proxy image embeddings
    image_embeddings = text_embeddings + np.random.normal(0, 0.1, text_embeddings.shape)
    
    # Search images by text
    query = "feline animal furniture"
    results = mm_embed.search_images_by_text(query, image_embeddings, image_metadata)
    
    print(f"\nSearch results for '{query}':")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['metadata']['filename']} (similarity: {result['similarity']:.3f})")
        print(f"   Description: {result['metadata']['description']}")
```

---

## 5. Production Vector Database Systems

### 5.1 Scalability Considerations

```python
# production_vector_db.py
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class VectorDBConfig:
    """Configuration for production vector database"""
    max_connections: int = 100
    connection_timeout: float = 30.0
    retry_attempts: int = 3
    batch_size: int = 100
    index_type: str = "hnsw"
    similarity_metric: str = "cosine"
    replication_factor: int = 2
    sharding_strategy: str = "hash"

class ProductionVectorDB:
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection_pool = None
        self.executor = ThreadPoolExecutor(max_workers=config.max_connections)
        
        # Monitoring metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def initialize(self):
        """Initialize database connections and resources"""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections // 4
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
        self.connection_pool = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        self.logger.info("Vector database initialized")
    
    async def close(self):
        """Clean up resources"""
        if self.connection_pool:
            await self.connection_pool.close()
        self.executor.shutdown(wait=True)
    
    async def batch_insert(self, vectors: List[np.ndarray], 
                          metadata: List[Dict[str, Any]],
                          collection_name: str) -> Dict[str, Any]:
        """Insert vectors in batches for better performance"""
        start_time = time.time()
        
        try:
            # Split into batches
            batches = []
            for i in range(0, len(vectors), self.config.batch_size):
                batch_vectors = vectors[i:i + self.config.batch_size]
                batch_metadata = metadata[i:i + self.config.batch_size]
                batches.append((batch_vectors, batch_metadata))
            
            # Process batches concurrently
            tasks = []
            for batch_vectors, batch_metadata in batches:
                task = self._insert_batch(batch_vectors, batch_metadata, collection_name)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            total_inserted = 0
            errors = []
            
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                else:
                    total_inserted += result.get('inserted_count', 0)
            
            end_time = time.time()
            
            # Update metrics
            self.metrics['total_requests'] += 1
            if not errors:
                self.metrics['successful_requests'] += 1
            else:
                self.metrics['failed_requests'] += 1
            
            response_time = end_time - start_time
            self._update_avg_response_time(response_time)
            
            return {
                'inserted_count': total_inserted,
                'batch_count': len(batches),
                'errors': errors,
                'response_time': response_time
            }
        
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            self.metrics['failed_requests'] += 1
            raise
    
    async def _insert_batch(self, vectors: List[np.ndarray], 
                           metadata: List[Dict[str, Any]],
                           collection_name: str) -> Dict[str, Any]:
        """Insert a single batch of vectors"""
        # Simulate database insertion
        await asyncio.sleep(0.1)  # Simulate network/processing delay
        
        return {
            'inserted_count': len(vectors),
            'collection': collection_name
        }
    
    async def similarity_search(self, query_vector: np.ndarray,
                               collection_name: str,
                               k: int = 10,
                               filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform similarity search with caching and optimization"""
        start_time = time.time()
        
        try:
            # Check cache first (simplified cache key)
            cache_key = f"{collection_name}_{hash(query_vector.tobytes())}_{k}"
            cached_result = await self._get_from_cache(cache_key)
            
            if cached_result:
                self.metrics['cache_hits'] += 1
                return cached_result
            
            self.metrics['cache_misses'] += 1
            
            # Perform actual search
            results = await self._perform_search(query_vector, collection_name, k, filters)
            
            # Cache results
            await self._cache_results(cache_key, results)
            
            end_time = time.time()
            self._update_avg_response_time(end_time - start_time)
            
            self.metrics['successful_requests'] += 1
            return results
        
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            self.metrics['failed_requests'] += 1
            raise
    
    async def _perform_search(self, query_vector: np.ndarray,
                             collection_name: str,
                             k: int,
                             filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Perform the actual similarity search"""
        # Simulate search operation
        await asyncio.sleep(0.05)
        
        # Return mock results
        results = []
        for i in range(k):
            results.append({
                'id': f"doc_{i}",
                'score': 0.9 - (i * 0.05),
                'metadata': {'index': i, 'collection': collection_name}
            })
        
        return results
    
    async def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get results from cache"""
        # Simplified cache implementation
        return None
    
    async def _cache_results(self, cache_key: str, results: List[Dict[str, Any]]):
        """Cache search results"""
        # Simplified cache implementation
        pass
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time metric"""
        total_requests = self.metrics['total_requests']
        current_avg = self.metrics['avg_response_time']
        
        # Calculate new average
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.metrics['avg_response_time'] = new_avg
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get database health and performance metrics"""
        total_requests = self.metrics['total_requests']
        success_rate = (self.metrics['successful_requests'] / total_requests * 100) if total_requests > 0 else 0
        cache_hit_rate = (self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) * 100) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        
        return {
            'status': 'healthy' if success_rate > 95 else 'degraded',
            'metrics': {
                **self.metrics,
                'success_rate_percent': success_rate,
                'cache_hit_rate_percent': cache_hit_rate
            },
            'config': {
                'max_connections': self.config.max_connections,
                'batch_size': self.config.batch_size,
                'index_type': self.config.index_type
            }
        }

# Usage example
async def main():
    config = VectorDBConfig(
        max_connections=50,
        batch_size=100,
        index_type="hnsw"
    )
    
    db = ProductionVectorDB(config)
    await db.initialize()
    
    try:
        # Generate sample data
        vectors = [np.random.random(128).astype(np.float32) for _ in range(1000)]
        metadata = [{'id': i, 'category': f'cat_{i % 10}'} for i in range(1000)]
        
        # Batch insert
        print("Performing batch insert...")
        insert_result = await db.batch_insert(vectors, metadata, "test_collection")
        print(f"Inserted {insert_result['inserted_count']} vectors in {insert_result['response_time']:.2f}s")
        
        # Perform searches
        print("\nPerforming similarity searches...")
        for i in range(10):
            query = np.random.random(128).astype(np.float32)
            results = await db.similarity_search(query, "test_collection", k=5)
            print(f"Search {i+1}: Found {len(results)} results")
        
        # Check health status
        health = db.get_health_status()
        print(f"\nDatabase Health: {health['status']}")
        print(f"Success Rate: {health['metrics']['success_rate_percent']:.1f}%")
        print(f"Avg Response Time: {health['metrics']['avg_response_time']:.3f}s")
        print(f"Cache Hit Rate: {health['metrics']['cache_hit_rate_percent']:.1f}%")
    
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Hands-on Exercises

### Exercise 1: Building a Multi-Modal Search Engine

**Objective**: Create a search engine that can find images using text queries and vice versa.

```python
# exercise_1_multimodal_search.py
import numpy as np
import chromadb
from chromadb.config import Settings
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from typing import List, Dict, Any
import os
import json

class MultiModalSearchEngine:
    def __init__(self, collection_name: str = "multimodal_search"):
        # Initialize CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text using CLIP"""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode images using CLIP"""
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue
        
        if not images:
            return np.array([])
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def add_images(self, image_paths: List[str], descriptions: List[str]):
        """Add images to the search index"""
        # Encode images
        image_embeddings = self.encode_images(image_paths)
        
        if len(image_embeddings) == 0:
            print("No valid images to add")
            return
        
        # Prepare data for ChromaDB
        ids = [f"img_{i}" for i in range(len(image_paths))]
        metadatas = [
            {
                "type": "image",
                "path": path,
                "description": desc
            }
            for path, desc in zip(image_paths, descriptions)
        ]
        
        # Add to collection
        self.collection.add(
            embeddings=image_embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(image_paths)} images to search index")
    
    def add_texts(self, texts: List[str], sources: List[str]):
        """Add text documents to the search index"""
        # Encode texts
        text_embeddings = self.encode_text(texts)
        
        # Prepare data for ChromaDB
        ids = [f"txt_{i}" for i in range(len(texts))]
        metadatas = [
            {
                "type": "text",
                "content": text,
                "source": source
            }
            for text, source in zip(texts, sources)
        ]
        
        # Add to collection
        self.collection.add(
            embeddings=text_embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(texts)} text documents to search index")
    
    def search_by_text(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content using text query"""
        # Encode query
        query_embedding = self.encode_text([query])[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            formatted_results.append({
                'rank': i + 1,
                'similarity': 1 - distance,  # Convert distance to similarity
                'type': metadata['type'],
                'metadata': metadata
            })
        
        return formatted_results
    
    def search_by_image(self, image_path: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content using image query"""
        # Encode query image
        query_embedding = self.encode_images([image_path])[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            formatted_results.append({
                'rank': i + 1,
                'similarity': 1 - distance,
                'type': metadata['type'],
                'metadata': metadata
            })
        
        return formatted_results

# Usage example
if __name__ == "__main__":
    # Initialize search engine
    search_engine = MultiModalSearchEngine()
    
    # Sample data (you would replace with actual image paths and texts)
    sample_image_paths = [
        "sample_images/cat.jpg",
        "sample_images/dog.jpg",
        "sample_images/sunset.jpg",
        "sample_images/coding.jpg"
    ]
    
    sample_descriptions = [
        "A cute orange cat sitting on a wooden chair",
        "A golden retriever running in a green park",
        "Beautiful sunset over mountain landscape",
        "Person coding on laptop in modern office"
    ]
    
    sample_texts = [
        "Machine learning is transforming how we process and understand data",
        "Vector databases enable efficient similarity search at scale",
        "Natural language processing helps computers understand human language",
        "Computer vision allows machines to interpret visual information"
    ]
    
    sample_sources = [
        "ML Article 1",
        "Vector DB Guide",
        "NLP Tutorial",
        "CV Overview"
    ]
    
    # Add sample data (commented out since we don't have actual images)
    # search_engine.add_images(sample_image_paths, sample_descriptions)
    search_engine.add_texts(sample_texts, sample_sources)
    
    # Perform searches
    print("\n=== Text Search Results ===")
    text_results = search_engine.search_by_text("artificial intelligence and data processing")
    for result in text_results:
        print(f"Rank {result['rank']}: {result['metadata']['content'][:100]}...")
        print(f"  Similarity: {result['similarity']:.3f}")
        print(f"  Source: {result['metadata']['source']}\n")
    
    # Image search example (commented out since we don't have actual images)
    # print("\n=== Image Search Results ===")
    # image_results = search_engine.search_by_image("query_image.jpg")
    # for result in image_results:
    #     print(f"Rank {result['rank']}: {result['metadata']['description']}")
    #     print(f"  Similarity: {result['similarity']:.3f}")
    #     print(f"  Path: {result['metadata']['path']}\n")
```

### Exercise 2: Vector Database Performance Comparison

**Objective**: Compare the performance of different vector databases and indexing strategies.

```python
# exercise_2_performance_comparison.py
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
import faiss
from dataclasses import dataclass
import psutil
import os

@dataclass
class PerformanceMetrics:
    insert_time: float
    search_time: float
    memory_usage: float
    accuracy: float
    index_size: float

class VectorDBBenchmark:
    def __init__(self, dimension: int = 128, num_vectors: int = 10000):
        self.dimension = dimension
        self.num_vectors = num_vectors
        self.vectors = self._generate_test_data()
        self.query_vectors = self._generate_test_data(100)  # 100 query vectors
        
    def _generate_test_data(self, count: int = None) -> np.ndarray:
        """Generate random test vectors"""
        if count is None:
            count = self.num_vectors
        
        # Generate random vectors with some structure
        vectors = np.random.random((count, self.dimension)).astype(np.float32)
        # Normalize vectors
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_chromadb(self) -> PerformanceMetrics:
        """Benchmark ChromaDB performance"""
        print("Benchmarking ChromaDB...")
        
        # Initialize ChromaDB
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            name="benchmark",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Measure insertion time
        memory_before = self._get_memory_usage()
        start_time = time.time()
        
        # Insert vectors in batches
        batch_size = 1000
        for i in range(0, len(self.vectors), batch_size):
            batch_vectors = self.vectors[i:i + batch_size]
            batch_ids = [f"vec_{j}" for j in range(i, min(i + batch_size, len(self.vectors)))]
            
            collection.add(
                embeddings=batch_vectors.tolist(),
                ids=batch_ids
            )
        
        insert_time = time.time() - start_time
        memory_after = self._get_memory_usage()
        
        # Measure search time
        start_time = time.time()
        
        for query_vector in self.query_vectors[:10]:  # Test with 10 queries
            results = collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=10
            )
        
        search_time = (time.time() - start_time) / 10  # Average per query
        
        # Calculate accuracy (simplified - using distance threshold)
        accuracy = 0.95  # Placeholder - would need ground truth for real accuracy
        
        return PerformanceMetrics(
            insert_time=insert_time,
            search_time=search_time,
            memory_usage=memory_after - memory_before,
            accuracy=accuracy,
            index_size=memory_after - memory_before
        )
    
    def benchmark_faiss_flat(self) -> PerformanceMetrics:
        """Benchmark FAISS Flat index performance"""
        print("Benchmarking FAISS Flat...")
        
        # Initialize FAISS Flat index
        index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        
        # Measure insertion time
        memory_before = self._get_memory_usage()
        start_time = time.time()
        
        index.add(self.vectors)
        
        insert_time = time.time() - start_time
        memory_after = self._get_memory_usage()
        
        # Measure search time
        start_time = time.time()
        
        for query_vector in self.query_vectors[:10]:
            distances, indices = index.search(query_vector.reshape(1, -1), 10)
        
        search_time = (time.time() - start_time) / 10
        
        return PerformanceMetrics(
            insert_time=insert_time,
            search_time=search_time,
            memory_usage=memory_after - memory_before,
            accuracy=1.0,  # Flat index gives exact results
            index_size=memory_after - memory_before
        )
    
    def benchmark_faiss_hnsw(self) -> PerformanceMetrics:
        """Benchmark FAISS HNSW index performance"""
        print("Benchmarking FAISS HNSW...")
        
        # Initialize FAISS HNSW index
        index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections per node
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        
        # Measure insertion time
        memory_before = self._get_memory_usage()
        start_time = time.time()
        
        index.add(self.vectors)
        
        insert_time = time.time() - start_time
        memory_after = self._get_memory_usage()
        
        # Measure search time
        start_time = time.time()
        
        for query_vector in self.query_vectors[:10]:
            distances, indices = index.search(query_vector.reshape(1, -1), 10)
        
        search_time = (time.time() - start_time) / 10
        
        return PerformanceMetrics(
            insert_time=insert_time,
            search_time=search_time,
            memory_usage=memory_after - memory_before,
            accuracy=0.98,  # HNSW gives approximate results
            index_size=memory_after - memory_before
        )
    
    def run_benchmark(self) -> Dict[str, PerformanceMetrics]:
        """Run all benchmarks and return results"""
        results = {}
        
        try:
            results['ChromaDB'] = self.benchmark_chromadb()
        except Exception as e:
            print(f"ChromaDB benchmark failed: {e}")
        
        try:
            results['FAISS_Flat'] = self.benchmark_faiss_flat()
        except Exception as e:
            print(f"FAISS Flat benchmark failed: {e}")
        
        try:
            results['FAISS_HNSW'] = self.benchmark_faiss_hnsw()
        except Exception as e:
            print(f"FAISS HNSW benchmark failed: {e}")
        
        return results
    
    def plot_results(self, results: Dict[str, PerformanceMetrics]):
        """Plot benchmark results"""
        if not results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Vector Database Performance Comparison')
        
        databases = list(results.keys())
        
        # Insert time
        insert_times = [results[db].insert_time for db in databases]
        axes[0, 0].bar(databases, insert_times)
        axes[0, 0].set_title('Insert Time (seconds)')
        axes[0, 0].set_ylabel('Time (s)')
        
        # Search time
        search_times = [results[db].search_time * 1000 for db in databases]  # Convert to ms
        axes[0, 1].bar(databases, search_times)
        axes[0, 1].set_title('Average Search Time (milliseconds)')
        axes[0, 1].set_ylabel('Time (ms)')
        
        # Memory usage
        memory_usage = [results[db].memory_usage for db in databases]
        axes[1, 0].bar(databases, memory_usage)
        axes[1, 0].set_title('Memory Usage (MB)')
        axes[1, 0].set_ylabel('Memory (MB)')
        
        # Accuracy
        accuracy = [results[db].accuracy * 100 for db in databases]
        axes[1, 1].bar(databases, accuracy)
        axes[1, 1].set_title('Search Accuracy (%)')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_ylim(90, 100)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n=== Detailed Results ===")
        for db_name, metrics in results.items():
            print(f"\n{db_name}:")
            print(f"  Insert Time: {metrics.insert_time:.2f}s")
            print(f"  Search Time: {metrics.search_time*1000:.2f}ms")
            print(f"  Memory Usage: {metrics.memory_usage:.2f}MB")
            print(f"  Accuracy: {metrics.accuracy*100:.1f}%")
            print(f"  Index Size: {metrics.index_size:.2f}MB")

# Usage example
if __name__ == "__main__":
    # Run benchmark with different dataset sizes
    dataset_sizes = [1000, 5000, 10000]
    
    for size in dataset_sizes:
        print(f"\n{'='*50}")
        print(f"Benchmarking with {size} vectors")
        print(f"{'='*50}")
        
        benchmark = VectorDBBenchmark(dimension=128, num_vectors=size)
        results = benchmark.run_benchmark()
        
        if results:
            benchmark.plot_results(results)
        
        # Add delay between benchmarks
        time.sleep(2)
```

### Exercise 3: Building a Hybrid Search System

**Objective**: Combine keyword search with vector similarity search for better results.

```python
# exercise_3_hybrid_search.py
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
from collections import Counter
import math
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

@dataclass
class SearchResult:
    id: str
    content: str
    vector_score: float
    keyword_score: float
    hybrid_score: float
    metadata: Dict[str, Any]

class HybridSearchEngine:
    def __init__(self, collection_name: str = "hybrid_search"):
        # Initialize sentence transformer for embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Document storage for keyword search
        self.documents = {}
        self.inverted_index = {}
        self.document_frequencies = {}
        self.total_documents = 0
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword search"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Split into words and remove empty strings
        words = [word for word in text.split() if word]
        return words
    
    def _build_inverted_index(self, doc_id: str, content: str):
        """Build inverted index for keyword search"""
        words = self._preprocess_text(content)
        word_counts = Counter(words)
        
        # Update inverted index
        for word, count in word_counts.items():
            if word not in self.inverted_index:
                self.inverted_index[word] = {}
            self.inverted_index[word][doc_id] = count
        
        # Update document frequencies
        unique_words = set(words)
        for word in unique_words:
            self.document_frequencies[word] = self.document_frequencies.get(word, 0) + 1
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both vector and keyword search indices"""
        contents = [doc['content'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.encoder.encode(contents)
        
        # Add to vector database
        self.collection.add(
            embeddings=embeddings.tolist(),
            metadatas=[{k: v for k, v in doc.items() if k != 'content'} for doc in documents],
            documents=contents,
            ids=ids
        )
        
        # Add to keyword search index
        for doc in documents:
            doc_id = doc['id']
            content = doc['content']
            
            self.documents[doc_id] = doc
            self._build_inverted_index(doc_id, content)
            self.total_documents += 1
        
        print(f"Added {len(documents)} documents to hybrid search index")
    
    def _calculate_tf_idf(self, query_words: List[str], doc_id: str) -> float:
        """Calculate TF-IDF score for a document given query words"""
        if doc_id not in self.documents:
            return 0.0
        
        doc_content = self.documents[doc_id]['content']
        doc_words = self._preprocess_text(doc_content)
        doc_word_counts = Counter(doc_words)
        doc_length = len(doc_words)
        
        score = 0.0
        
        for word in query_words:
            if word in doc_word_counts:
                # Term frequency
                tf = doc_word_counts[word] / doc_length
                
                # Inverse document frequency
                df = self.document_frequencies.get(word, 0)
                if df > 0:
                    idf = math.log(self.total_documents / df)
                else:
                    idf = 0
                
                score += tf * idf
        
        return score
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform keyword-based search using TF-IDF"""
        query_words = self._preprocess_text(query)
        
        if not query_words:
            return []
        
        # Find candidate documents
        candidate_docs = set()
        for word in query_words:
            if word in self.inverted_index:
                candidate_docs.update(self.inverted_index[word].keys())
        
        # Calculate TF-IDF scores
        doc_scores = []
        for doc_id in candidate_docs:
            score = self._calculate_tf_idf(query_words, doc_id)
            if score > 0:
                doc_scores.append((doc_id, score))
        
        # Sort by score and return top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]
    
    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform vector similarity search"""
        # Generate query embedding
        query_embedding = self.encoder.encode([query])
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Extract results
        doc_scores = []
        for doc_id, distance in zip(results['ids'][0], results['distances'][0]):
            similarity = 1 - distance  # Convert distance to similarity
            doc_scores.append((doc_id, similarity))
        
        return doc_scores
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search"""
        # Perform both searches
        vector_results = self._vector_search(query, top_k * 2)  # Get more candidates
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Normalize scores
        def normalize_scores(scores: List[Tuple[str, float]]) -> Dict[str, float]:
            if not scores:
                return {}
            
            max_score = max(score for _, score in scores)
            min_score = min(score for _, score in scores)
            
            if max_score == min_score:
                return {doc_id: 1.0 for doc_id, _ in scores}
            
            normalized = {}
            for doc_id, score in scores:
                normalized[doc_id] = (score - min_score) / (max_score - min_score)
            
            return normalized
        
        vector_scores = normalize_scores(vector_results)
        keyword_scores = normalize_scores(keyword_results)
        
        # Combine scores
        all_doc_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        hybrid_results = []
        
        for doc_id in all_doc_ids:
            vector_score = vector_scores.get(doc_id, 0.0)
            keyword_score = keyword_scores.get(doc_id, 0.0)
            
            # Calculate hybrid score
            hybrid_score = (vector_weight * vector_score) + (keyword_weight * keyword_score)
            
            if doc_id in self.documents:
                result = SearchResult(
                    id=doc_id,
                    content=self.documents[doc_id]['content'],
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    hybrid_score=hybrid_score,
                    metadata=self.documents[doc_id]
                )
                hybrid_results.append(result)
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return hybrid_results[:top_k]
    
    def explain_search(self, query: str, doc_id: str) -> Dict[str, Any]:
        """Explain why a document was ranked for a query"""
        if doc_id not in self.documents:
            return {"error": "Document not found"}
        
        # Get individual scores
        vector_results = self._vector_search(query, 100)
        keyword_results = self._keyword_search(query, 100)
        
        vector_score = next((score for did, score in vector_results if did == doc_id), 0.0)
        keyword_score = next((score for did, score in keyword_results if did == doc_id), 0.0)
        
        # Analyze query terms
        query_words = self._preprocess_text(query)
        doc_content = self.documents[doc_id]['content']
        doc_words = self._preprocess_text(doc_content)
        
        term_analysis = {}
        for word in query_words:
            term_analysis[word] = {
                'in_document': word in doc_words,
                'document_frequency': self.document_frequencies.get(word, 0),
                'term_frequency': doc_words.count(word) if word in doc_words else 0
            }
        
        return {
            'document_id': doc_id,
            'query': query,
            'vector_score': vector_score,
            'keyword_score': keyword_score,
            'term_analysis': term_analysis,
            'document_preview': doc_content[:200] + '...' if len(doc_content) > 200 else doc_content
        }

# Usage example
if __name__ == "__main__":
    # Initialize hybrid search engine
    search_engine = HybridSearchEngine()
    
    # Sample documents
    documents = [
        {
            'id': 'doc1',
            'content': 'Machine learning algorithms can automatically learn patterns from data without explicit programming.',
            'category': 'AI',
            'author': 'John Doe'
        },
        {
            'id': 'doc2', 
            'content': 'Vector databases enable efficient similarity search and retrieval of high-dimensional data.',
            'category': 'Database',
            'author': 'Jane Smith'
        },
        {
            'id': 'doc3',
            'content': 'Natural language processing helps computers understand and generate human language.',
            'category': 'NLP',
            'author': 'Bob Johnson'
        },
        {
            'id': 'doc4',
            'content': 'Deep learning neural networks have revolutionized computer vision and image recognition.',
            'category': 'Deep Learning',
            'author': 'Alice Brown'
        },
        {
            'id': 'doc5',
            'content': 'Search algorithms and information retrieval systems help users find relevant content quickly.',
            'category': 'Search',
            'author': 'Charlie Wilson'
        }
    ]
    
    # Add documents to search engine
    search_engine.add_documents(documents)
    
    # Perform hybrid search
    query = "machine learning algorithms data"
    print(f"\nSearching for: '{query}'")
    print("=" * 50)
    
    results = search_engine.hybrid_search(query, top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"\nRank {i}: {result.id}")
        print(f"Content: {result.content[:100]}...")
        print(f"Vector Score: {result.vector_score:.3f}")
        print(f"Keyword Score: {result.keyword_score:.3f}")
        print(f"Hybrid Score: {result.hybrid_score:.3f}")
        print(f"Category: {result.metadata.get('category', 'N/A')}")
    
    # Explain search for top result
    if results:
        print("\n" + "=" * 50)
        print("SEARCH EXPLANATION")
        print("=" * 50)
        
        explanation = search_engine.explain_search(query, results[0].id)
        print(f"\nExplanation for document: {explanation['document_id']}")
        print(f"Vector Score: {explanation['vector_score']:.3f}")
        print(f"Keyword Score: {explanation['keyword_score']:.3f}")
        print("\nTerm Analysis:")
        for term, analysis in explanation['term_analysis'].items():
            print(f"  '{term}': In doc: {analysis['in_document']}, "
                  f"TF: {analysis['term_frequency']}, DF: {analysis['document_frequency']}")
```

---

## Module Summary

In this module, you've learned the fundamentals of vector databases and their applications in modern AI systems. Here are the key takeaways:

### Key Concepts Covered:
1. **Vector Embeddings**: Understanding how to convert text, images, and other data into numerical representations
2. **Similarity Metrics**: Cosine similarity, Euclidean distance, and dot product for measuring vector similarity
3. **Vector Database Architecture**: Storage, indexing, and retrieval mechanisms
4. **Indexing Strategies**: HNSW, IVF, and other approximate nearest neighbor algorithms
5. **Multi-Modal Embeddings**: Combining text and image embeddings for cross-modal search
6. **Production Considerations**: Scalability, performance optimization, and monitoring

### Practical Skills Developed:
- Implementing vector databases using ChromaDB and FAISS
- Building embedding pipelines for text and images
- Creating multi-modal search engines
- Performance benchmarking and optimization
- Hybrid search combining vector and keyword approaches

### Next Steps:
In the next module, we'll explore **Multi-Agent Applications**, where you'll learn how to:
- Design and implement multi-agent systems
- Coordinate communication between agents
- Build collaborative AI workflows
- Handle agent orchestration and task distribution

The vector database knowledge from this module will be essential for building agents that can efficiently store, retrieve, and share knowledge across the system.