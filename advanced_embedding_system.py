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