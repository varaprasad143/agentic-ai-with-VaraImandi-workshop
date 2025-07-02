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

if __name__ == "__main__":
    metrics = SimilarityMetrics()
    metrics.compare_all_metrics()
    metrics.demonstrate_metric_properties() 