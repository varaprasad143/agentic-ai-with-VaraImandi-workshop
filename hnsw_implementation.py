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
        
        if self.entry_point is None or level > self.max_level:
            self.entry_point = node_id
            self.max_level = level
        
        # Connect to neighbors (simplified for demo)
        for lev in range(level + 1):
            if len(self.graph[lev]) > 1:
                # Connect to a random existing node at this level
                candidates = [nid for nid in self.graph[lev] if nid != node_id]
                if candidates:
                    neighbor = random.choice(candidates)
                    self.graph[lev][node_id].append(neighbor)
                    self.graph[lev][neighbor].append(node_id)
    
    def search(self, query: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors"""
        if self.entry_point is None:
            return []
        
        # Start from the top layer
        ep = self.entry_point
        level = self.max_level
        candidates = [ep]
        
        while level > 0:
            results = self._search_layer(query, candidates, 1, level)
            candidates = [nid for _, nid in results]
            level -= 1
        
        # Final search at layer 0
        results = self._search_layer(query, candidates, k, 0)
        # Return sorted by distance
        return sorted([(nid, dist) for dist, nid in results], key=lambda x: x[1])

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    dim = 8
    hnsw = HNSWIndex(dimension=dim)
    # Add 20 random vectors
    for i in range(20):
        vec = np.random.rand(dim)
        hnsw.add(vec, i)
    # Query with a new random vector
    query_vec = np.random.rand(dim)
    results = hnsw.search(query_vec, k=5)
    print("Top 5 nearest neighbors:")
    for nid, dist in results:
        print(f"Node {nid} at distance {dist:.4f}") 