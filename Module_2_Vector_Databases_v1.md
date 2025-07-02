# Module II (Enhanced): Vector Databases for Agentic AI

![Vector Database Pipeline](02_Vector_Databases/module_flowchart.png)
*Figure: Data flows from raw input through vectorization, indexing, similarity search, and result retrieval.*

## Setup

### 1. Install Python 3.13.5
> Download and install the latest Python from [python.org](https://www.python.org/downloads/).

### 2. Create and Activate a Virtual Environment
```bash
python3.13 -m venv vector_db_env
# Creates a new isolated Python environment

source vector_db_env/bin/activate
# Activates the environment (use the appropriate command for your OS)
```

### 3. Install Required Packages
```bash
pip install --upgrade pip
# Upgrades pip to the latest version

pip install chromadb sentence-transformers faiss-cpu numpy pandas scikit-learn transformers torch openai plotly streamlit
# Installs core vector database, embedding, and utility libraries

# Optional: For advanced features, also install
pip install pinecone-client weaviate-client qdrant-client cohere
# Installs additional vector DB connectors and embedding providers (optional)
```

### 4. Configure Environment Variables
Create a `.env` file in your project root with your API keys and settings. At minimum, set:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Verify Installation
Save the following as `test_vector_setup.py` and run it:
```python
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
> This script checks that all required packages and environment variables are set up correctly. If you see warnings for Pinecone, Weaviate, or Qdrant, those are optional unless you plan to use those databases.

---

## Validation Results

- All core dependencies are installed and working.
- Vector operations, FAISS, and Sentence Transformers are functional.
- Optional connectors (Pinecone, Weaviate, Qdrant) are not installed by default—install them if you need those features.
- The environment is ready for hands-on exercises!

---

## Comprehensive Vector Database System: Summary

In this section, you built and validated a production-ready vector database system supporting three major index types:

- **Flat Index:** Brute-force, highly accurate, and easy to understand. All add, search, filter, retrieve, delete, and stats operations passed.
- **LSH Index:** Fast, approximate search for large-scale data. All operations passed.
- **IVF Index:** Efficient clustering-based search. After reducing the number of clusters to 5, all operations passed and the system is robust even with small datasets.

**Key Takeaways:**
- All core features (adding, searching, filtering, retrieving, deleting, and statistics) are validated and reproducible.
- The code is ready for real-world use and experimentation by readers.
- The environment supports advanced vector search and can be extended with more data or custom index parameters.

---

## Section Summaries

### 1. Vector Similarity Metrics (Foundational)
- Explored cosine similarity, Euclidean and Manhattan distances, and dot product.
- Demonstrated how different metrics behave with vector scaling and normalization.
- All code ran successfully, providing clear, reproducible outputs for learners.

### 2. Practical Vector Database with Chroma
- Built a local vector database using Chroma with a SentenceTransformer embedding function.
- Successfully added, queried, and filtered documents by metadata.
- Validated that the latest Chroma version requires an explicit embedding function.
- All operations worked as intended, ensuring reproducibility for readers.

### 3. Advanced Chroma Features
- Demonstrated batch operations by adding 1,000 documents in manageable chunks, showing how to scale vector databases efficiently.
- Showcased complex filtering with multiple metadata conditions, confirming flexible and powerful search capabilities.
- Performed similarity search with a score threshold, illustrating how to refine results for higher relevance (no results above 0.8 for random data, as expected).
- All advanced features worked as intended, ensuring reproducibility and practical value for real-world applications.

### 4. HNSW Index Implementation
- Built a custom Hierarchical Navigable Small World (HNSW) index for fast, scalable vector search.
- Added 20 random vectors and performed a nearest neighbor search, confirming the algorithm's logic and output.
- Demonstrated how advanced indexing strategies can be implemented from scratch for research or custom production needs.
- All code ran successfully, ensuring full reproducibility for readers.

---

**You have now completed all core hands-on vector database exercises, from foundational metrics to advanced indexing!**

# Next: Advanced Hands-On Exercise

Stay tuned for the next hands-on exercise, where you'll explore even more advanced vector database and embedding workflows, including multi-modal search, hybrid indexes, and real-time updates! 