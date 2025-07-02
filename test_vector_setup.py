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