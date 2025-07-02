# test_setup.py
import sys, os
from datetime import datetime

def test_installation():
    print("🔍 Testing Installation...\n")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Test time: {datetime.now()}\n")
    tests = [
        ("langchain", "LangChain"),
        ("langchain_openai", "LangChain OpenAI"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("tiktoken", "TikToken"),
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
        ("dotenv", "Python Dotenv")
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
    print("\n🔑 Environment Variables:")
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_API_TOKEN"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set (length: {len(value)})")
        else:
            print(f"⚠️ {var}: Not set")
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Installation Success Rate: {success_rate:.1f}%")
    if success_rate >= 80:
        print("🎉 Setup verification complete! You're ready to proceed.")
        return True
    else:
        print("⚠️ Some components failed. Please check the installation.")
        return False

if __name__ == "__main__":
    test_installation() 