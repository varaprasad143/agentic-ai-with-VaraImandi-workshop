# Module I (Enhanced): Building Context-Aware LLM Applications and MCP

![Context-Aware LLM Pipeline](01_Context_Aware_LLM_Applications/module_flowchart_v1.png)
*Figure: The flow of user input through preprocessing, context injection, LLM inference, postprocessing, and feedback loop.*

## Setup

### 1. Install Python 3.15
> Download and install the latest Python from [python.org](https://www.python.org/downloads/).

### 2. Create and Activate a Virtual Environment
```bash
python3.15 -m venv context_aware_llm_env
# Creates a new isolated Python environment

source context_aware_llm_env/bin/activate
# Activates the environment (use the appropriate command for your OS)
```

### 3. Install Required Packages
```bash
pip install --upgrade pip
# Upgrades pip to the latest version

pip install langchain langchain-openai langchain-community langchain-experimental chromadb sentence-transformers faiss-cpu pypdf python-docx beautifulsoup4 unstructured tiktoken requests python-dotenv pydantic streamlit plotly matplotlib pandas numpy redis openai anthropic
# Installs the LangChain ecosystem for LLM applications

pip install chromadb sentence-transformers faiss-cpu
# Installs vector database and embedding libraries

pip install pypdf python-docx beautifulsoup4 unstructured
# Installs document processing libraries

pip install tiktoken requests python-dotenv pydantic
# Installs utility and API libraries

pip install streamlit plotly matplotlib
# Installs visualization and UI libraries

pip install pandas numpy
# Installs data processing libraries

pip install redis openai anthropic
# Installs optional advanced features
```

### 4. Configure Environment Variables
Create a `.env` file in your project root with your API keys and settings.

### 5. Verify Installation
Save the following as `test_setup.py` and run it:
```python
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
```
> This script checks that all required packages and environment variables are set up correctly.

---

## 1. Understanding Model I/O: The Foundation of LLM Applications

Imagine the Model I/O pipeline as the nervous system of your AI application. Just as your brain processes sensory input, reasons about it, and produces actions, the Model I/O pipeline transforms raw user input into intelligent, structured responses.

![Model I/O Pipeline](images/model_io_pipeline.png)
*Figure: The Model I/O pipeline transforms raw user input into intelligent, structured responses.*

--- 