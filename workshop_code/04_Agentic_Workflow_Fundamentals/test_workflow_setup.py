import sys, os
from datetime import datetime

def test_installation():
    print("�� Testing Workflow Environment Setup...
")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Test time: {datetime.now()}
")
    tests = [
        ("asyncio", "AsyncIO"),
        ("pydantic", "Pydantic"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("aiohttp", "AioHTTP"),
        ("pytest", "Pytest"),
        ("asyncpg", "AsyncPG"),
        ("motor", "Motor"),
        ("aio_pika", "AioPika"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("rich", "Rich")
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
    success_rate = sum(results) / len(results) * 100
    print(f"
📊 Installation Success Rate: {success_rate:.1f}%")
    if success_rate >= 80:
        print("🎉 Setup verification complete! You're ready to proceed.")
        return True
    else:
        print("⚠️ Some components failed. Please check the installation.")
        return False

if __name__ == "__main__":
    test_installation()

