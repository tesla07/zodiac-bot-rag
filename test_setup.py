#!/usr/bin/env python3
"""
Test script to verify the Zodiac RAG chatbot setup
"""

import os
import sys
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import fitz
        print("✅ PyMuPDF imported successfully")
    except ImportError as e:
        print(f"❌ PyMuPDF import failed: {e}")
        return False
    
    try:
        from unstructured.partition.auto import partition
        print("✅ unstructured imported successfully")
    except ImportError as e:
        print(f"❌ unstructured import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    return True

def test_local_modules():
    """Test if local modules can be imported"""
    print("\n🔍 Testing local modules...")
    
    try:
        from parser_chunker import ParserChunker
        print("✅ parser_chunker imported successfully")
    except ImportError as e:
        print(f"❌ parser_chunker import failed: {e}")
        return False
    
    try:
        from embed_index import EmbedIndexManager
        print("✅ embed_index imported successfully")
    except ImportError as e:
        print(f"❌ embed_index import failed: {e}")
        return False
    
    try:
        from rag_engine import RAGEngine, QueryProcessor
        print("✅ rag_engine imported successfully")
    except ImportError as e:
        print(f"❌ rag_engine import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment configuration"""
    print("\n🔍 Testing environment...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OpenAI API key found")
    else:
        print("❌ OpenAI API key not found")
        print("   Please set OPENAI_API_KEY in your .env file")
        return False
    
    # Test if directories exist or can be created
    dirs_to_check = ["./uploads", "./data"]
    for dir_path in dirs_to_check:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Directory {dir_path} ready")
        except Exception as e:
            print(f"❌ Cannot create directory {dir_path}: {e}")
            return False
    
    return True

def test_api_connection():
    """Test if the API is running and accessible"""
    print("\n🔍 Testing API connection...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API is running and healthy")
            print(f"   Status: {data.get('status')}")
            print(f"   OpenAI configured: {data.get('openai_configured')}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API at http://localhost:8000")
        print("   Make sure the FastAPI server is running: python app.py")
        return False
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality if API is available"""
    print("\n🔍 Testing basic functionality...")
    
    # Test suggestions endpoint
    try:
        response = requests.get("http://localhost:8000/suggestions", timeout=5)
        if response.status_code == 200:
            data = response.json()
            suggestions = data.get("suggestions", [])
            print(f"✅ Suggestions endpoint working ({len(suggestions)} suggestions)")
        else:
            print(f"❌ Suggestions endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Suggestions test failed: {e}")
        return False
    
    # Test stats endpoint
    try:
        response = requests.get("http://localhost:8000/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Stats endpoint working")
            print(f"   Uploaded files: {len(data.get('uploaded_files', []))}")
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("✨ Zodiac RAG Chatbot - Setup Test ✨")
    print("=" * 50)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Local Modules", test_local_modules),
        ("Environment", test_environment),
        ("API Connection", test_api_connection),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n🚀 Next steps:")
        print("1. Start the API: python app.py")
        print("2. Upload your PDFs via the /upload endpoint")
        print("3. Start the frontend: streamlit run streamlit_app.py")
        print("4. Ask questions about astrology!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up your .env file with OPENAI_API_KEY")
        print("3. Start the API server: python app.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 