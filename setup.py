#!/usr/bin/env python3
"""
Setup script for Zodiac RAG Chatbot
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major != 3 or version.minor not in [10, 11]:
        print(f"❌ Python 3.10 or 3.11 required, found {version.major}.{version.minor}")
        print("   Please install Python 3.10 or 3.11 for stability")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible (stable version)")
    return True

def create_directories():
    """Create necessary directories"""
    print("🔧 Creating directories...")
    directories = [
        "uploads",
        "data",
        "logs"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("🔧 Creating .env file...")
    env_content = """# Zodiac RAG Chatbot Configuration
# Replace with your actual OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002

# File Paths
FAISS_INDEX_PATH=./data/faiss_index
UPLOAD_DIR=./uploads

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("⚠️  Please edit .env file and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("🔧 Installing dependencies...")
    
    # Check if pip is available
    if not shutil.which("pip"):
        print("❌ pip not found. Please install pip first.")
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    return True

def main():
    """Main setup function"""
    print("✨ Zodiac RAG Chatbot - Setup Wizard ✨")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("❌ Setup failed: Incompatible Python version")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed: Could not create directories")
        return False
    
    # Create .env file
    if not create_env_file():
        print("❌ Setup failed: Could not create .env file")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed: Could not install dependencies")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run the test script: python test_setup.py")
    print("3. Start the API: python app.py")
    print("4. Upload your PDFs via the /upload endpoint")
    print("5. Start the frontend: streamlit run streamlit_app.py")
    print("\n📚 Required PDFs:")
    print("- Linda Goodman's Sun Signs")
    print("- Linda Goodman's Love Signs")
    print("\n🔗 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 