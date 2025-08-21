#!/usr/bin/env python3
"""
Setup script for AlgoArena RAG Chatbot
This script helps users set up the environment and verify installation
"""

import subprocess
import sys
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version < (3, 8):
        print("❌ Python 3.8+ required. Current version:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    required_packages = [
        'streamlit',
        'transformers', 
        'torch',
        'faiss',
        'sentence_transformers',
        'langchain'
    ]

    print("🔍 Verifying installation...")
    for package in required_packages:
        try:
            if package == 'faiss':
                importlib.import_module('faiss')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not found")
            return False

    return True

def create_sample_document():
    """Create a sample document for testing"""
    sample_content = """# AlgoArena RAG Chatbot - Sample Document

## Overview
This is a sample document to test the RAG chatbot functionality.

## Key Features
- Document processing and text extraction
- Vector embedding generation using sentence-transformers
- FAISS vector database for efficient similarity search
- Hugging Face transformers for response generation
- Interactive Streamlit user interface

## Technical Architecture
The RAG (Retrieval-Augmented Generation) system works by:
1. Processing uploaded documents into text chunks
2. Converting text chunks into vector embeddings
3. Storing embeddings in a FAISS vector database
4. When a user asks a question, finding the most relevant chunks
5. Using those chunks as context for generating responses

## Benefits
- Accurate answers based on your specific documents
- Source citation for transparency
- Real-time document processing
- Scalable to large document collections

## Use Cases
- Technical documentation Q&A
- Research paper analysis
- Business document search
- Educational content exploration
"""

    with open("sample_document.txt", "w") as f:
        f.write(sample_content)

    print("✅ Created sample_document.txt for testing")

def main():
    """Main setup function"""
    print("🚀 AlgoArena RAG Chatbot Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        sys.exit(1)

    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        print("Try running: pip install -r requirements.txt")
        sys.exit(1)

    # Create sample document
    create_sample_document()

    print("\n🎉 Setup complete!")
    print("\n🚀 Next steps:")
    print("1. Run: streamlit run app.py")
    print("2. Upload sample_document.txt to test the system")
    print("3. Ask questions like 'What are the key features?'")
    print("\n📖 See README.md for detailed documentation")

if __name__ == "__main__":
    main()
