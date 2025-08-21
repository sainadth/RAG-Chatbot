# Create additional helpful files for the RAG project

# 1. Setup script
setup_py = '''#!/usr/bin/env python3
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
    
    print("\\n🎉 Setup complete!")
    print("\\n🚀 Next steps:")
    print("1. Run: streamlit run app.py")
    print("2. Upload sample_document.txt to test the system")
    print("3. Ask questions like 'What are the key features?'")
    print("\\n📖 See README.md for detailed documentation")

if __name__ == "__main__":
    main()
'''

# 2. Test script
test_rag_py = '''#!/usr/bin/env python3
"""
Test script for RAG components
Run this to verify that core functionality works
"""

import sys
import tempfile
import os
from pathlib import Path

def test_document_processing():
    """Test document processing functionality"""
    print("🔍 Testing document processing...")
    
    try:
        from document_processor import DocumentProcessor
        
        # Create test content
        test_content = "This is a test document. It contains multiple sentences to test chunking."
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        # Test processing
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        chunks = processor.process_file(temp_path, "test.txt")
        
        # Cleanup
        os.unlink(temp_path)
        
        if chunks:
            print(f"✅ Document processing: {len(chunks)} chunks created")
            return True
        else:
            print("❌ Document processing: No chunks created")
            return False
            
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return False

def test_embeddings():
    """Test embedding generation"""
    print("🔍 Testing embedding generation...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_texts = ["This is a test sentence.", "Another test sentence."]
        
        embeddings = model.encode(test_texts)
        
        if embeddings.shape[0] == 2:
            print(f"✅ Embeddings: Shape {embeddings.shape}")
            return True
        else:
            print("❌ Embeddings: Incorrect shape")
            return False
            
    except Exception as e:
        print(f"❌ Embedding generation error: {e}")
        return False

def test_vector_store():
    """Test FAISS vector store"""
    print("🔍 Testing FAISS vector store...")
    
    try:
        import faiss
        import numpy as np
        
        # Create test vectors
        dimension = 384
        vectors = np.random.random((5, dimension)).astype(np.float32)
        
        # Create FAISS index
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        
        # Test search
        query = np.random.random((1, dimension)).astype(np.float32)
        faiss.normalize_L2(query)
        
        scores, indices = index.search(query, 2)
        
        if len(indices[0]) == 2:
            print("✅ FAISS vector store: Search working")
            return True
        else:
            print("❌ FAISS vector store: Search failed")
            return False
            
    except Exception as e:
        print(f"❌ FAISS vector store error: {e}")
        return False

def test_language_model():
    """Test language model loading"""
    print("🔍 Testing language model...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer and model:
            print("✅ Language model: Loaded successfully")
            return True
        else:
            print("❌ Language model: Failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Language model error: {e}")
        return False

def test_streamlit():
    """Test Streamlit installation"""
    print("🔍 Testing Streamlit...")
    
    try:
        import streamlit
        print(f"✅ Streamlit: Version {streamlit.__version__}")
        return True
    except Exception as e:
        print(f"❌ Streamlit error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAG Chatbot Component Tests")
    print("=" * 40)
    
    tests = [
        test_streamlit,
        test_embeddings,
        test_vector_store,
        test_language_model,
        test_document_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\\n🚀 Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Check installation.")
        print("\\n🔧 Try: python setup.py")

if __name__ == "__main__":
    main()
'''

# 3. Docker configuration
dockerfile = '''# Dockerfile for RAG Chatbot
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

# 4. GitHub Actions workflow
github_workflow = '''name: RAG Chatbot CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python test_rag.py
    
    - name: Test Streamlit app
      run: |
        timeout 30 streamlit run app.py --server.headless=true || true
'''

# 5. Sample documents for testing
sample_ai_doc = '''# Artificial Intelligence and Machine Learning Guide

## Introduction
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.

## Machine Learning Fundamentals

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common algorithms include:
- Linear Regression
- Decision Trees
- Random Forest
- Support Vector Machines
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds patterns in data without labeled examples:
- Clustering (K-means, Hierarchical)
- Dimensionality Reduction (PCA, t-SNE)
- Association Rules

### Reinforcement Learning
An agent learns to make decisions by interacting with an environment and receiving rewards or penalties.

## Deep Learning
Deep learning uses neural networks with multiple layers to learn complex patterns:

### Neural Network Architectures
- Feedforward Networks
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformer Architecture

### Applications
- Computer Vision
- Natural Language Processing
- Speech Recognition
- Autonomous Vehicles

## RAG Systems
Retrieval-Augmented Generation combines information retrieval with text generation:

1. **Document Processing**: Convert documents into searchable chunks
2. **Embedding Generation**: Create vector representations of text
3. **Vector Storage**: Store embeddings in a database
4. **Query Processing**: Convert questions into vectors
5. **Retrieval**: Find relevant document chunks
6. **Generation**: Use retrieved context to generate answers

## Best Practices
- Data Quality is crucial
- Regular model evaluation
- Consider ethical implications
- Continuous learning and adaptation
'''

sample_tech_doc = '''# Software Development Best Practices

## Version Control
Version control is essential for tracking changes and collaboration:

### Git Workflow
- Feature branches for new development
- Pull requests for code review
- Continuous integration
- Regular commits with clear messages

## Code Quality

### Clean Code Principles
1. **Meaningful Names**: Use descriptive variable and function names
2. **Single Responsibility**: Functions should do one thing well
3. **DRY Principle**: Don't Repeat Yourself
4. **Comments**: Explain why, not what

### Testing
- Unit tests for individual components
- Integration tests for system interactions
- End-to-end tests for user workflows
- Test-driven development (TDD)

## Architecture Patterns

### Microservices
Benefits:
- Independent deployment
- Technology diversity
- Fault isolation
- Scalability

Challenges:
- Distributed system complexity
- Service communication
- Data consistency
- Monitoring and debugging

### RESTful APIs
Design principles:
- Resource-based URLs
- HTTP methods (GET, POST, PUT, DELETE)
- Stateless communication
- JSON data format
- Proper status codes

## DevOps Practices

### Continuous Integration/Continuous Deployment (CI/CD)
- Automated testing
- Code quality checks
- Automated deployment
- Infrastructure as code

### Monitoring
- Application performance monitoring
- Error tracking and alerting
- Log aggregation and analysis
- Health checks

## Security
- Input validation
- Authentication and authorization
- HTTPS encryption
- Regular security updates
- Vulnerability scanning

## Documentation
- README files
- API documentation
- Code comments
- Architecture diagrams
- User guides
'''

# Save all additional files
additional_files = {
    'setup.py': setup_py,
    'test_rag.py': test_rag_py,
    'Dockerfile': dockerfile,
    '.github/workflows/ci.yml': github_workflow,
    'sample_documents/ai_ml_guide.txt': sample_ai_doc,
    'sample_documents/software_dev_guide.txt': sample_tech_doc
}

print("📁 Creating additional project files...")
print("=" * 50)

for filepath, content in additional_files.items():
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ {filepath}")

print("=" * 50)
print("🎯 Additional files created for AlgoArena project!")
print("\n📋 Complete project structure:")
print("""
algoarena-rag-chatbot/
├── 📄 Core Application Files
│   ├── app.py                    # Main Streamlit app
│   ├── rag_pipeline.py           # RAG implementation  
│   ├── document_processor.py     # Document handling
│   ├── utils.py                  # Utility functions
│   └── config.py                 # Configuration
├── 📦 Setup & Testing
│   ├── requirements.txt          # Dependencies
│   ├── setup.py                  # Setup script
│   ├── test_rag.py              # Test suite
│   └── .env.example             # Environment template
├── 🐳 Deployment
│   ├── Dockerfile               # Container config
│   └── .github/workflows/ci.yml # CI/CD pipeline
├── 📚 Documentation & Samples
│   ├── README.md                # Project documentation
│   └── sample_documents/        # Test documents
│       ├── ai_ml_guide.txt
│       └── software_dev_guide.txt
""")

print("\n🚀 Quick Start Commands:")
print("1. python setup.py          # Set up environment")
print("2. python test_rag.py       # Test components") 
print("3. streamlit run app.py     # Launch app")
print("4. Upload sample documents and start chatting!")