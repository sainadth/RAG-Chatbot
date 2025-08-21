#!/usr/bin/env python3
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
        print("\n🚀 Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Check installation.")
        print("\n🔧 Try: python setup.py")

if __name__ == "__main__":
    main()
