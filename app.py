import streamlit as st
import os
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Import custom modules
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
from utils import initialize_session_state, display_chat_history, create_sidebar

# Page configuration
st.set_page_config(
    page_title="🤖 AlgoArena RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""

    # Initialize session state
    initialize_session_state()

    # Create sidebar
    create_sidebar()

    # Main header
    st.title("🤖 AlgoArena RAG Chatbot")
    st.markdown("### Chat with your documents using Retrieval-Augmented Generation")

    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Initializing RAG pipeline..."):
            st.session_state.rag_pipeline = RAGPipeline()

    # Document upload section
    st.sidebar.header("📄 Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files to chat with them"
    )

    process_clicked = st.sidebar.button("🔄 Process Documents")
    show_duplicate_warning = False
    if uploaded_files and process_clicked:
        uploaded_names = [f.name for f in uploaded_files]
        # Only show warning if last_uploaded_files exists and matches current upload
        if st.session_state.get('last_uploaded_files') and \
           set(st.session_state.last_uploaded_files) == set(uploaded_names) and \
           st.session_state.vector_store_ready:
            show_duplicate_warning = True
        else:
            # Store uploaded filenames to session to prevent reprocessing
            st.session_state.last_uploaded_files = uploaded_names
            process_documents(uploaded_files)
            uploaded_files = None

    # Knowledge base status
    if st.session_state.vector_store_ready:
        st.sidebar.success(f"✅ Knowledge base ready with {st.session_state.doc_count} documents")
        if show_duplicate_warning:
            st.sidebar.info("⚠️ These documents have already been processed.")
        # Display chat interface
        display_chat_interface()
    else:
        st.info("👆 Please upload documents to start chatting!")

        # Show demo section
        display_demo_section()

def process_documents(uploaded_files):
    """Process uploaded documents and create vector store"""
    try:
        uploaded_names = [f.name for f in uploaded_files]
        # Prevent repeated processing: check if already processed AND vector store is ready
        if st.session_state.get('last_uploaded_files') and \
           set(st.session_state.last_uploaded_files) == set(uploaded_names) and \
           st.session_state.vector_store_ready:
            st.sidebar.info("⚠️ These documents have already been processed.")
            return

        # Show overlay spinner and message during processing
        with st.spinner("⏳ Processing documents and generating embeddings... This may take several minutes for large files. Please wait."):
            processor = DocumentProcessor()
            all_chunks = []

            # Disable chat and upload during processing
            st.session_state.processing_in_progress = True

            # Process each uploaded file
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name

                # Process document
                chunks = processor.process_file(temp_path, uploaded_file.name)
                all_chunks.extend(chunks)
                print(len(chunks), "chunks created from", uploaded_file.name)

                # Clean up temp file
                os.unlink(temp_path)

            # Create vector store
            if all_chunks:
                st.session_state.rag_pipeline.create_vector_store(all_chunks)
                st.session_state.vector_store_ready = True
                st.session_state.doc_count = len(uploaded_files)
                st.session_state.processing_in_progress = False
                st.sidebar.success(f"✅ Processed {len(all_chunks)} chunks from {len(uploaded_files)} documents!")
                # Do NOT clear last_uploaded_files here; keep for duplicate check
                st.rerun()
            else:
                st.session_state.processing_in_progress = False
                st.sidebar.error("❌ No content extracted from documents")

    except Exception as e:
        st.session_state.processing_in_progress = False
        st.sidebar.error(f"❌ Error processing documents: {str(e)}")

def display_chat_interface():
    """Display the main chat interface"""
    # Disable chat input if processing is ongoing
    if st.session_state.get("processing_in_progress", False):
        st.info("⏳ Please wait while your documents are being processed. Chat will be enabled once processing is complete.")
        return

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history
        display_chat_history()

        # Chat input
        if user_question := st.chat_input("Ask me anything about your documents..."):
            handle_user_input(user_question)

def handle_user_input(user_question: str):
    """Handle user input and generate response"""

    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question,
        "timestamp": datetime.now().isoformat()
    })

    # Display user message
    with st.chat_message("user"):
        st.write(user_question)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get response from RAG pipeline
                response_data = st.session_state.rag_pipeline.get_response(user_question)

                # Display response
                st.write(response_data["answer"])

                # Display sources if available
                if response_data.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(response_data["sources"], 1):
                            st.markdown(f"**Source {i}:** {source['filename']}")
                            st.markdown(f"*Content:* {source['content'][:200]}...")
                            st.markdown("---")

                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_data["answer"],
                    "sources": response_data.get("sources", []),
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })

def display_demo_section():
    """Display demo information and sample questions"""
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚀 How it works")
        st.markdown("""
        1. **Upload Documents**: Add PDF, TXT, or DOCX files
        2. **Processing**: Documents are split into chunks and converted to vectors
        3. **Chat**: Ask questions and get answers based on your documents
        4. **Sources**: See which parts of your documents were used for answers
        """)

    with col2:
        st.markdown("### 💡 Sample Questions")
        st.markdown("""
        - "What is the main topic of the document?"
        - "Summarize the key points"
        - "What are the conclusions?"
        - "Explain the methodology used"
        """)

    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Model Architecture:**
        - **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
        - **Language Model**: microsoft/DialoGPT-medium (Hugging Face)
        - **Vector Database**: FAISS (Facebook AI Similarity Search)
        - **Text Chunking**: Recursive character splitting with overlap

        **RAG Pipeline:**
        1. Document processing and chunking
        2. Embedding generation for text chunks
        3. Vector store creation with FAISS
        4. Query embedding and similarity search
        5. Context-aware response generation
        6. Source attribution and citation
        """)

if __name__ == "__main__":
    main()
