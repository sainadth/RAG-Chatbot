import streamlit as st
from datetime import datetime
import json
import pandas as pd

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False

    if 'doc_count' not in st.session_state:
        st.session_state.doc_count = 0

def display_chat_history():
    """Display chat history in the main interface"""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.write(content)

            # Display sources for assistant messages
            if role == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['filename']}")
                        st.markdown(f"*Similarity Score:* {source['similarity_score']:.3f}")
                        st.markdown(f"*Content:* {source['content'][:200]}...")
                        st.markdown("---")

def create_sidebar():
    """Create and populate the sidebar"""
    st.sidebar.title("🤖 RAG Chatbot Settings")

    # Model information
    with st.sidebar.expander("🔧 Model Information"):
        st.markdown("""
        **Embedding Model:** 
        - sentence-transformers/all-MiniLM-L6-v2

        **Language Model:**
        - microsoft/DialoGPT-small

        **Vector Database:**
        - FAISS (Facebook AI Similarity Search)
        """)

    # Chat management
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Management")

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []

    # Export chat history
    if st.session_state.chat_history:
        if st.sidebar.button("📤 Export Chat History"):
            export_chat_history()

    # Statistics
    if st.session_state.vector_store_ready:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Statistics")
        st.sidebar.metric("Documents Processed", st.session_state.doc_count)
        st.sidebar.metric("Chat Messages", len(st.session_state.chat_history))

def export_chat_history():
    """Export chat history as JSON"""
    try:
        # Prepare data for export
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "chat_history": st.session_state.chat_history,
            "session_info": {
                "vector_store_ready": st.session_state.vector_store_ready,
                "doc_count": st.session_state.doc_count
            }
        }

        # Convert to JSON
        json_data = json.dumps(export_data, indent=2)

        # Create download button
        st.sidebar.download_button(
            label="📥 Download Chat History",
            data=json_data,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    except Exception as e:
        st.sidebar.error(f"Error exporting chat history: {str(e)}")

def format_sources(sources):
    """Format sources for display"""
    if not sources:
        return "No sources available"

    formatted = []
    for i, source in enumerate(sources, 1):
        formatted.append(f"""
        **Source {i}:** {source['filename']}
        - Similarity: {source['similarity_score']:.3f}
        - Content: {source['content'][:150]}...
        """)

    return "\n".join(formatted)

def get_chat_statistics():
    """Get statistics about the current chat session"""
    if not st.session_state.chat_history:
        return {}

    user_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'user']
    assistant_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'assistant']

    return {
        "total_messages": len(st.session_state.chat_history),
        "user_messages": len(user_messages),
        "assistant_messages": len(assistant_messages),
        "avg_message_length": sum(len(msg['content']) for msg in st.session_state.chat_history) / len(st.session_state.chat_history)
    }
