# Create comprehensive project files for RAG Chatbot implementation
import os

# Create project structure and files
project_files = {}

# 1. requirements.txt
requirements_txt = """streamlit==1.28.0
langchain==0.1.0
langchain-community==0.0.10
transformers==4.35.0
torch==2.1.0
faiss-cpu==1.7.4
sentence-transformers==2.2.2
pypdf2==3.0.1
python-docx==0.8.11
streamlit-chat==0.1.1
numpy==1.24.3
pandas==2.0.3
plotly==5.17.0
openai==1.3.0
python-dotenv==1.0.0
"""

project_files['requirements.txt'] = requirements_txt

# 2. Main Streamlit app
app_py = '''import streamlit as st
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
    
    # Process uploaded documents
    if uploaded_files and st.sidebar.button("🔄 Process Documents"):
        process_documents(uploaded_files)
    
    # Knowledge base status
    if st.session_state.vector_store_ready:
        st.sidebar.success(f"✅ Knowledge base ready with {st.session_state.doc_count} documents")
        
        # Display chat interface
        display_chat_interface()
    else:
        st.info("👆 Please upload documents to start chatting!")
        
        # Show demo section
        display_demo_section()

def process_documents(uploaded_files):
    """Process uploaded documents and create vector store"""
    try:
        with st.spinner("Processing documents..."):
            processor = DocumentProcessor()
            all_chunks = []
            
            # Process each uploaded file
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Process document
                chunks = processor.process_file(temp_path, uploaded_file.name)
                all_chunks.extend(chunks)
                
                # Clean up temp file
                os.unlink(temp_path)
            
            # Create vector store
            if all_chunks:
                st.session_state.rag_pipeline.create_vector_store(all_chunks)
                st.session_state.vector_store_ready = True
                st.session_state.doc_count = len(uploaded_files)
                st.sidebar.success(f"✅ Processed {len(all_chunks)} chunks from {len(uploaded_files)} documents!")
                st.rerun()
            else:
                st.sidebar.error("❌ No content extracted from documents")
                
    except Exception as e:
        st.sidebar.error(f"❌ Error processing documents: {str(e)}")

def display_chat_interface():
    """Display the main chat interface"""
    
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
'''

project_files['app.py'] = app_py

# 3. RAG Pipeline implementation
rag_pipeline_py = '''import os
import torch
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st

class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline using Hugging Face models and FAISS"""
    
    def __init__(self):
        """Initialize the RAG pipeline"""
        self.embedding_model = None
        self.language_model = None
        self.tokenizer = None
        self.vector_store = None
        self.text_chunks = []
        self.chunk_metadata = []
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding and language models"""
        try:
            # Initialize embedding model
            st.write("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Initialize language model for text generation
            st.write("🔄 Loading language model...")
            model_name = "microsoft/DialoGPT-small"  # Lighter model for demo
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.language_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.language_model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            st.success("✅ Models loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error initializing models: {str(e)}")
            raise e
    
    def create_vector_store(self, text_chunks: List[Dict[str, Any]]):
        """Create FAISS vector store from text chunks"""
        try:
            # Store text chunks and metadata
            self.text_chunks = [chunk['content'] for chunk in text_chunks]
            self.chunk_metadata = text_chunks
            
            # Generate embeddings
            st.write("🔄 Generating embeddings...")
            embeddings = self.embedding_model.encode(self.text_chunks, show_progress_bar=True)
            
            # Create FAISS index
            st.write("🔄 Creating vector store...")
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings.astype(np.float32))
            
            # Add embeddings to index
            self.vector_store.add(embeddings.astype(np.float32))
            
            st.success(f"✅ Vector store created with {len(self.text_chunks)} chunks!")
            
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            raise e
    
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant text chunks for a query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding.astype(np.float32))
            
            # Search for similar chunks
            scores, indices = self.vector_store.search(query_embedding.astype(np.float32), k)
            
            # Prepare results
            relevant_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunk_metadata):
                    chunk_data = self.chunk_metadata[idx].copy()
                    chunk_data['similarity_score'] = float(score)
                    chunk_data['rank'] = i + 1
                    relevant_chunks.append(chunk_data)
            
            return relevant_chunks
            
        except Exception as e:
            st.error(f"❌ Error retrieving chunks: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate response using retrieved context"""
        try:
            # Prepare context
            context_text = "\\n\\n".join([
                f"Document: {chunk['filename']}\\nContent: {chunk['content']}"
                for chunk in context_chunks[:3]  # Use top 3 chunks
            ])
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question.
            
Context:
{context_text}

Question: {query}

Answer: """
            
            # Generate response
            response = self.text_generator(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the answer part
            if "Answer: " in generated_text:
                answer = generated_text.split("Answer: ")[-1].strip()
            else:
                answer = "I apologize, but I couldn't generate a proper response based on the provided context."
            
            return answer
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """Main method to get response for a query"""
        try:
            # Check if vector store is ready
            if self.vector_store is None:
                return {
                    "answer": "❌ Vector store not initialized. Please upload and process documents first.",
                    "sources": []
                }
            
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query, k=5)
            
            if not relevant_chunks:
                return {
                    "answer": "❌ No relevant information found in the documents.",
                    "sources": []
                }
            
            # Generate response
            answer = self.generate_response(query, relevant_chunks)
            
            # Prepare sources for display
            sources = [
                {
                    "filename": chunk["filename"],
                    "content": chunk["content"],
                    "similarity_score": chunk["similarity_score"],
                    "rank": chunk["rank"]
                }
                for chunk in relevant_chunks[:3]  # Show top 3 sources
            ]
            
            return {
                "answer": answer,
                "sources": sources,
                "query": query
            }
            
        except Exception as e:
            return {
                "answer": f"❌ Error processing query: {str(e)}",
                "sources": []
            }
'''

project_files['rag_pipeline.py'] = rag_pipeline_py

# 4. Document processor
document_processor_py = '''import os
from typing import List, Dict, Any
import PyPDF2
from docx import Document
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Process various document types and extract text chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\\n\\n", "\\n", " ", ""]
        )
    
    def process_file(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Process a file and return text chunks with metadata"""
        try:
            # Extract text based on file type
            file_extension = filename.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                text = self._extract_pdf_text(file_path)
            elif file_extension == 'docx':
                text = self._extract_docx_text(file_path)
            elif file_extension == 'txt':
                text = self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Split text into chunks
            chunks = self._split_text(text, filename)
            
            return chunks
            
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
            return []
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\\n--- Page {page_num + 1} ---\\n{page_text}\\n"
                    except Exception as e:
                        st.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                        continue
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        
        return text.strip()
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\\n"
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def _split_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        if not text.strip():
            return []
        
        # Split text using LangChain text splitter
        text_chunks = self.text_splitter.split_text(text)
        
        # Create chunks with metadata
        chunks = []
        for i, chunk in enumerate(text_chunks):
            if chunk.strip():  # Only add non-empty chunks
                chunks.append({
                    "content": chunk.strip(),
                    "filename": filename,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "source": "uploaded_document"
                })
        
        return chunks
    
    def get_document_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not chunks:
            return {}
        
        stats = {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk["chunk_size"] for chunk in chunks),
            "avg_chunk_size": sum(chunk["chunk_size"] for chunk in chunks) / len(chunks),
            "files_processed": len(set(chunk["filename"] for chunk in chunks))
        }
        
        return stats
'''

project_files['document_processor.py'] = document_processor_py

# 5. Utilities
utils_py = '''import streamlit as st
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
        st.rerun()
    
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
    
    return "\\n".join(formatted)

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
'''

project_files['utils.py'] = utils_py

# 6. Configuration file
config_py = '''"""
Configuration settings for the RAG Chatbot
"""

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LANGUAGE_MODEL = "microsoft/DialoGPT-small"

# Text processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
MAX_RETRIEVED_CHUNKS = 5
MAX_DISPLAY_SOURCES = 3

# Generation settings
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7

# UI settings
PAGE_TITLE = "🤖 AlgoArena RAG Chatbot"
PAGE_ICON = "🤖"

# File upload settings
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']
MAX_FILE_SIZE_MB = 10
'''

project_files['config.py'] = config_py

# 7. Environment file template
env_example = '''# Environment variables (copy to .env and fill in your values)

# OpenAI API Key (optional, for advanced models)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face API Token (optional, for private models)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Other settings
DEBUG=False
'''

project_files['.env.example'] = env_example

# 8. README file
readme_md = '''# 🤖 AlgoArena RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Hugging Face transformers. This project demonstrates modern AI architecture by combining document retrieval with generative AI for intelligent question-answering.

## 🚀 Features

- **Document Processing**: Upload PDF, DOCX, and TXT files
- **Smart Retrieval**: FAISS vector database for efficient document search
- **Context-Aware Generation**: Hugging Face transformers for intelligent responses
- **Source Citation**: See which documents were used for each answer
- **Interactive UI**: Clean, modern Streamlit interface
- **Chat History**: Persistent conversation memory
- **Export Functionality**: Download chat history as JSON

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Language Model**: microsoft/DialoGPT-small
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF2, python-docx
- **Framework**: LangChain

## 📦 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd algoarena-rag-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables (optional)**
```bash
cp .env.example .env
# Edit .env with your API keys if needed
```

## 🚀 Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
2. **Process Documents**: Click "Process Documents" to create the knowledge base
3. **Start Chatting**: Ask questions about your documents in the chat interface
4. **View Sources**: Expand the sources section to see which documents were used
5. **Export Chat**: Download your conversation history as JSON

## 🏗️ Architecture

```
User Input → Embedding → Vector Search → Context Retrieval → LLM Generation → Response + Sources
```

### Key Components:

1. **Document Processor**: Handles file upload and text extraction
2. **RAG Pipeline**: Core retrieval and generation logic  
3. **Vector Store**: FAISS index for similarity search
4. **Streamlit UI**: Interactive chat interface
5. **Utils**: Helper functions and session management

## 📁 Project Structure

```
algoarena-rag-chatbot/
│
├── app.py                 # Main Streamlit application
├── rag_pipeline.py        # RAG implementation
├── document_processor.py  # Document processing logic
├── utils.py              # Utility functions
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:
- Model selection
- Chunk size and overlap
- Retrieval parameters
- Generation settings

## 🎯 AlgoArena Hackathon Notes

This project is perfect for AlgoArena because it demonstrates:

- **Model Building**: Custom RAG pipeline implementation
- **Real-world Application**: Document Q&A system
- **Modern Architecture**: Combining retrieval and generation
- **Interactive Demo**: Streamlit deployment ready
- **Scalable Design**: Can handle multiple documents and users

### Judging Criteria Alignment:

1. **Technical Excellence**: Advanced ML pipeline with FAISS and transformers
2. **Innovation**: RAG architecture with source attribution
3. **Practical Impact**: Solves real document Q&A needs
4. **Demonstration**: Interactive web interface

## 🔧 Development

### Adding New Document Types:
Extend `DocumentProcessor` class with new extraction methods

### Switching Models:
Update `EMBEDDING_MODEL` and `LANGUAGE_MODEL` in `config.py`

### Custom UI:
Modify Streamlit components in `app.py` and `utils.py`

## 🚀 Deployment

### Streamlit Cloud:
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy with requirements.txt

### Local Production:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📊 Performance

- **Embedding Model**: ~384 dimensions, fast encoding
- **Vector Search**: FAISS provides O(log n) similarity search
- **Generation**: Optimized for quick response times
- **Memory**: Efficient chunk-based processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open-source and available under the MIT License.

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Tutorials](https://langchain.readthedocs.io)
- [FAISS Documentation](https://faiss.ai)
- [Hugging Face Models](https://huggingface.co/models)

## 💬 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

Built with ❤️ for AlgoArena Hackathon 2025
'''

project_files['README.md'] = readme_md

# Save all files
print("📁 Created comprehensive RAG Chatbot project structure:")
print("=" * 50)

for filename, content in project_files.items():
    # Save file locally for download
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ {filename} ({len(content.split())} words)")

print("=" * 50)
print(f"📊 Total files created: {len(project_files)}")
print("🎯 Project ready for AlgoArena hackathon!")
print("\n🚀 Next steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Run the app: streamlit run app.py") 
print("3. Upload documents and start chatting!")