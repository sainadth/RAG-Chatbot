# 🤖 AlgoArena RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, Hugging Face, and FAISS.  
Easily search, chat, and cite answers from your own documents.

## 🚀 Features

- Upload PDF, DOCX, TXT files
- Smart retrieval with FAISS
- Context-aware answers with transformers
- Source citation for every answer
- Export chat history
- Docker & CI/CD ready

## 🛠️ Quick Start

```bash
git clone <your-repo-url>
cd algoarena-rag-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## 🏗️ Architecture

- **DocumentProcessor**: File upload & chunking
- **RAGPipeline**: Embedding, retrieval, generation
- **FAISS**: Vector similarity search
- **Streamlit**: Chat UI
- **Utils**: Session & sidebar management

## 🐳 Docker

```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 rag-chatbot
```

## ⚙️ Publishing & GitHub

- Push to GitHub
- Use `.github/workflows/ci.yml` for CI/CD
- Add `.env.example` for environment variables
- Add `.gitignore` (see below)

## 📦 .gitignore

```
# filepath: c:\Users\spagadala1\Documents\RAG-Chatbot\.gitignore
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.db
*.sqlite3
.env
.env.*
.DS_Store
*.log
*.png
*.jpg
*.jpeg
*.pdf
*.docx
sample_documents/
rag_pipeline_flowchart.png
```

## 🤝 Contributing

- Fork, branch, PR
- Add tests to `test_rag.py`
- Document changes in `README.md`

## 📄 License

MIT License

---

Built for AlgoArena Hackathon 2025 🚀
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
