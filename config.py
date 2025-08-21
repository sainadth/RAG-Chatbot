"""
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
