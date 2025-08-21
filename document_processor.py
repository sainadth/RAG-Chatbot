import os
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
            separators=["\n\n", "\n", " ", ""]
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
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
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
                    text += paragraph.text + "\n"

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
