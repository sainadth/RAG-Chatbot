import os
import torch
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import re

class TextCleaner:
    """Clean and format text extracted from PDFs for better readability"""

    def clean_pdf_text(self, raw_text: str) -> str:
        """Clean raw PDF text for better readability"""
        if not raw_text or not raw_text.strip():
            return raw_text

        # Remove page numbers and artifacts like "13895 41 13897 Etc"
        text = re.sub(r'\b\d{4,}\s+\d+\s+\d{4,}\b', '', raw_text)
        text = re.sub(r'\b\d+\s+Etc\b', '', text)
        text = re.sub(r'--- Page \d+ ---', '', text)

        # Split into lines and clean
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip artifacts: very short lines, just numbers, single chars
            if len(line) < 3 or re.match(r'^[\d\s]+$', line):
                continue

            # Keep meaningful lines
            cleaned_lines.append(line)

        # Join lines with proper spacing
        cleaned_text = '\n'.join(cleaned_lines)

        # Fix fragmented sentences - join lines that don't end with punctuation
        lines = cleaned_text.split('\n')
        joined_lines = []
        current_sentence = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # If line starts with bullet point, finish current sentence and start new
            if line.startswith(('●', '•', '-', '*')):
                if current_sentence:
                    joined_lines.append(current_sentence.strip())
                    current_sentence = ""
                joined_lines.append('• ' + line.lstrip('●•-* '))
            # If line ends with punctuation, it's complete
            elif line.endswith(('.', ':', '?', '!')):
                current_sentence += " " + line if current_sentence else line
                joined_lines.append(current_sentence.strip())
                current_sentence = ""
            # Otherwise, it's a fragment - continue building sentence
            else:
                current_sentence += " " + line if current_sentence else line

        # Add any remaining sentence
        if current_sentence:
            joined_lines.append(current_sentence.strip())

        return '\n\n'.join(joined_lines)

class RAGPipeline:
    """RAG Pipeline with LOGICAL query responses that match user expectations"""

    def __init__(self):
        """Initialize RAG pipeline and models"""
        self.embedding_model = None
        self.vector_store = None
        self.text_chunks = []
        self.chunk_metadata = []
        self.text_cleaner = TextCleaner()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize embedding model"""
        try:
            st.write("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing models: {str(e)}")
            raise e

    def create_vector_store(self, text_chunks: List[Dict[str, Any]]):
        """Create FAISS vector store from cleaned text chunks"""
        try:
            cleaned_chunks = []
            for chunk in text_chunks:
                cleaned_content = self.text_cleaner.clean_pdf_text(chunk['content'])
                if cleaned_content and len(cleaned_content.strip()) > 50:
                    cleaned_chunk = chunk.copy()
                    cleaned_chunk['content'] = cleaned_content
                    cleaned_chunks.append(cleaned_chunk)

            self.text_chunks = [chunk['content'] for chunk in cleaned_chunks]
            self.chunk_metadata = cleaned_chunks

            if not self.text_chunks:
                raise Exception("No substantial content found after cleaning")

            st.write("🔄 Generating embeddings...")
            embeddings = self.embedding_model.encode(self.text_chunks, show_progress_bar=True)

            st.write("🔄 Creating vector store...")
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings.astype(np.float32))
            self.vector_store.add(embeddings.astype(np.float32))

            st.success(f"✅ Vector store created with {len(self.text_chunks)} cleaned chunks!")
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            raise e

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding.astype(np.float32))
            scores, indices = self.vector_store.search(query_embedding.astype(np.float32), k)

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
        """Generate LOGICAL responses based on query intent"""
        if not context_chunks:
            return "❌ No relevant information found."

        # Analyze query intent
        query_lower = query.lower().strip()
        filename = context_chunks[0].get('filename', 'document')

        # Combine all context
        all_content = " ".join([chunk.get('content', '') for chunk in context_chunks])

        # Route to appropriate handler based on ACTUAL user intent
        if self._is_summary_query(query_lower):
            return self._create_document_summary(all_content, filename)
        elif self._is_keywords_query(query_lower):
            return self._extract_keywords(all_content, filename)
        elif self._is_charts_query(query_lower):
            return self._find_charts_info(all_content, filename)
        elif self._is_explanation_query(query_lower):
            return self._provide_explanation(all_content, filename, query)
        elif self._is_topic_query(query_lower):
            return self._identify_main_topic(all_content, filename)
        else:
            return self._handle_general_query(query, all_content, filename)

    def _is_summary_query(self, query: str) -> bool:
        """Detect if user wants a summary"""
        return any(word in query for word in ['summary', 'summarize', 'overview', 'about document'])

    def _is_keywords_query(self, query: str) -> bool:
        """Detect if user wants keywords"""
        return any(word in query for word in ['keyword', 'key word', 'key terms', 'main terms'])

    def _is_charts_query(self, query: str) -> bool:
        """Detect if user wants chart information"""
        return any(word in query for word in ['chart', 'figure', 'graph', 'diagram', 'visualization'])

    def _is_explanation_query(self, query: str) -> bool:
        """Detect if user wants explanation"""
        return any(word in query for word in ['explain', 'how', 'why', 'what does', 'describe'])

    def _is_topic_query(self, query: str) -> bool:
        """Detect if user wants main topic"""
        return any(phrase in query for phrase in ['main topic', 'topic', 'subject', 'focus'])

    def _create_document_summary(self, content: str, filename: str) -> str:
        """Create a REAL document summary"""
        # Extract document title/subject
        title = self._extract_title(filename, content)

        # Identify document type and purpose  
        doc_type = self._identify_document_type(content, filename)

        # Extract key points
        key_points = self._extract_key_points(content)

        # Build comprehensive summary
        summary = f"## 📋 Document Summary: {title}\n\n"
        summary += f"**Document Type:** {doc_type}\n\n"

        # Main focus/purpose
        purpose = self._extract_purpose(content)
        if purpose:
            summary += f"**Purpose:** {purpose}\n\n"

        # Key findings/points
        if key_points:
            summary += "**Key Points:**\n"
            for point in key_points[:5]:
                summary += f"• {point}\n"
            summary += "\n"

        # Methods/techniques mentioned
        methods = self._extract_methods(content)
        if methods:
            summary += f"**Methods/Techniques:** {', '.join(methods)}\n\n"

        # Results/conclusions
        results = self._extract_results(content)
        if results:
            summary += f"**Key Results:** {results}"

        return summary

    def _extract_keywords(self, content: str, filename: str) -> str:
        """Extract actual KEYWORDS, not paragraphs"""
        # Technical terms
        tech_terms = set()

        # Common ML/AI terms
        ml_keywords = ['CNN', 'BiLSTM', 'Transformer', 'Random Forest', 'Neural Network', 
                      'Machine Learning', 'Deep Learning', 'Classification', 'Model Training']

        # Domain-specific terms (based on filename/content)
        if 'malicious' in filename.lower() or 'malicious' in content.lower():
            domain_terms = ['Malicious Domains', 'Cybersecurity', 'Domain Detection', 
                           'Security Classification', 'Threat Detection']
        else:
            domain_terms = []

        # Extract from content
        content_upper = content.upper()
        for term in ml_keywords + domain_terms:
            if term.upper() in content_upper:
                tech_terms.add(term)

        # Find capitalized terms (likely technical terms)
        capitalized_terms = re.findall(r'\b[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]+)*\b', content)
        for term in capitalized_terms:
            if len(term.split()) <= 3:  # Only short phrases
                tech_terms.add(term)

        # Performance metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC', 'AUC']
        for metric in metrics:
            if metric.lower() in content.lower():
                tech_terms.add(metric)

        # Format as keyword list
        if tech_terms:
            keywords_list = sorted(list(tech_terms))
            response = f"## 🔑 Key Terms from {filename}\n\n"

            # Group by category
            ml_found = [k for k in keywords_list if any(ml in k for ml in ['CNN', 'LSTM', 'Forest', 'Learning', 'Model'])]
            security_found = [k for k in keywords_list if any(sec in k for sec in ['Malicious', 'Domain', 'Security', 'Threat'])]
            metrics_found = [k for k in keywords_list if k in metrics]
            other_found = [k for k in keywords_list if k not in ml_found + security_found + metrics_found]

            if ml_found:
                response += "**Machine Learning:** " + ", ".join(ml_found) + "\n\n"
            if security_found:
                response += "**Cybersecurity:** " + ", ".join(security_found) + "\n\n"
            if metrics_found:
                response += "**Performance Metrics:** " + ", ".join(metrics_found) + "\n\n"
            if other_found:
                response += "**Other Terms:** " + ", ".join(other_found[:10])  # Limit to 10

            return response
        else:
            return f"## 🔑 Key Terms\n\nCould not extract specific keywords from {filename}. Try asking for a summary instead."

    def _find_charts_info(self, content: str, filename: str) -> str:
        """Find information about charts/figures"""
        charts_info = []

        # Look for figure/chart references
        figure_patterns = [
            r'Figure\s+(\d+)[.:](.*?)(?=\.|\n)',
            r'Fig\s+(\d+)[.:](.*?)(?=\.|\n)',
            r'Chart\s+(\d+)[.:](.*?)(?=\.|\n)',
            r'Graph\s+(\d+)[.:](.*?)(?=\.|\n)'
        ]

        for pattern in figure_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for num, desc in matches:
                charts_info.append(f"Figure {num}: {desc.strip()}")

        # Look for chart types mentioned
        chart_types = []
        chart_keywords = ['ROC curve', 'confusion matrix', 'bar chart', 'histogram', 
                         'scatter plot', 'line graph', 'heatmap', 'flowchart']

        for chart_type in chart_keywords:
            if chart_type.lower() in content.lower():
                chart_types.append(chart_type.title())

        # Build response
        response = f"## 📊 Charts & Figures in {filename}\n\n"

        if charts_info:
            response += "**Identified Figures:**\n"
            for info in charts_info[:10]:  # Limit to 10
                response += f"• {info}\n"
            response += "\n"

        if chart_types:
            response += f"**Chart Types Found:** {', '.join(set(chart_types))}\n\n"

        if not charts_info and not chart_types:
            response += "No specific chart references found. The document may contain visual elements not explicitly labeled in the text."

        return response

    def _extract_title(self, filename: str, content: str) -> str:
        """Extract document title"""
        # Clean filename
        title = filename.replace('.pdf', '').replace('annotated-', '').replace('%20', ' ')

        # Look for title in content (usually in first few lines)
        lines = content.split('\n')[:5]
        for line in lines:
            if len(line) > 10 and len(line) < 100:  # Reasonable title length
                if any(word in line.lower() for word in ['detection', 'analysis', 'system', 'study']):
                    return line.strip()

        return title

    def _identify_document_type(self, content: str, filename: str) -> str:
        """Identify document type"""
        if any(word in content.lower() for word in ['abstract', 'introduction', 'conclusion', 'references']):
            return "Research Paper/Academic Study"
        elif 'requirements' in filename.lower():
            return "Requirements Document"
        elif 'specification' in filename.lower():
            return "Technical Specification"
        else:
            return "Technical Document"

    def _extract_purpose(self, content: str) -> str:
        """Extract document purpose"""
        # Look for purpose indicators
        sentences = content.split('.')
        for sentence in sentences[:20]:  # Check first 20 sentences
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in ['purpose', 'aim', 'goal', 'objective']):
                return sentence
            elif sentence.startswith('This') and len(sentence) > 50:
                return sentence
        return None

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points"""
        points = []

        # Look for sentences with important keywords
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]

        for sentence in sentences[:50]:  # Check first 50 sentences
            if any(keyword in sentence.lower() for keyword in 
                   ['algorithm', 'model', 'method', 'approach', 'technique', 'result', 'finding']):
                if len(sentence) < 200:  # Keep it concise
                    points.append(sentence)
                    if len(points) >= 5:
                        break

        return points

    def _extract_methods(self, content: str) -> List[str]:
        """Extract methods/techniques"""
        methods = set()

        method_keywords = ['Random Forest', 'CNN', 'BiLSTM', 'Transformer', 'SVM', 'Naive Bayes',
                          'Neural Network', 'Deep Learning', 'Machine Learning']

        for method in method_keywords:
            if method.lower() in content.lower():
                methods.add(method)

        return list(methods)

    def _extract_results(self, content: str) -> str:
        """Extract key results"""
        # Look for results/conclusions
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in 
                   ['result', 'conclusion', 'finding', 'performance', 'accuracy']):
                if len(sentence) > 30 and len(sentence) < 150:
                    return sentence
        return None

    def _provide_explanation(self, content: str, filename: str, query: str) -> str:
        """Provide explanation based on query"""
        # Find most relevant sentences for explanation
        query_words = set(query.lower().split())
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]

        relevant_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append((overlap, sentence))

        # Sort by relevance
        relevant_sentences.sort(key=lambda x: x[0], reverse=True)

        response = f"## 💡 Explanation from {filename}\n\n"
        for _, sentence in relevant_sentences[:3]:
            response += f"{sentence}.\n\n"

        return response

    def _identify_main_topic(self, content: str, filename: str) -> str:
        """Identify main topic"""
        # Extract from filename
        topic_from_filename = filename.replace('.pdf', '').replace('annotated-', '').replace('%20', ' ')

        response = f"## 🎯 Main Topic\n\n"
        response += f"**Document:** {topic_from_filename}\n\n"

        # Identify domain
        if 'malicious' in content.lower():
            response += "**Domain:** Cybersecurity - Malicious Domain Detection\n\n"
        elif 'machine learning' in content.lower():
            response += "**Domain:** Machine Learning & AI\n\n"

        # Extract first substantial sentence as topic description
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 50]
        if sentences:
            response += f"**Description:** {sentences[0]}."

        return response

    def _handle_general_query(self, query: str, content: str, filename: str) -> str:
        """Handle general queries with focused responses"""
        query_words = set(query.lower().split())
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]

        relevant_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if len(query_words.intersection(sentence_words)) > 0:
                relevant_sentences.append(sentence)

        if relevant_sentences:
            response = f"**Information from {filename}:**\n\n"
            response += ".\n\n".join(relevant_sentences[:3]) + "."
        else:
            response = f"**From {filename}:**\n\n{sentences[0] if sentences else content[:300]}..."

        return response

    def get_response(self, query: str) -> Dict[str, Any]:
        """Main response method with logical outputs"""
        try:
            if self.vector_store is None:
                return {"answer": "❌ Please upload documents first.", "sources": []}

            relevant_chunks = self.retrieve_relevant_chunks(query, k=5)
            if not relevant_chunks:
                return {"answer": "❌ No relevant information found.", "sources": []}

            answer = self.generate_response(query, relevant_chunks)

            sources = []
            for chunk in relevant_chunks[:3]:
                sources.append({
                    "filename": chunk.get("filename", "Unknown"),
                    "content": chunk.get("content", "")[:200] + "...",
                    "similarity_score": chunk.get("similarity_score", 0.0),
                    "rank": chunk.get("rank", 0)
                })

            return {"answer": answer, "sources": sources, "query": query}

        except Exception as e:
            return {"answer": f"❌ Error: {str(e)}", "sources": []}

    def get_status(self) -> Dict[str, Any]:
        """Return health/status info for pipeline (for monitoring)"""
        return {
            "embedding_model_loaded": self.embedding_model is not None,
            "vector_store_ready": self.vector_store is not None,
            "num_chunks": len(self.text_chunks),
            "num_documents": len(set(chunk.get("filename") for chunk in self.chunk_metadata))
        }