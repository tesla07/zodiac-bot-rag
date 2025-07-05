import fitz  # PyMuPDF
import os
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from unstructured.partition.auto import partition


class PDFParser:
    """Handles PDF parsing using PyMuPDF and unstructured"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def parse_pdf(self, file_path: str) -> str:
        """Parse PDF and extract text content"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try PyMuPDF first
        try:
            return self._parse_with_pymupdf(file_path)
        except Exception as e:
            print(f"PyMuPDF failed, trying unstructured: {e}")
            return self._parse_with_unstructured(file_path)
    
    def _parse_with_pymupdf(self, file_path: str) -> str:
        """Parse PDF using PyMuPDF"""
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    
    def _parse_with_unstructured(self, file_path: str) -> str:
        """Parse PDF using unstructured as fallback"""
        elements = partition(filename=file_path)
        text = ""
        
        for element in elements:
            if hasattr(element, 'text'):
                text += element.text + "\n"
        
        return text


class AdaptiveChunker:
    """Performs adaptive, embedding-aware chunking based on semantic similarity"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, similarity_threshold: float = 0.7):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by paragraph breaks (double newlines or large spacing)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs and very short ones
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        return paragraphs
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text chunks"""
        try:
            # Use TF-IDF for similarity calculation
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            return 0.0
    
    def merge_similar_chunks(self, chunks: List[str]) -> List[str]:
        """Merge adjacent chunks based on semantic similarity"""
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            # Calculate similarity between current and next chunk
            similarity = self.calculate_similarity(current_chunk, next_chunk)
            
            # If similar enough and combined length is reasonable, merge
            if (similarity > self.similarity_threshold and 
                len(current_chunk) + len(next_chunk) <= self.chunk_size * 1.5):
                current_chunk += "\n\n" + next_chunk
            else:
                # Add current chunk to results and start new chunk
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # Add the last chunk
        merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create adaptive chunks from text"""
        # Split into paragraphs first
        paragraphs = self.split_into_paragraphs(text)
        
        # Merge similar adjacent paragraphs
        merged_chunks = self.merge_similar_chunks(paragraphs)
        
        # Further split large chunks if needed
        final_chunks = []
        chunk_id = 0
        
        for chunk in merged_chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append({
                    'id': chunk_id,
                    'text': chunk,
                    'length': len(chunk)
                })
                chunk_id += 1
            else:
                # Split large chunks with overlap
                sub_chunks = self._split_large_chunk(chunk)
                for sub_chunk in sub_chunks:
                    final_chunks.append({
                        'id': chunk_id,
                        'text': sub_chunk,
                        'length': len(sub_chunk)
                    })
                    chunk_id += 1
        
        return final_chunks
    
    def _split_large_chunk(self, chunk: str) -> List[str]:
        """Split large chunks with overlap"""
        if len(chunk) <= self.chunk_size:
            return [chunk]
        
        sub_chunks = []
        start = 0
        
        while start < len(chunk):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(chunk):
                # Look for sentence endings near the end
                for i in range(end, max(start + self.chunk_size - 100, start), -1):
                    if chunk[i] in '.!?':
                        end = i + 1
                        break
            
            sub_chunk = chunk[start:end].strip()
            if sub_chunk:
                sub_chunks.append(sub_chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(chunk):
                break
        
        return sub_chunks


class ParserChunker:
    """Main class that combines PDF parsing and adaptive chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.parser = PDFParser()
        self.chunker = AdaptiveChunker(chunk_size, chunk_overlap)
    
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a PDF file and return chunks with metadata"""
        print(f"Processing PDF: {file_path}")
        
        # Parse PDF
        text = self.parser.parse_pdf(file_path)
        print(f"Extracted {len(text)} characters from PDF")
        
        # Create chunks
        chunks = self.chunker.create_chunks(text)
        print(f"Created {len(chunks)} chunks")
        
        # Add file metadata
        filename = os.path.basename(file_path)
        for chunk in chunks:
            chunk['source_file'] = filename
            chunk['source_path'] = file_path
        
        return chunks 