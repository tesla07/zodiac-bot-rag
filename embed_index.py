import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()


class EmbeddingManager:
    """Manages OpenAI embeddings for text chunks"""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.embedding_dim = 1536  # OpenAI ada-002 embedding dimension
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text chunk"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Get embeddings for multiple text chunks in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add None embeddings for failed chunks
                embeddings.extend([None] * len(batch))
        
        return embeddings


class FAISSIndex:
    """Manages FAISS vector index for similarity search"""
    
    def __init__(self, index_path: str = "./data/faiss_index"):
        self.index_path = index_path
        self.index = None
        self.chunks_metadata = []
        self.embedding_dim = 1536
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    def create_index(self, embeddings: List[List[float]]) -> None:
        """Create FAISS index from embeddings"""
        if not embeddings:
            raise ValueError("No embeddings provided")
        
        # Filter out None embeddings
        valid_embeddings = []
        valid_indices = []
        
        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_indices.append(i)
        
        if not valid_embeddings:
            raise ValueError("No valid embeddings found")
        
        # Convert to numpy array
        embeddings_array = np.array(valid_embeddings, dtype=np.float32)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.index.add(embeddings_array)
        
        print(f"Created FAISS index with {len(valid_embeddings)} vectors")
        
        # Store valid indices for metadata mapping
        self.valid_indices = valid_indices
    
    def add_to_index(self, embeddings: List[List[float]]) -> None:
        """Add new embeddings to existing index"""
        if self.index is None:
            self.create_index(embeddings)
            return
        
        # Filter out None embeddings
        valid_embeddings = []
        valid_indices = []
        
        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_indices.append(i)
        
        if not valid_embeddings:
            print("No valid embeddings to add")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(valid_embeddings, dtype=np.float32)
        
        # Add to existing index
        self.index.add(embeddings_array)
        
        # Update valid indices
        start_idx = len(self.chunks_metadata)
        self.valid_indices.extend([start_idx + i for i in valid_indices])
        
        print(f"Added {len(valid_embeddings)} vectors to existing index")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Search for similar vectors"""
        if self.index is None:
            raise ValueError("Index not initialized")
        
        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        return indices[0].tolist(), distances[0].tolist()
    
    def get_chunk_by_index(self, index: int) -> Dict[str, Any]:
        """Get chunk metadata by FAISS index"""
        if index < len(self.chunks_metadata):
            return self.chunks_metadata[index]
        return None
    
    def save_index(self) -> None:
        """Save FAISS index and metadata"""
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        
        # Save metadata
        metadata_path = f"{self.index_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.chunks_metadata, f, indent=2)
        
        print(f"Saved index to {self.index_path}.faiss")
        print(f"Saved metadata to {metadata_path}")
    
    def load_index(self) -> bool:
        """Load existing FAISS index and metadata"""
        faiss_path = f"{self.index_path}.faiss"
        metadata_path = f"{self.index_path}_metadata.json"
        
        if not (os.path.exists(faiss_path) and os.path.exists(metadata_path)):
            print("Index files not found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(faiss_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.chunks_metadata = json.load(f)
            
            print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks_metadata)} metadata entries")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if self.index is None:
            return {"status": "No index loaded"}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "metadata_entries": len(self.chunks_metadata),
            "index_type": type(self.index).__name__
        }


class EmbedIndexManager:
    """Main class that combines embedding and indexing functionality"""
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002", index_path: str = "./data/faiss_index"):
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.index_manager = FAISSIndex(index_path)
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Process chunks: create embeddings and add to index"""
        if not chunks:
            print("No chunks to process")
            return
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Get embeddings
        print("Creating embeddings...")
        embeddings = self.embedding_manager.get_embeddings_batch(texts)
        
        # Filter out chunks with failed embeddings
        valid_chunks = []
        valid_embeddings = []
        
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            if emb is not None:
                valid_chunks.append(chunk)
                valid_embeddings.append(emb)
            else:
                print(f"Failed to get embedding for chunk {i}")
        
        if not valid_chunks:
            print("No valid chunks after embedding")
            return
        
        # Add to index
        if self.index_manager.index is None:
            self.index_manager.create_index(valid_embeddings)
        else:
            self.index_manager.add_to_index(valid_embeddings)
        
        # Store metadata
        self.index_manager.chunks_metadata.extend(valid_chunks)
        
        print(f"Successfully processed {len(valid_chunks)} chunks")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks given a query"""
        # Get query embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        if query_embedding is None:
            return []
        
        # Search index
        indices, scores = self.index_manager.search(query_embedding, top_k)
        
        # Get chunks
        results = []
        for idx, score in zip(indices, scores):
            chunk = self.index_manager.get_chunk_by_index(idx)
            if chunk:
                chunk_copy = chunk.copy()
                chunk_copy['similarity_score'] = score
                results.append(chunk_copy)
        
        return results
    
    def save_index(self) -> None:
        """Save the current index"""
        self.index_manager.save_index()
    
    def load_index(self) -> bool:
        """Load existing index"""
        return self.index_manager.load_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return self.index_manager.get_index_stats() 