"""
Semantic Search - Document embedding and similarity search utilities
Embedding generation, FAISS index management, and semantic search functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import faiss
import pickle
import json
import logging
from datetime import datetime
import os
from pathlib import Path
import hashlib
import re
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Data class for document representation"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    embedding_generated: bool = False

@dataclass
class SearchResult:
    """Data class for search results"""
    document: Document
    similarity_score: float
    rank: int
    highlighted_content: str
    relevance_reason: str

class DocumentProcessor:
    """Document processing and text extraction utilities"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.md']
        self.text_cleaners = []
        self.setup_text_cleaners()
    
    def setup_text_cleaners(self):
        """Setup text cleaning functions"""
        self.text_cleaners = [
            self.remove_extra_whitespace,
            self.remove_special_characters,
            self.normalize_quotes,
            self.fix_encoding
        ]
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def remove_special_characters(self, text: str) -> str:
        """Remove or replace special characters"""
        # Keep alphanumeric, spaces, punctuation, and common symbols
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']', ' ', text)
        return text
    
    def normalize_quotes(self, text: str) -> str:
        """Normalize quote characters"""
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text
    
    def fix_encoding(self, text: str) -> str:
        """Fix common encoding issues"""
        # Handle common encoding problems
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        return text
    
    def clean_text(self, text: str) -> str:
        """Apply all text cleaning functions"""
        for cleaner in self.text_cleaners:
            text = cleaner(text)
        return text
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from various file formats
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            elif file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            elif file_path.suffix.lower() == '.pdf':
                # TODO: Implement PDF text extraction
                # For now, return placeholder
                text = f"PDF content from {file_path.name} would be extracted here..."
            
            elif file_path.suffix.lower() == '.docx':
                # TODO: Implement DOCX text extraction
                # For now, return placeholder
                text = f"DOCX content from {file_path.name} would be extracted here..."
            
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Clean the extracted text
            cleaned_text = self.clean_text(text)
            logger.info(f"Extracted {len(cleaned_text)} characters from {file_path.name}")
            
            return cleaned_text
        
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['.', '!', '?', '\n\n']
                for ending in sentence_endings:
                    last_ending = text.rfind(ending, start, end)
                    if last_ending > start + chunk_size // 2:  # Only break if it's not too early
                        end = last_ending + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Extract metadata from document
        
        Args:
            file_path: Path to the document
            content: Document content
            
        Returns:
            Metadata dictionary
        """
        file_path = Path(file_path)
        
        metadata = {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "created_date": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat() if file_path.exists() else None,
            "modified_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None,
            "content_length": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(re.split(r'[.!?]+', content)),
            "paragraph_count": len(content.split('\n\n'))
        }
        
        return metadata

class EmbeddingGenerator:
    """Text embedding generation utilities"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_name}")
        
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            if self.model is not None:
                # Use actual model
                embedding = self.model.encode(text, convert_to_numpy=True)
            else:
                # Mock embedding for testing
                embedding = np.random.rand(384)  # 384-dim vector like MiniLM
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        try:
            if self.model is not None:
                # Use actual model with batching
                embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            else:
                # Mock embeddings for testing
                embeddings = np.random.rand(len(texts), 384)
                # Normalize each embedding
                for i in range(len(embeddings)):
                    embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def update_document_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Update embeddings for a list of documents
        
        Args:
            documents: List of documents
            
        Returns:
            Updated documents with embeddings
        """
        try:
            # Extract texts that need embeddings
            texts_to_embed = []
            doc_indices = []
            
            for i, doc in enumerate(documents):
                if not doc.embedding_generated:
                    texts_to_embed.append(doc.content)
                    doc_indices.append(i)
            
            if not texts_to_embed:
                logger.info("All documents already have embeddings")
                return documents
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts_to_embed)
            
            # Update documents
            for i, doc_idx in enumerate(doc_indices):
                documents[doc_idx].embedding = embeddings[i]
                documents[doc_idx].embedding_generated = True
            
            logger.info(f"Updated embeddings for {len(texts_to_embed)} documents")
            return documents
        
        except Exception as e:
            logger.error(f"Error updating document embeddings: {str(e)}")
            raise

class FAISSIndex:
    """FAISS index management for similarity search"""
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.document_ids = []
        self.documents = {}
        self.is_trained = False
        self.create_index()
    
    def create_index(self):
        """Create FAISS index"""
        try:
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            elif self.index_type == "ivf":
                self.index = faiss.IndexIVFFlat(
                    faiss.IndexFlatIP(self.dimension),
                    self.dimension,
                    100  # Number of clusters
                )
            elif self.index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            logger.info(f"Created FAISS index: {self.index_type} with dimension {self.dimension}")
        
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the index
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            Success status
        """
        try:
            # Filter documents with embeddings
            docs_with_embeddings = [doc for doc in documents if doc.embedding is not None]
            
            if not docs_with_embeddings:
                logger.warning("No documents with embeddings to add")
                return False
            
            # Prepare embeddings
            embeddings = np.array([doc.embedding for doc in docs_with_embeddings])
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            if self.index_type == "ivf" and not self.is_trained:
                # Train IVF index
                self.index.train(embeddings)
                self.is_trained = True
            
            self.index.add(embeddings)
            
            # Store document metadata
            for doc in docs_with_embeddings:
                self.document_ids.append(doc.id)
                self.documents[doc.id] = doc
            
            logger.info(f"Added {len(docs_with_embeddings)} documents to index")
            return True
        
        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of search results
        """
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            similarities, indices = self.index.search(query_embedding, k)
            
            # Create search results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                if similarity < threshold:
                    continue
                
                doc_id = self.document_ids[idx]
                document = self.documents[doc_id]
                
                # Create highlighted content
                highlighted_content = self.highlight_relevant_content(
                    document.content, query_embedding.flatten()
                )
                
                # Generate relevance reason
                relevance_reason = self.generate_relevance_reason(
                    document, similarity
                )
                
                result = SearchResult(
                    document=document,
                    similarity_score=float(similarity),
                    rank=i + 1,
                    highlighted_content=highlighted_content,
                    relevance_reason=relevance_reason
                )
                
                results.append(result)
            
            logger.info(f"Found {len(results)} results for search query")
            return results
        
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            raise
    
    def highlight_relevant_content(
        self,
        content: str,
        query_embedding: np.ndarray,
        max_length: int = 200
    ) -> str:
        """
        Highlight relevant content in document
        
        Args:
            content: Document content
            query_embedding: Query embedding
            max_length: Maximum length of highlighted content
            
        Returns:
            Highlighted content snippet
        """
        # Simple highlighting - take first part of content
        if len(content) <= max_length:
            return content
        
        # Find a good break point
        snippet = content[:max_length]
        last_period = snippet.rfind('.')
        last_newline = snippet.rfind('\n')
        
        if last_period > max_length * 0.7:
            snippet = snippet[:last_period + 1]
        elif last_newline > max_length * 0.7:
            snippet = snippet[:last_newline]
        
        return snippet + "..."
    
    def generate_relevance_reason(
        self,
        document: Document,
        similarity: float
    ) -> str:
        """
        Generate human-readable relevance reason
        
        Args:
            document: Document
            similarity: Similarity score
            
        Returns:
            Relevance reason string
        """
        if similarity > 0.8:
            return "Very high similarity - excellent match"
        elif similarity > 0.6:
            return "High similarity - strong relevance"
        elif similarity > 0.4:
            return "Moderate similarity - relevant content"
        else:
            return "Low similarity - minimal relevance"
    
    def save_index(self, file_path: str):
        """Save FAISS index to file"""
        try:
            # Save index
            faiss.write_index(self.index, f"{file_path}.index")
            
            # Save metadata
            metadata = {
                "document_ids": self.document_ids,
                "documents": {doc_id: {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding_generated": doc.embedding_generated
                } for doc_id, doc in self.documents.items()},
                "dimension": self.dimension,
                "index_type": self.index_type,
                "is_trained": self.is_trained
            }
            
            with open(f"{file_path}.metadata", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved index to {file_path}")
        
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, file_path: str):
        """Load FAISS index from file"""
        try:
            # Load index
            self.index = faiss.read_index(f"{file_path}.index")
            
            # Load metadata
            with open(f"{file_path}.metadata", 'rb') as f:
                metadata = pickle.load(f)
            
            self.document_ids = metadata["document_ids"]
            self.documents = {}
            
            for doc_id, doc_data in metadata["documents"].items():
                self.documents[doc_id] = Document(
                    id=doc_data["id"],
                    title=doc_data["title"],
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                    embedding_generated=doc_data["embedding_generated"]
                )
            
            self.dimension = metadata["dimension"]
            self.index_type = metadata["index_type"]
            self.is_trained = metadata["is_trained"]
            
            logger.info(f"Loaded index from {file_path}")
        
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

class SemanticSearchEngine:
    """Main semantic search engine"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.search_index = FAISSIndex()
        self.documents = []
    
    def add_document(
        self,
        file_path: str,
        title: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Document]:
        """
        Add a document to the search engine
        
        Args:
            file_path: Path to the document
            title: Document title (if None, use filename)
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            List of created documents
        """
        try:
            # Extract text
            content = self.document_processor.extract_text_from_file(file_path)
            
            # Extract metadata
            metadata = self.document_processor.extract_metadata(file_path, content)
            
            # Set title
            if title is None:
                title = Path(file_path).stem
            
            # Create document
            doc = Document(
                id=self.generate_document_id(file_path),
                title=title,
                content=content,
                metadata=metadata
            )
            
            # Chunk document if it's too long
            if len(content) > chunk_size:
                chunks = self.document_processor.chunk_text(content, chunk_size, overlap)
                documents = []
                
                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        id=f"{doc.id}_chunk_{i}",
                        title=f"{title} (Part {i+1})",
                        content=chunk,
                        metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)}
                    )
                    documents.append(chunk_doc)
                
                self.documents.extend(documents)
                logger.info(f"Added document {title} as {len(documents)} chunks")
                return documents
            
            else:
                self.documents.append(doc)
                logger.info(f"Added document {title}")
                return [doc]
        
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            raise
    
    def generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"doc_{file_hash}"
    
    def build_index(self) -> bool:
        """
        Build search index from documents
        
        Returns:
            Success status
        """
        try:
            # Generate embeddings for documents that don't have them
            self.documents = self.embedding_generator.update_document_embeddings(self.documents)
            
            # Add documents to index
            success = self.search_index.add_documents(self.documents)
            
            return success
        
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for documents similar to query
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search index
            results = self.search_index.search(query_embedding, k, threshold)
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise
    
    def search_by_metadata(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        k: int = 10
    ) -> List[SearchResult]:
        """
        Search with metadata filtering
        
        Args:
            query: Search query
            metadata_filter: Metadata filter criteria
            k: Number of results to return
            
        Returns:
            Filtered search results
        """
        try:
            # Get all results first
            all_results = self.search(query, k=len(self.documents))
            
            # Apply metadata filter
            filtered_results = []
            for result in all_results:
                if self.matches_metadata_filter(result.document.metadata, metadata_filter):
                    filtered_results.append(result)
                    if len(filtered_results) >= k:
                        break
            
            return filtered_results
        
        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            raise
    
    def matches_metadata_filter(
        self,
        metadata: Dict[str, Any],
        filter_criteria: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            
            if isinstance(value, (list, tuple)):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def save_engine(self, file_path: str):
        """Save search engine to file"""
        try:
            # Save index
            self.search_index.save_index(file_path)
            
            # Save documents
            with open(f"{file_path}.documents", 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved search engine to {file_path}")
        
        except Exception as e:
            logger.error(f"Error saving search engine: {str(e)}")
            raise
    
    def load_engine(self, file_path: str):
        """Load search engine from file"""
        try:
            # Load index
            self.search_index.load_index(file_path)
            
            # Load documents
            with open(f"{file_path}.documents", 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Loaded search engine from {file_path}")
        
        except Exception as e:
            logger.error(f"Error loading search engine: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        stats = {
            "total_documents": len(self.documents),
            "index_size": self.search_index.index.ntotal if self.search_index.index else 0,
            "index_type": self.search_index.index_type,
            "embedding_dimension": self.search_index.dimension,
            "documents_with_embeddings": sum(1 for doc in self.documents if doc.embedding_generated),
            "total_content_length": sum(len(doc.content) for doc in self.documents),
            "average_document_length": np.mean([len(doc.content) for doc in self.documents]) if self.documents else 0
        }
        
        return stats 