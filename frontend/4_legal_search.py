"""
Legal Search - Streamlit page for semantic search over legal documents
Upload contracts/opinions, generate embeddings, and perform semantic search
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import requests
import json
import time
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Legal Search - ML Playground",
    page_icon="⚖️",
    layout="wide"
)

def get_sample_legal_documents() -> List[Dict[str, Any]]:
    """Get sample legal documents for demonstration"""
    return [
        {
            "title": "Employment Contract Template",
            "content": "This employment agreement is made between [Company Name] and [Employee Name]...",
            "type": "contract",
            "date": "2024-01-15"
        },
        {
            "title": "Non-Disclosure Agreement",
            "content": "This Non-Disclosure Agreement (NDA) is entered into by and between...",
            "type": "contract",
            "date": "2024-01-10"
        },
        {
            "title": "Supreme Court Opinion - Data Privacy",
            "content": "The court finds that the right to privacy extends to digital communications...",
            "type": "opinion",
            "date": "2023-12-20"
        },
        {
            "title": "Patent Application Guidelines",
            "content": "When filing a patent application, the following requirements must be met...",
            "type": "guideline",
            "date": "2024-01-05"
        }
    ]

def create_document_uploader() -> List[Dict[str, Any]]:
    """Create interface for uploading legal documents"""
    st.subheader("📁 Upload Legal Documents")
    
    uploaded_files = st.file_uploader(
        "Choose legal documents (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload your legal documents for semantic search"
    )
    
    documents = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # TODO: Implement actual file parsing
            # For now, create mock document structure
            doc = {
                "title": uploaded_file.name,
                "content": f"Content from {uploaded_file.name} would be extracted here...",
                "type": "uploaded",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "filename": uploaded_file.name
            }
            documents.append(doc)
            st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Add sample documents option
    if st.button("📚 Load Sample Documents"):
        sample_docs = get_sample_legal_documents()
        documents.extend(sample_docs)
        st.success(f"✅ Loaded {len(sample_docs)} sample documents")
    
    return documents

def simulate_embedding_generation(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate embedding generation for documents"""
    # TODO: Replace with actual embedding generation
    for doc in documents:
        # Simulate embedding vector
        doc["embedding"] = np.random.rand(768).tolist()  # 768-dim vector
        doc["embedding_generated"] = True
    
    return documents

def create_search_interface(documents: List[Dict[str, Any]]) -> str:
    """Create the search interface"""
    st.subheader("🔍 Semantic Search")
    
    # Search query input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., data privacy rights, employment termination, patent requirements",
        help="Enter a natural language query to search through your documents"
    )
    
    # Search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_type = st.selectbox(
            "Search Type:",
            ["Semantic", "Keyword", "Hybrid"],
            help="Choose the type of search to perform"
        )
    
    with col2:
        top_k = st.slider("Number of Results", 1, 20, 5)
    
    with col3:
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, step=0.1)
    
    return query, search_type, top_k, similarity_threshold

def simulate_semantic_search(query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Simulate semantic search results"""
    # TODO: Replace with actual FAISS search
    results = []
    
    for i, doc in enumerate(documents):
        # Simulate similarity score
        similarity = np.random.uniform(0.3, 0.95)
        
        result = {
            "document": doc,
            "similarity_score": similarity,
            "rank": i + 1,
            "highlighted_content": f"...{doc['content'][:100]}...",
            "relevance_reason": f"Document contains relevant terms and concepts related to '{query}'"
        }
        results.append(result)
    
    # Sort by similarity score and return top_k
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]

def display_search_results(results: List[Dict[str, Any]], query: str):
    """Display search results with visualizations"""
    st.subheader(f"📋 Search Results for: '{query}'")
    
    if not results:
        st.info("No results found. Try adjusting your search query or parameters.")
        return
    
    # Results overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Results", len(results))
    with col2:
        avg_similarity = np.mean([r["similarity_score"] for r in results])
        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    with col3:
        st.metric("Top Score", f"{results[0]['similarity_score']:.3f}")
    
    # Similarity distribution
    fig = px.histogram(
        x=[r["similarity_score"] for r in results],
        title="Similarity Score Distribution",
        labels={"x": "Similarity Score", "y": "Count"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual results
    for i, result in enumerate(results):
        with st.expander(f"#{result['rank']} - {result['document']['title']} (Score: {result['similarity_score']:.3f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Content:** {result['highlighted_content']}")
                st.write(f"**Type:** {result['document']['type']}")
                st.write(f"**Date:** {result['document']['date']}")
                st.write(f"**Relevance:** {result['relevance_reason']}")
            
            with col2:
                # Similarity score visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["similarity_score"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Similarity"},
                    gauge={'axis': {'range': [0, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                    {'range': [0.5, 0.8], 'color': "yellow"},
                                    {'range': [0.8, 1], 'color': "green"}]}
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)

def create_document_analytics(documents: List[Dict[str, Any]]):
    """Create analytics for uploaded documents"""
    st.subheader("📊 Document Analytics")
    
    if not documents:
        st.info("No documents uploaded yet.")
        return
    
    # Document type distribution
    doc_types = [doc["type"] for doc in documents]
    type_counts = pd.Series(doc_types).value_counts()
    
    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Document Type Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Document timeline
    dates = [doc["date"] for doc in documents]
    date_counts = pd.Series(dates).value_counts().sort_index()
    
    fig = px.line(
        x=date_counts.index,
        y=date_counts.values,
        title="Document Timeline",
        labels={"x": "Date", "y": "Number of Documents"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Document statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(documents))
    with col2:
        st.metric("Document Types", len(set(doc_types)))
    with col3:
        avg_length = np.mean([len(doc["content"]) for doc in documents])
        st.metric("Avg Content Length", f"{avg_length:.0f} chars")
    with col4:
        st.metric("Date Range", f"{min(dates)} to {max(dates)}")

def main():
    """Main function for the Legal Search page"""
    st.title("⚖️ Legal Search")
    st.markdown("Upload legal documents and perform semantic search to find relevant information.")
    
    # Document upload
    documents = create_document_uploader()
    
    if documents:
        # Generate embeddings
        if st.button("🔧 Generate Embeddings", type="primary"):
            with st.spinner("Generating embeddings..."):
                documents = simulate_embedding_generation(documents)
            st.success(f"✅ Generated embeddings for {len(documents)} documents")
        
        # Check if embeddings are ready
        embeddings_ready = all(doc.get("embedding_generated", False) for doc in documents)
        
        if embeddings_ready:
            # Search interface
            query, search_type, top_k, similarity_threshold = create_search_interface(documents)
            
            if query:
                # Perform search
                if st.button("🔍 Search Documents", type="primary"):
                    with st.spinner("Searching documents..."):
                        results = simulate_semantic_search(query, documents, top_k)
                    
                    # Filter by similarity threshold
                    filtered_results = [r for r in results if r["similarity_score"] >= similarity_threshold]
                    
                    if filtered_results:
                        display_search_results(filtered_results, query)
                    else:
                        st.warning(f"No results above similarity threshold {similarity_threshold}")
                
                # Document analytics
                create_document_analytics(documents)
            
            # Document management
            st.subheader("📚 Document Management")
            
            # Show uploaded documents
            with st.expander("View All Documents"):
                for i, doc in enumerate(documents):
                    st.write(f"**{i+1}. {doc['title']}** ({doc['type']})")
                    st.write(f"Date: {doc['date']}")
                    st.write(f"Content preview: {doc['content'][:100]}...")
                    st.divider()
            
            # Export results
            if st.button("📤 Export Search Results"):
                st.info("Export functionality coming soon!")
        
        else:
            st.info("👆 Please generate embeddings to enable search functionality")
    
    else:
        st.info("👆 Please upload documents to begin semantic search")

if __name__ == "__main__":
    main() 