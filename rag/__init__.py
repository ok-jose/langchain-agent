"""RAG package - document loading and retrieval."""

from rag.loader import get_embeddings, load_documents, split_documents
from rag.retriever import KnowledgeRetriever

__all__ = ["load_documents", "split_documents", "get_embeddings", "KnowledgeRetriever"]
