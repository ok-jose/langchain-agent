"""Vector store and retrieval."""

from dataclasses import dataclass, field
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from rag.loader import get_embeddings

COLLECTION_NAME = "agent_knowledge"
PERSIST_DIR = "chroma_db"


@dataclass
class KnowledgeRetriever:
    """Encapsulates Chroma vector store for RAG.

    Usage:
        retriever = KnowledgeRetriever()
        retriever.add_documents(docs)
        results = retriever.search("query")
    """

    embeddings: Optional[OllamaEmbeddings] = None
    _store: Optional[Chroma] = field(default=None, init=False, repr=False)

    @property
    def store(self) -> Chroma:
        """Lazy-initialize the Chroma vector store."""
        if self._store is None:
            self._store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings or get_embeddings(),
                persist_directory=PERSIST_DIR,
            )
        return self._store

    def add_documents(self, docs: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            docs: List of Document objects to index.

        Returns:
            List of added document IDs.
        """
        return self.store.add_documents(docs)

    def search(self, query: str, k: int = 4) -> list[Document]:
        """Search for relevant documents.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of top-k matching Document objects.
        """
        return self.store.similarity_search(query, k=k)
