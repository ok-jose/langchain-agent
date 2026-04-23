"""Document loading and splitting."""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def load_documents(knowledge_dir: str | Path = "knowledge"):
    """Load .txt and .md files from the knowledge directory.

    Args:
        knowledge_dir: Path to the directory containing knowledge files.

    Returns:
        List of Document objects.
    """
    doc_dir = Path(knowledge_dir)
    if not doc_dir.exists():
        return []

    docs = []
    for ext in ("*.txt", "*.md"):
        for file_path in doc_dir.glob(ext):
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
    return docs


def split_documents(docs, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        docs: List of Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of split Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def get_embeddings(model: str = "bge-m3:latest", base_url: str | None = None):
    """Create OllamaEmbeddings instance.

    Args:
        model: Embedding model name.
        base_url: Ollama base URL. Defaults to localhost.

    Returns:
        OllamaEmbeddings instance.
    """
    url = base_url or DEFAULT_OLLAMA_BASE_URL
    return OllamaEmbeddings(model=model, base_url=url)
