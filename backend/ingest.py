import os
import uuid
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTORSTORES_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstores")
os.makedirs(VECTORSTORES_DIR, exist_ok=True)

_EMBEDDINGS = OpenAIEmbeddings()


def load_pdf_file(path: str) -> List[Document]:
    """
    Load a PDF and return a list of LangChain Document objects.
    Uses PyPDFLoader for text PDFs and falls back to UnstructuredPDFLoader for scanned PDFs.
    """
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
    except Exception:
        # fallback if PyPDFLoader cannot parse (e.g., scanned image PDF)
        loader = UnstructuredPDFLoader(path)
        docs = loader.load()
    return docs


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks suitable for embedding and retrieval.
    Also attaches provenance metadata (source filename and chunk index).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)

    # Add provenance metadata if not present
    enhanced = []
    for i, d in enumerate(split_docs):
        meta = dict(d.metadata or {})
        # keep original filename if available, otherwise set unknown
        meta.setdefault("source", meta.get("source", getattr(d, "source", "unknown")))
        meta["chunk"] = i
        enhanced.append(Document(page_content=d.page_content, metadata=meta))
    return enhanced


def upsert_documents_to_faiss(
    documents: List[Document], index_name: str | None = None
) -> str:
    """
    Create a FAISS index from documents and save it locally.
    Returns path to saved index folder.
    """
    index_name = index_name or str(uuid.uuid4())
    out_dir = os.path.join(VECTORSTORES_DIR, index_name)
    os.makedirs(out_dir, exist_ok=True)

    # Build vectorstore and save locally
    db = FAISS.from_documents(documents, _EMBEDDINGS)
    db.save_local(out_dir)

    return out_dir


def ingest_pdf(path: str, index_name: str | None = None) -> Tuple[str, int]:
    """
    Full pipeline: load -> split -> embed & save.
    Returns (index_path, number_of_chunks)
    """
    docs = load_pdf_file(path)
    chunks = split_documents(docs)
    index_path = upsert_documents_to_faiss(chunks, index_name=index_name)
    return index_path, len(chunks)
