import re
from pathlib import Path
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever, ParentDocumentRetriever     
from langchain_community.retrievers import BM25Retriever

SOURCE_FILE = "TheHobbit.md"
DB_LOCATION = "./chroma_db"

# Track objects that need cleanup
_active_resources = {}


def _load_raw_text():
    return Path(SOURCE_FILE).read_text(encoding="utf-8")

@lru_cache(maxsize=1)
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

def get_advanced_retriever(llm=None):
    raw_text = _load_raw_text()
    
    # 1. Split by Chapter Headers first
    headers_to_split_on = [("##", "Chapter")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = md_splitter.split_text(raw_text)

    # 2. Setup Vectorstore and Storage for Parent-Child relationship
    vectorstore = Chroma(
        collection_name="split_parents",
        embedding_function=get_embeddings(),
        persist_directory=DB_LOCATION
    )
    store = InMemoryStore()
    
    # Child chunks for precise retrieval, Parents for context
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200),
    )
    
    parent_retriever.add_documents(parent_docs)

    # Track resources for cleanup
    _active_resources["vectorstore"] = vectorstore
    _active_resources["docstore"] = store
    _active_resources["parent_docs"] = parent_docs

    # 3. Hybrid Search Setup (BM25 + Vector)
    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = 4

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, parent_retriever],
        weights=[0.3, 0.7] # Heavily weight the parent/vector retrieval
    )

    # 4. Reranking for final precision
    compressor = FlashrankRerank(top_n=5)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

@lru_cache(maxsize=1)
def get_book_metadata() -> str:
    """Compute and return rich structural metadata about the book."""
    raw_text = _load_raw_text()

    total_words = len(raw_text.split())
    total_chars = len(raw_text)
    total_paragraphs = len([p for p in raw_text.split("\n\n") if p.strip()])

    # Parse chapter headers and their subtitles (e.g. "## Chapter I" followed by "## AN UNEXPECTED PARTY")
    chapter_pattern = re.compile(
        r"^##\s+(Chapter\s+[IVXLC]+)\s*\n+##\s+(.+)$", re.MULTILINE
    )
    chapter_matches = list(chapter_pattern.finditer(raw_text))

    # Compute per-chapter stats
    chapter_details = []
    for i, match in enumerate(chapter_matches):
        chapter_num = match.group(1).strip()
        chapter_name = match.group(2).strip().title()
        content_start = match.end()
        content_end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(raw_text)
        chapter_text = raw_text[content_start:content_end].strip()
        word_count = len(chapter_text.split())
        paragraph_count = len([p for p in chapter_text.split("\n\n") if p.strip()])
        chapter_details.append(
            f"  {chapter_num} — \"{chapter_name}\": {word_count:,} words, {paragraph_count} paragraphs"
        )

    meta = [
        "Book Title: The Hobbit, or There and Back Again",
        "Author: J.R.R. Tolkien",
        f"Source File: {SOURCE_FILE}",
        "",
        f"Total Chapters: {len(chapter_matches)}",
        f"Total Words: {total_words:,}",
        f"Total Characters: {total_chars:,}",
        f"Total Paragraphs: {total_paragraphs:,}",
        "",
        "Chapter Breakdown:",
        *chapter_details,
    ]

    return "\n".join(meta)


def cleanup():
    """Release all in-memory resources: vectorstore client, docstore, caches."""
    import gc

    # Close the Chroma vectorstore client
    vs = _active_resources.pop("vectorstore", None)
    if vs is not None:
        try:
            vs.delete_collection()
        except Exception:
            pass
        try:
            del vs._client
        except Exception:
            pass
        del vs

    # Clear the in-memory docstore
    store = _active_resources.pop("docstore", None)
    if store is not None:
        store.mdelete(list(store.yield_keys()))
        del store

    # Clear cached parent docs
    _active_resources.pop("parent_docs", None)

    # Clear any remaining tracked resources
    _active_resources.clear()

    # Clear lru_caches
    get_embeddings.cache_clear()
    get_book_metadata.cache_clear()

    # Force garbage collection
    gc.collect()
    print("Memory cleaned up.")


if __name__ == "__main__":
    
    import shutil
    if Path(DB_LOCATION).exists():
        shutil.rmtree(DB_LOCATION)
        print(f"Deleted existing DB at {DB_LOCATION}")
    
    # Build the retriever (which populates the vectorstore)
    retriever = get_advanced_retriever()
    print(f"Advanced retriever built successfully.")
    
    # Check the vectorstore
    vectorstore = Chroma(
        collection_name="split_parents",
        embedding_function=get_embeddings(),
        persist_directory=DB_LOCATION,
    )
    count = vectorstore._collection.count()
    print(f"Chroma DB at {DB_LOCATION} with {count} chunks.")

