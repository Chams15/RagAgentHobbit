import re
from pathlib import Path
from functools import lru_cache
import warnings
import argparse
import shutil
import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever, ParentDocumentRetriever
from langchain_community.retrievers import BM25Retriever

SOURCE_FILE = "TheHobbit.md"
DB_LOCATION = "./chroma_db"

_active_resources = {}
_source_file_mtime = None 

_ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19,
}
_NUM_TO_ROMAN = {v: k for k, v in _ROMAN_MAP.items()}


_CHAPTER_TITLES = {
    1: "AN UNEXPECTED PARTY",
    2: "ROAST MUTTON",
    3: "A SHORT REST",
    4: "OVER HILL AND UNDER HILL",
    5: "RIDDLES IN THE DARK",
    6: "OUT OF THE FRYING-PAN INTO THE FIRE",
    7: "QUEER LODGINGS",
    8: "FLIES AND SPIDERS",
    9: "BARRELS OUT OF BOND",
    10: "A WARM WELCOME",
    11: "ON THE DOORSTEP",
    12: "INSIDE INFORMATION",
    13: "NOT AT HOME",
    14: "FIRE AND WATER",
    15: "THE GATHERING OF THE CLOUDS",
    16: "A THIEF IN THE NIGHT",
    17: "THE CLOUDS BURST",
    18: "THE RETURN JOURNEY",
    19: "THE LAST STAGE",
}


_CHAPTER_PATTERN = re.compile(
    r"\b(?:chapter|ch\.?)\s+([0-9]{1,2}|[IVXLC]+)\b", re.IGNORECASE
)


def _reset_database_directory() -> None:
    """Remove the local Chroma DB directory so it can be rebuilt cleanly."""
    db_path = Path(DB_LOCATION)
    if db_path.exists():
        shutil.rmtree(db_path, ignore_errors=True)


def _create_vectorstore_with_recovery(verbose: bool = False) -> Chroma:
    """Create Chroma vectorstore and self-heal common local DB corruption issues."""
    try:
        return Chroma(
            collection_name="split_parents",
            embedding_function=get_embeddings(),
            persist_directory=DB_LOCATION,
        )
    except Exception as e:
        error_text = str(e)
        # This usually indicates broken or incompatible local DB metadata.
        if "default_tenant" in error_text or "Could not connect to tenant" in error_text:
            if verbose:
                print("\nDetected invalid local Chroma state. Rebuilding database...", flush=True)
            _reset_database_directory()
            return Chroma(
                collection_name="split_parents",
                embedding_function=get_embeddings(),
                persist_directory=DB_LOCATION,
            )
        raise


def detect_chapter(query: str) -> str | None:
    """Extract the chapter's metadata title from a user query, if a chapter is mentioned.
    
    Returns the chapter title as stored in metadata (e.g. 'AN UNEXPECTED PARTY') or None.
    """
    match = _CHAPTER_PATTERN.search(query)
    if not match:
        return None
    value = match.group(1).strip()
    
    # Convert to a chapter number
    chapter_num = None
    if value.upper() in _ROMAN_MAP:
        chapter_num = _ROMAN_MAP[value.upper()]
    else:
        try:
            chapter_num = int(value)
        except ValueError:
            return None
    
    return _CHAPTER_TITLES.get(chapter_num)


@lru_cache(maxsize=1)
def _load_raw_text():
    global _source_file_mtime
    _source_file_mtime = Path(SOURCE_FILE).stat().st_mtime
    return Path(SOURCE_FILE).read_text(encoding="utf-8")

def _check_source_file_freshness():
    """Warn if source file has been modified since cache was loaded."""
    global _source_file_mtime
    if _source_file_mtime is None:
        return 
    
    try:
        current_mtime = Path(SOURCE_FILE).stat().st_mtime
        if current_mtime > _source_file_mtime:
            warnings.warn(
                f"Source file '{SOURCE_FILE}' has been modified. "
                f"Consider restarting to refresh the embeddings.",
                UserWarning
            )
    except OSError:
        pass

@lru_cache(maxsize=1)
def get_embeddings():
    try:
        return OllamaEmbeddings(model="mxbai-embed-large")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Ollama embeddings: {str(e)}") from e

@lru_cache(maxsize=1)
def _get_parent_docs():
    """Parse and cache chapter-split documents (avoids re-parsing on every call)."""
    raw_text = _load_raw_text()
    headers_to_split_on = [("##", "Chapter")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return tuple(md_splitter.split_text(raw_text))


def get_advanced_retriever(llm=None, verbose=False):
    _check_source_file_freshness()
    
    if verbose:
        print("Loading documents...", end="", flush=True)
    
    parent_docs = list(_get_parent_docs())

    enriched_docs = []
    for doc in parent_docs:
        chapter = doc.metadata.get("Chapter", "")
        if chapter:
            enriched_content = f"[{chapter}] {doc.page_content}"
        else:
            enriched_content = doc.page_content
        enriched_docs.append(Document(
            page_content=enriched_content,
            metadata=doc.metadata.copy(),
        ))

    if verbose:
        print(" ✓\nConnecting to vector database...", end="", flush=True)
    
    vectorstore = _create_vectorstore_with_recovery(verbose=verbose)
    store = InMemoryStore()

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        collection_count = vectorstore._collection.count()
        if collection_count == 0:
            try:
                vectorstore.delete_collection()
            except Exception:
                pass
            vectorstore = _create_vectorstore_with_recovery(verbose=verbose)
    except Exception as e:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
        vectorstore = _create_vectorstore_with_recovery(verbose=verbose)
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200),
        search_kwargs={"k": 8},  # Over-fetch child matches for better parent coverage
    )

    if verbose:
        print(" ✓\nIndexing documents...", end="", flush=True)
    
    parent_retriever.add_documents(enriched_docs)

    _active_resources["vectorstore"] = vectorstore
    _active_resources["docstore"] = store
    _active_resources["parent_docs"] = parent_docs

    if verbose:
        print(" ✓\nBuilding search indices...", end="", flush=True)
    
    # BM25 should search over parent-sized chunks, not whole chapters
    # This prevents giant chapter docs from drowning out precise matches
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    bm25_chunks = []
    for doc in enriched_docs:
        bm25_chunks.extend(parent_splitter.split_documents([doc]))
    bm25_retriever = BM25Retriever.from_documents(bm25_chunks)
    bm25_retriever.k = 8

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, parent_retriever],
        weights=[0.3, 0.7]
    )

    if verbose:
        print(" ✓\nInitializing reranker...", end="", flush=True)
    
    compressor = FlashrankRerank(top_n=5)
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    if verbose:
        print(" ✓")
    
    return ChapterAwareRetriever(base_retriever=reranked_retriever)


class ChapterAwareRetriever:
    """Wraps a retriever to post-filter by chapter metadata when a chapter is mentioned in the query."""

    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def invoke(self, query: str, **kwargs) -> list[Document]:
        docs = self.base_retriever.invoke(query, **kwargs)
        chapter_title = detect_chapter(query)
        if chapter_title:
           
            filtered = [
                d for d in docs
                if d.metadata.get("Chapter", "").upper() == chapter_title.upper()
            ]
            
            return filtered if filtered else docs
        return docs

    def __or__(self, other):
        """Support `retriever | format_docs` pipe syntax."""
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self.invoke) | other

@lru_cache(maxsize=1)
def get_book_metadata() -> str:
    """Compute and return rich structural metadata about the book."""
    raw_text = _load_raw_text()

    total_words = len(raw_text.split())
    total_chars = len(raw_text)
    total_paragraphs = len([p for p in raw_text.split("\n\n") if p.strip()])

   
    chapter_pattern = re.compile(
        r"^##\s+(Chapter\s+[IVXLC]+)\s*\n+##\s+(.+)$", re.MULTILINE
    )
    chapter_matches = list(chapter_pattern.finditer(raw_text))

    
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

    vs = _active_resources.pop("vectorstore", None)
    if vs is not None:
        try:
            del vs._client
        except Exception:
            pass
        del vs

   
    store = _active_resources.pop("docstore", None)
    if store is not None:
        store.mdelete(list(store.yield_keys()))
        del store

    
    _active_resources.pop("parent_docs", None)

   
    _active_resources.clear()

    
    get_embeddings.cache_clear()
    get_book_metadata.cache_clear()

    
    gc.collect()
    print("Memory cleaned up.")


def initialize_database(source_file=None, db_location=None, force=False):
    """Initialize or reinitialize the vector database."""
    global SOURCE_FILE, DB_LOCATION
    
    if source_file:
        SOURCE_FILE = source_file
    if db_location:
        DB_LOCATION = db_location
    
    # Validate source file exists
    if not Path(SOURCE_FILE).exists():
        print(f"Error: Source file '{SOURCE_FILE}' not found.", file=sys.stderr)
        return False
    
    # Handle existing database
    if Path(DB_LOCATION).exists():
        if not force:
            response = input(f"Database at '{DB_LOCATION}' already exists. Delete and rebuild? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Initialization cancelled.")
                return False
        
        try:
            shutil.rmtree(DB_LOCATION)
            print(f"✓ Deleted existing database at {DB_LOCATION}")
        except Exception as e:
            print(f"Error deleting database: {str(e)}", file=sys.stderr)
            return False
    
    # Build the retriever (this creates and populates the database)
    try:
        print(f"Building vector database from '{SOURCE_FILE}'...")
        retriever = get_advanced_retriever(verbose=True)
        print("✓ Advanced retriever built successfully")
    except Exception as e:
        print(f"Error building retriever: {str(e)}", file=sys.stderr)
        return False
    
    # Verify the database
    try:
        vectorstore = Chroma(
            collection_name="split_parents",
            embedding_function=get_embeddings(),
            persist_directory=DB_LOCATION,
        )
        count = vectorstore._collection.count()
        print(f"✓ Database created at '{DB_LOCATION}' with {count} chunks")
        print("\nInitialization complete! You can now run the chatbot with 'python main.py'")
        return True
    except Exception as e:
        print(f"Error verifying database: {str(e)}", file=sys.stderr)
        return False


def parse_arguments():
    """Parse command-line arguments for vector database initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize vector database for The Hobbit RAG chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with defaults (TheHobbit.md -> ./chroma_db)
  python vector.py
  
  # Force rebuild without confirmation
  python vector.py --force
  
  # Use a different source file
  python vector.py --source MyBook.md
  
  # Specify custom database location
  python vector.py --db ./my_custom_db
  
  # Show database info without rebuilding
  python vector.py --info
        """
    )
    
    parser.add_argument(
        "-s", "--source",
        type=str,
        default="TheHobbit.md",
        help="Source markdown file to process (default: TheHobbit.md)"
    )
    
    parser.add_argument(
        "-d", "--db",
        type=str,
        default="./chroma_db",
        help="Database directory path (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force rebuild without confirmation if database exists"
    )
    
    parser.add_argument(
        "-i", "--info",
        action="store_true",
        help="Display book metadata and exit (no database initialization)"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="Vector DB Initializer v1.0.0"
    )
    
    return parser.parse_args()


def main_cli():
    """Entry point for console script (used by setup.py)."""
    args = parse_arguments()
    
    # Update global constants if custom paths provided
    if args.source:
        global SOURCE_FILE
        SOURCE_FILE = args.source
    if args.db:
        global DB_LOCATION
        DB_LOCATION = args.db
    
    # Info mode: just show metadata
    if args.info:
        try:
            metadata = get_book_metadata()
            print("\n" + "="*60)
            print("  Book Metadata")
            print("="*60)
            print(metadata)
            print("="*60)
            sys.exit(0)
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    # Initialize the database
    success = initialize_database(
        source_file=args.source if args.source != "TheHobbit.md" else None,
        db_location=args.db if args.db != "./chroma_db" else None,
        force=args.force
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main_cli()
