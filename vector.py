from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCE_FILE = "TheHobbit.md"
DB_LOCATION = "./chroma_langchain_db"


def _load_documents() -> list:
    raw_text = Path(SOURCE_FILE).read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.create_documents([raw_text], metadatas=[{"source": SOURCE_FILE}])


@lru_cache(maxsize=1)
def _get_embeddings():
    return embeddings.OllamaEmbeddings(model="mxbai-embed-large")


def get_vectorstore() -> Chroma:
    embeddings_model = _get_embeddings()

    if not Path(DB_LOCATION).exists():
        documents = _load_documents()
        Chroma.from_documents(documents, embeddings_model, persist_directory=DB_LOCATION)

    return Chroma(persist_directory=DB_LOCATION, embedding_function=embeddings_model)


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    store = get_vectorstore()
    count = store._collection.count()  # type: ignore[attr-defined]
    print(f"Chroma DB ready at {DB_LOCATION} with {count} chunks.")

