import atexit
import signal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from vector import get_advanced_retriever, get_book_metadata, cleanup

def format_docs(docs):
    # Ensure we don't duplicate context if parent retrieval returns the same large doc
    unique_contents = list(set(doc.page_content for doc in docs))
    return "\n\n---\n\n".join(unique_contents)

def build_rag_chain():
    model = OllamaLLM(
        model="llama3.1",
        temperature=0.1,
        num_ctx=12000, # Increased to handle larger parent chunks
    )
    
    retriever = get_advanced_retriever(llm=model)
    metadata = get_book_metadata()

    template = """
You are a scholar of Middle-earth specializing in 'The Hobbit'. 
Answer the question using ONLY the Metadata and Context provided.

Rules:
1. If the answer isn't in the context, state that clearly but offer any related facts found.
2. Use a formal yet engaging tone.
3. Reference specific chapters (e.g., 'As seen in Chapter II...') based on the context headers.

Metadata:
{metadata}

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough(), "metadata": lambda _: metadata}
        | prompt 
        | model
    )

def main():
    # Register cleanup on normal exit and Ctrl+C
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda sig, frame: exit(0))  # atexit runs on exit()

    print("Initializing Hobbit Scholar RAG...")
    chain = build_rag_chain()
    
    while True:
        query = input("\nWhat would you like to know about Bilbo's journey? (exit to quit): ")
        if query.lower() in ["exit", "quit"]: break
        
        print("\nConsulting the archives...")
        print(f"\nResponse: {chain.invoke(query)}")

if __name__ == "__main__":
    main()