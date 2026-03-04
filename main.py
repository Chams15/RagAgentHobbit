import atexit
import signal
import time
from collections import deque
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama.llms import OllamaLLM
from langchain_core.exceptions import LangChainException
from vector import get_advanced_retriever, get_book_metadata, cleanup

# Keep last 4 Q&A turns for conversation continuity
MAX_HISTORY_TURNS = 4
chat_history: deque[tuple[str, str]] = deque(maxlen=MAX_HISTORY_TURNS)


MAX_HISTORY_ANSWER_CHARS = 500  # Truncate long past answers to save context window


def format_history() -> str:
    """Format conversation history into a readable string, truncating long answers."""
    if not chat_history:
        return "No previous conversation."
    lines = []
    for q, a in chat_history:
        lines.append(f"User: {q}")
        truncated = a[:MAX_HISTORY_ANSWER_CHARS] + "..." if len(a) > MAX_HISTORY_ANSWER_CHARS else a
        lines.append(f"Assistant: {truncated}")
    return "\n".join(lines)

def format_docs(docs):
    # Deduplicate while preserving reranker relevance order
    unique_contents = list(dict.fromkeys(doc.page_content for doc in docs))
    return "\n\n---\n\n".join(unique_contents)

def build_rag_chain():
    try:
        model = OllamaLLM(
            model="llama3.1",
            temperature=0.1,
            num_ctx=12000,
        )
        # Test connection to Ollama
        _ = model.invoke("test")
    except Exception as e:
        print(f"\nError: Failed to connect to Ollama. Make sure Ollama is running.")
        print(f"Details: {str(e)}")
        sys.exit(1)
    
    try:
        retriever = get_advanced_retriever(llm=model)
        metadata = get_book_metadata()
    except Exception as e:
        print(f"\nError: Failed to initialize retriever. {str(e)}")
        sys.exit(1)

    template = """
You are a scholar of Middle-earth specializing in 'The Hobbit'. 
Answer the question using ONLY the Metadata, Context, and Conversation History provided.

Rules:
1. If the answer isn't in the context, state that clearly but offer any related facts found.
2. Use a formal yet engaging tone.
3. Reference specific chapters (e.g., 'As seen in Chapter II...') based on the context headers.
4. Use the Conversation History to understand follow-up questions (e.g., "tell me more" or "what happened next").

Metadata:
{metadata}

Conversation History:
{history}

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    return (
        {
            "context": (lambda x: x["question"]) | RunnableLambda(retriever.invoke) | format_docs,
            "question": lambda x: x["question"],
            "metadata": lambda _: metadata,
            "history": lambda _: format_history(),
        }
        | prompt 
        | model
    )

def main():
    
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda sig, frame: exit(0))  

    print("Initializing Hobbit Scholar RAG...")
    chain = build_rag_chain()
    
    while True:
        try:
            query = input("\nWhat would you like to know about Bilbo's journey? (exit to quit): ").strip()
            
            # Validate input
            if not query:
                print("Please enter a question.")
                continue
            
            if query.lower() in ["exit", "quit"]:
                break
            
            if len(query) > 1000:
                print("Question too long. Please keep it under 1000 characters.")
                continue
            
            print("\nConsulting the archives...\n")
            start_time = time.perf_counter()
            print("Response: ", end="", flush=True)
            response_chunks = []
            
            try:
                for chunk in chain.stream({"question": query}):
                    print(chunk, end="", flush=True)
                    response_chunks.append(chunk)
            except Exception as e:
                print(f"\nError during generation: {str(e)}")
                continue
            
            elapsed = time.perf_counter() - start_time
            print(f"\n\n[{elapsed:.1f}s]")
            
            # Save to chat history
            full_response = "".join(response_chunks)
            chat_history.append((query, full_response))
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            continue
        
        # Save this turn to conversation history
        chat_history.append((query, "".join(response_chunks)))

if __name__ == "__main__":
    main()