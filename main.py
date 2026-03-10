import atexit
import signal
import time
from collections import deque
import sys
import argparse

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

# Global cache for lazy-loaded retriever
_cached_retriever = None
_cached_metadata = None

def get_or_create_retriever(verbose=True):
    """Lazy-load the retriever only when needed."""
    global _cached_retriever, _cached_metadata
    
    if _cached_retriever is None:
        if verbose:
            print("Loading knowledge base...")
        try:
            _cached_retriever = get_advanced_retriever(verbose=verbose)
            _cached_metadata = get_book_metadata()
            if verbose:
                print("Knowledge base ready!\n")
        except Exception as e:
            print(f"\n\nError: Failed to load retriever. {str(e)}")
            print("Try rebuilding the local vector DB with: hobbit-init --force")
            sys.exit(1)
    
    return _cached_retriever, _cached_metadata

def build_rag_chain(model_name="llama3.1", temperature=0.1, num_ctx=12000, test_connection=False):
    try:
        model = OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        # Skip connection test at startup for faster launch
        if test_connection:
            _ = model.invoke("test")
    except Exception as e:
        print(f"\nError: Failed to initialize Ollama model.")
        print(f"Details: {str(e)}")
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

    # Lazy-load retriever on first use
    def get_context_with_lazy_load(inputs):
        retriever, _ = get_or_create_retriever(verbose=True)
        docs = retriever.invoke(inputs["question"])
        return format_docs(docs)
    
    def get_metadata_with_lazy_load(_):
        _, metadata = get_or_create_retriever(verbose=False)
        return metadata

    return (
        {
            "context": RunnableLambda(get_context_with_lazy_load),
            "question": lambda x: x["question"],
            "metadata": RunnableLambda(get_metadata_with_lazy_load),
            "history": lambda _: format_history(),
        }
        | prompt 
        | model
    )

def interactive_mode(chain):
    """Run the chatbot in interactive mode with continuous Q&A."""
    print("\n" + "="*60)
    print("  The Hobbit Scholar - Interactive Mode")
    print("="*60)
    print("Ask questions about Bilbo's journey. Type 'exit' or 'quit' to leave.\n")
    
    while True:
        try:
            query = input("Question: ").strip()
            
            # Validate input
            if not query:
                print("Please enter a question.")
                continue
            
            if query.lower() in ["exit", "quit"]:
                print("\nFarewell, dear scholar! May your travels be as memorable as Bilbo's.")
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
            print("\n\nExiting gracefully...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            continue

def single_query_mode(chain, question, verbose=False):
    """Answer a single question and exit."""
    if not question or not question.strip():
        print("Error: Empty question provided.")
        sys.exit(1)
    
    if len(question) > 1000:
        print("Error: Question too long. Please keep it under 1000 characters.")
        sys.exit(1)
    
    start_time = time.perf_counter()
    
    if verbose:
        print(f"\nQuestion: {question}\n")
        print("Consulting the archives...\n")
        print("Response: ", end="", flush=True)
    
    try:
        response_chunks = []
        for chunk in chain.stream({"question": question}):
            if verbose:
                print(chunk, end="", flush=True)
            response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        
        if verbose:
            elapsed = time.perf_counter() - start_time
            print(f"\n\n[{elapsed:.1f}s]")
        else:
            print(full_response)
            
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="The Hobbit Scholar - RAG-powered Chatbot for answering questions about The Hobbit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python main.py
  
  # Ask a single question
  python main.py -q "Who is Bilbo Baggins?"
  
  # Use a different Ollama model
  python main.py --model llama3.2
  
  # Adjust creativity
  python main.py -q "Describe Smaug" --temperature 0.5
  
  # Non-verbose output (just the answer)
  python main.py -q "What is Sting?" --quiet
        """
    )
    
    parser.add_argument(
        "-q", "--query", "--question",
        type=str,
        help="Ask a single question and exit (non-interactive mode)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="llama3.1",
        help="Ollama model to use (default: llama3.1)"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.1,
        help="Model temperature for creativity (0.0-1.0, default: 0.1)"
    )
    
    parser.add_argument(
        "-c", "--context-size",
        type=int,
        default=12000,
        help="Model context window size (default: 12000)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only show answer in single-query mode)"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="The Hobbit Scholar v1.0.0"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI application."""
    args = parse_arguments()
    
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda sig, frame: exit(0))
    
    # Fast initialization - build chain without loading retriever yet
    if not args.quiet:
        print("Initializing Hobbit Scholar...")
    
    try:
        chain = build_rag_chain(
            model_name=args.model,
            temperature=args.temperature,
            num_ctx=args.context_size,
            test_connection=False  # Skip connection test for faster startup
        )
        
        if not args.quiet:
            print("Ready! (Knowledge base will load on first query)\n")
    except Exception as e:
        print(f"Failed to initialize: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Run in appropriate mode
    if args.query:
        single_query_mode(chain, args.query, verbose=not args.quiet)
    else:
        interactive_mode(chain)

if __name__ == "__main__":
    main()