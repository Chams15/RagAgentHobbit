from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM

from vector import get_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    model = OllamaLLM(model="llama3.2", temperature=0.7)
    retriever = get_retriever()

    template = """
You are an expert on the story The Hobbit. Use only the provided context to answer the question.
If the answer is not in the context, say you do not know.

Context:
{context}

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    return {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | prompt | model


def main():
    rag_chain = build_rag_chain()

    while True:
        print("----------------------------------------------------------------------------------------------------")
        question = input("Ask a question about The Hobbit (or 'exit' to quit): ")
        if question.lower() == "exit":
            break
        response = rag_chain.invoke(question)
        print("Response:", response)


if __name__ == "__main__":
    main()