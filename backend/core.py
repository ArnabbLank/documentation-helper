from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


def run_llm(querry: str):
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
    vectorstore = FAISS.load_local(
        "faiss_index_lagchain_documentation", embeddings, allow_dangerous_deserialization=True
    )
    chat = OllamaLLM(model="llama3", temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        vectorstore.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": querry})
    new_result = {
        "querry": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }

    return new_result


if __name__ == "__main__":
    res = run_llm(querry="What is a LangChain Chain?")
    print(res["result"])
