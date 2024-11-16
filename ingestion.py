from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS



embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")


def ingest_doc():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest", encoding="utf8")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)

    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add documents {len(documents)} to FAISS")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_lagchain_documentation")

    print("***saving to local vector store done***")

if __name__ == "__main__":
    ingest_doc()