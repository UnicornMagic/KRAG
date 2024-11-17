import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directory containing the documents
DOCUMENTS_DIR = "data"

# Initialize the OpenAI embeddings
# Make sure you have your OpenAI API key set as an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set your OpenAI API key as an environment variable!")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load documents
def load_documents(documents_dir):
    documents = []
    for filename in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, filename)
        if os.path.isfile(file_path):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

# Ingest and vectorize documents
def vectorize_documents(documents_dir, embeddings):
    documents = load_documents(documents_dir)
    print(f"Loaded {len(documents)} documents.")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print(f"Split documents into {len(docs)} chunks.")

    # Use ChromaDB as the vector store
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="chromadb_store")
    vectorstore.persist()
    print("Vectorstore created and persisted!")

if __name__ == "__main__":
    vectorize_documents(DOCUMENTS_DIR, embeddings)