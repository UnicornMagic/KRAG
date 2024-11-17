import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Directory for persisted vector database
VECTOR_DB_DIR = "chromadb_store"

# Initialize the OpenAI embeddings
# Ensure your OpenAI API key is set as an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set your OpenAI API key as an environment variable!")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load the vector store
def load_vectorstore(vector_db_dir):
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)
    return vectorstore

# Query the vector store
def query_vectorstore(vectorstore, query, top_k=3):
    results = vectorstore.similarity_search(query, k=top_k)
    return results

if __name__ == "__main__":
    # Load the vector store
    vectorstore = load_vectorstore(VECTOR_DB_DIR)
    print("Vector store loaded.")

    # Accept user input
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Search the vector store
        results = query_vectorstore(vectorstore, query)
        print("\nTop Results:")
        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}:")
            print(result.page_content)