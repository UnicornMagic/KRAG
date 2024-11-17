import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Directory for persisted vector database
VECTOR_DB_DIR = "chromadb_store"

# Initialize the OpenAI embeddings
# Ensure your OpenAI API key is set as an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Set your OpenAI API key as an environment variable!")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load the vector store
@st.cache_resource
def load_vectorstore(vector_db_dir):
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)
    return vectorstore

# Query the vector store
def query_vectorstore(vectorstore, query, top_k=3):
    results = vectorstore.similarity_search(query, k=top_k)
    return results

# Streamlit app
def main():
    st.title("Knowledge Retrieval App")
    st.write("Ask a question about internal policies, tools, or ways of working!")

    # Load the vector store
    vectorstore = load_vectorstore(VECTOR_DB_DIR)

    # User input
    query = st.text_input("Enter your query:", "")
    if query:
        with st.spinner("Searching for answers..."):
            results = query_vectorstore(vectorstore, query)
        st.success("Results:")
        for i, result in enumerate(results, start=1):
            st.markdown(f"### Result {i}")
            st.write(result.page_content)

if __name__ == "__main__":
    main()