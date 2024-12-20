import os
import csv
from datetime import datetime
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Directory for persisted vector database
VECTOR_DB_DIR = "chromadb_store"

# Initialize the OpenAI embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Set your OpenAI API key as an environment variable!")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load the vector store
@st.cache_resource
def load_vectorstore(vector_db_dir):
    return Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)

# Query the vector store
def query_vectorstore(vectorstore, query, top_k=3):
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return results
    except Exception as e:
        st.error(f"Error during query: {e}")
        return []

# Save feedback
def save_feedback(query, result, rating):
    feedback_file = "feedback.csv"
    with open(feedback_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), query, result, rating])

# Process uploaded file and add to vectorstore
def process_uploaded_file(uploaded_file, vectorstore, embeddings):
    # Save the file locally
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Use the appropriate loader based on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith((".txt", ".md")):
        loader = TextLoader(file_path)
    else:
        os.remove(file_path)  # Clean up the temporary file
        raise ValueError("Unsupported file type. Please upload a .txt, .md, or .pdf file.")
    
    # Load and split the document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Add new documents to the vector store
    vectorstore.add_documents(docs, embeddings=embeddings)
    vectorstore.persist()  # Save the updated vector store
    
    # Clean up the temporary file
    os.remove(file_path)

    return len(docs)

# Initialize the LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

#Streamlit app 
def main():
    st.title("Knowledge Retrieval App")
    st.write("Ask a question about internal policies, tools, or ways of working!")

    # Load the vector store
    vectorstore = load_vectorstore(VECTOR_DB_DIR)

    # Create a retriever from the vectorstore
    # Set top_k to reduce the number of retrieved chunks if desired
    query = st.text_input("Enter your query:", "")
    top_k = st.slider("Number of results to display:", 1, 10, 3)

    # Create a RetrievalQA chain that uses the vectorstore as a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",    # "stuff" simply puts all retrieved text into context
        retriever=retriever
    )

    if query:
        with st.spinner("Searching for answers..."):
            # Instead of just retrieving chunks, we now get a final answer from the LLM
            answer = qa_chain.run(query)

        if answer:
            st.success("Answer:")
            st.write(answer)

            # Optional: If you still want to show the raw chunks
            # results = retriever.get_relevant_documents(query)
            # for i, doc in enumerate(results, start=1):
            #     st.markdown(f"### Source {i}")
            #     st.write(doc.page_content)

            # Feedback buttons
            st.write("Was this result helpful?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes"):
                    save_feedback(query, answer, "Yes")
                    st.success("Feedback saved!")
            with col2:
                if st.button("👎 No"):
                    save_feedback(query, answer, "No")
                    st.warning("Feedback saved!")
    
    # File upload
    st.write("---")
    st.write("Upload new documents:")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "pdf"])
    if uploaded_file:
        with st.spinner("Processing file..."):
            num_chunks = process_uploaded_file(uploaded_file, vectorstore, embeddings)
        st.success(f"File uploaded and processed successfully! Added {num_chunks} document chunks.")

if __name__ == "__main__":
    main()