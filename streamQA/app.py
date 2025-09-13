import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Streamlit Title
st.title("Chat with multiple PDFs using ObjectBox and GROQ")

# LLM setup
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    <context>
    Question: {input}
    """
)

from langchain_community.embeddings import HuggingFaceEmbeddings

# Function for embedding PDFs into vector DB
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Load PDFs
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=384
        )

# Button to trigger embeddings
if st.button("Documents Embedding"):
    vector_embedding()
    st.success("✅ ObjectBox Database is ready!")

# Text input for questions
input_prompt = st.text_input("Enter Your Question From Documents")

# Run retrieval only if vectors exist
if input_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please click 'Documents Embedding' first to build the database.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': input_prompt})
        st.write("### Answer:")
        st.write(response['answer'])
        st.write(f"⏱️ Response time: {time.process_time() - start:.2f} seconds")

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
