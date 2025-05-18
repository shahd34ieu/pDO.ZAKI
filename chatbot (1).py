
import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq

def load_medical_docs(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(f'Documents loaded: {len(docs)}')
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def embed_documents(docs):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embed_model)

def build_qa_system(faiss_index):
    retriever = faiss_index.as_retriever(search_type="similarity", k=4)
    llm = ChatGroq(
        api_key=("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"),
        model_name="llama3-8b-8192"
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def main():
    st.set_page_config(page_title="Pharma Chatbot (LLaMA 3 + RAG)", layout="centered")
    st.title("DRZAKI🧑‍⚕️")

    if "qa_chain" not in st.session_state:
        with st.spinner("Embedding medical documents..."):
            docs = load_medical_docs("/content/Clinical Pharmacology.pdf")
            faiss_index = embed_documents(docs)
            st.session_state.qa_chain = build_qa_system(faiss_index)

    query = st.text_input("Ask a medical question:")
    if query:
        with st.spinner("Generating answer..."):
            result = st.session_state.qa_chain.run(query)
            st.markdown(f"**Answer:** {result}")

if __name__ == "__main__":
    main()
