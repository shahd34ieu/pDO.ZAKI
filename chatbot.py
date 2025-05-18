
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
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def analyze_blood_pressure(bp_value):
    try:
        bp = int(bp_value)
        if bp < 90:
            return "الضغط منخفض"
        elif 90 <= bp <= 120:
            return "الضغط طبيعي"
        else:
            return "الضغط مرتفع"
    except ValueError:
        return "الرجاء إدخال رقم صحيح لقيمة الضغط"

def main():
    st.set_page_config(page_title="الدكتور الذكي (LLaMA 3 + RAG)", layout="centered")
    st.title("الدكتور الذكي")

    if "qa_chain" not in st.session_state:
        with st.spinner("يتم تحميل المستندات الطبية..."):
            docs = load_medical_docs("/content/Clinical Pharmacology.pdf")
            faiss_index = embed_documents(docs)
            st.session_state.qa_chain = build_qa_system(faiss_index)

    st.header("تحليل ضغط الدم:")
    bp_input = st.text_input("ادخل قيمة ضغطك:")
    if bp_input:
        result = analyze_blood_pressure(bp_input)
        st.success(f"تحليل الضغط: {result}")

    st.header("اسأل الدكتور الذكي:")
    query = st.text_input("اكتب سؤالك الطبي هنا:")
    if query:
        with st.spinner("يتم توليد الإجابة..."):
            result = st.session_state.qa_chain.run(query)
            st.markdown(f"**الإجابة:** {result}")

if __name__ == "__main__":
    main()
