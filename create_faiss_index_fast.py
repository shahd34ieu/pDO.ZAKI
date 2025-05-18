
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# تحميل الملف الطبي
loader = PyPDFLoader("Clinical Pharmacology.pdf")
docs = loader.load()
docs = docs[:10]  # معالجة أول 10 صفحات فقط لتسريع الاختبار

# تقسيم النص إلى أجزاء أكبر لتقليل عدد التضمينات
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# استخدام نموذج تضمين سريع
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# إنشاء الفهرس باستخدام FAISS
faiss_index = FAISS.from_documents(split_docs, embed_model)

# حفظ الفهرس
faiss_index.save_local("clinical_pharmacology_index")
