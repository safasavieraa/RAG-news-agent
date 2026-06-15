import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Load API key dari file .env
load_dotenv()

def load_embedding_model():
    """Load embedding model untuk similarity search"""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embedding model siap!")
    return embeddings

def load_vector_store(index_path, embeddings):
    """Load FAISS index dari disk"""
    print("Loading vector database...")
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector database siap!")
    return vector_store

def load_llm():
    """Load Groq LLM"""
    print("Menghubungkan ke Groq LLM...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    print("Groq LLM siap!")
    return llm

def format_docs(docs):
    """Gabungkan artikel jadi satu teks konteks"""
    result = ""
    for i, doc in enumerate(docs):
        result += f"[Artikel {i+1}]\n"
        result += f"Judul  : {doc.metadata['judul']}\n"
        result += f"Sumber : {doc.metadata['source']}\n"
        result += f"Waktu  : {doc.metadata['waktu']}\n"
        result += f"Isi    : {doc.page_content[:300]}\n\n"
    return result

def build_rag_chain(vector_store, llm):
    """Bangun RAG chain"""
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    prompt_template = """Kamu adalah asisten berita Indonesia.
Gunakan artikel berita berikut untuk menjawab pertanyaan pengguna.
Jawab langsung berdasarkan isi artikel — jangan bilang "tidak menemukan" jika artikelnya ada.
Rangkum informasi dari artikel dengan bahasa Indonesia yang natural.
Sebutkan sumber di akhir jawaban.

Artikel berita:
{context}

Pertanyaan: {question}

Jawaban (langsung berdasarkan artikel di atas):"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain, retriever

def initialize_rag(index_path='faiss_index'):
    """Inisialisasi semua komponen RAG"""
    print("=== INISIALISASI RAG SYSTEM ===\n")
    embeddings = load_embedding_model()
    vector_store = load_vector_store(index_path, embeddings)
    llm = load_llm()
    rag_chain, retriever = build_rag_chain(vector_store, llm)
    print("\n✅ RAG System siap digunakan!\n")
    return rag_chain, retriever

if __name__ == "__main__":
    rag_chain, retriever = initialize_rag()

    test_questions = [
        "Apa berita terbaru tentang ekonomi Indonesia?",
        "Bagaimana kondisi politik Indonesia?",
    ]

    print("=== TEST RAG CHAIN ===\n")
    for question in test_questions:
        print(f"Pertanyaan: {question}")
        print("Mencari jawaban...")
        answer = rag_chain.invoke(question)
        print(f"Jawaban: {answer.content}")
        print("-" * 50)