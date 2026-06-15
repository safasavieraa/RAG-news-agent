import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Build FAISS index otomatis jika belum ada
if not os.path.exists('faiss_index'):
    print("Membangun FAISS index dari CSV...")
    from src.vector_store import build_vector_store
    build_vector_store(
        csv_path='data/clean_articles.csv',
        save_path='faiss_index'
    )

from src.rag_chain import initialize_rag

print("Memulai RAG News Agent...")
rag_chain, retriever = initialize_rag()

def get_sources(question):
    docs = retriever.invoke(question)
    sources = []
    for doc in docs:
        sources.append(
            f"📰 {doc.metadata['judul'][:60]}...\n"
            f"   {doc.metadata['source'].upper()} | {doc.metadata['waktu']}"
        )
    return "\n\n".join(sources)

def chat(message, history):
    if not message.strip():
        return ""
    response = rag_chain.invoke(message)
    answer = response.content
    sources = get_sources(message)
    return f"{answer}\n\n---\n📚 Artikel yang digunakan:\n{sources}"

demo = gr.ChatInterface(
    fn=chat,
    title="RAG News Agent Indonesia",
    description="Chatbot berita Indonesia dari Kompas, Tempo, Detik menggunakan RAG + Groq LLaMA 3.1",
    examples=[
        "Apa berita terbaru tentang Prabowo?",
        "Bagaimana kondisi ekonomi Indonesia?",
        "Ceritakan tentang demo mahasiswa",
    ],
    retry_btn=None,
    undo_btn=None,
)

demo.launch(server_name="0.0.0.0", server_port=7860)