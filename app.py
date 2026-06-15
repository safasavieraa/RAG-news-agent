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
            f"📰 {doc.metadata['judul'][:70]}...\n"
            f"   📌 {doc.metadata['source'].upper()} | 🕐 {doc.metadata['waktu']}"
        )
    return "\n\n".join(sources)

def chat(message, history):
    if not message.strip():
        return ""
    response = rag_chain.invoke(message)
    answer = response.content
    sources = get_sources(message)
    return f"{answer}\n\n---\n📚 **Sumber Artikel:**\n{sources}"

# Custom CSS
css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
#header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 10px;
    margin-bottom: 20px;
}
#header h1 {
    color: #e94560;
    font-size: 2em;
    margin: 0;
}
#header p {
    color: #a8b2d8;
    margin: 5px 0 0 0;
}
.badge {
    display: inline-block;
    background: #e94560;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    margin: 2px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("""
    <div id="header">
        <h1>RAG News Agent Indonesia</h1>
        <p>Chatbot berita Indonesia berbasis Retrieval-Augmented Generation</p>
        <br>
        <span class="badge">🤖 Groq LLaMA 3.1</span>
        <span class="badge">🔍 FAISS Vector Search</span>
        <span class="badge">📰 Kompas • Tempo • Detik</span>
    </div>
    """)

    chatbot = gr.Chatbot(
        label="",
        height=450,
        bubble_full_width=False,
        show_label=False,
        avatar_images=("👤", "🗞️"),
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="💬 Tanyakan berita Indonesia di sini...",
            label="",
            scale=9,
            lines=1,
            autofocus=True
        )
        send_btn = gr.Button("Kirim →", scale=1, variant="primary")

    with gr.Row():
        clear_btn = gr.Button("🗑️ Bersihkan Chat", variant="secondary")

    gr.Examples(
        examples=[
            "Apa berita terbaru tentang Prabowo?",
            "Bagaimana kondisi ekonomi Indonesia?",
            "Ceritakan tentang demo mahasiswa",
            "Apa itu reshuffle kabinet?",
        ],
        inputs=msg_input,
        label="💡 Contoh Pertanyaan"
    )

    gr.HTML("""
    <div style="text-align:center; padding:15px; color:#666; font-size:0.85em; border-top: 1px solid #333; margin-top:10px">
        <b>Tech Stack:</b> Python • LangChain • FAISS • Groq LLaMA 3.1 • Gradio &nbsp;|&nbsp;
        <b>Dataset:</b> 5.000 artikel berita Indonesia (2024-2025)
    </div>
    """)

    state = gr.State([])

    def chat_fn(message, history):
        if not message.strip():
            return "", history
        response = rag_chain.invoke(message)
        answer = response.content
        sources = get_sources(message)
        full_response = f"{answer}\n\n---\n📚 **Sumber:**\n{sources}"
        history = history + [[message, full_response]]
        return "", history

    send_btn.click(
        fn=chat_fn,
        inputs=[msg_input, state],
        outputs=[msg_input, state]
    ).then(fn=lambda h: h, inputs=[state], outputs=[chatbot])

    msg_input.submit(
        fn=chat_fn,
        inputs=[msg_input, state],
        outputs=[msg_input, state]
    ).then(fn=lambda h: h, inputs=[state], outputs=[chatbot])

    clear_btn.click(fn=lambda: ([], []), outputs=[state, chatbot])

demo.launch(server_name="0.0.0.0")