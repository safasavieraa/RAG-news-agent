import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

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

css = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("""
# 🗞️ RAG News Agent Indonesia
### Chatbot berita Indonesia berbasis Retrieval-Augmented Generation
**🤖 Groq LLaMA 3.1** &nbsp;|&nbsp; **🔍 FAISS Vector Search** &nbsp;|&nbsp; **📰 Kompas • Tempo • Detik**
---
""")

    chatbot = gr.Chatbot(
        label="",
        height=450,
        bubble_full_width=False,
        show_label=False,
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="💬 Tanyakan berita Indonesia di sini...",
            label="",
            scale=9,
            lines=1,
        )
        send_btn = gr.Button("Kirim →", scale=1, variant="primary")

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

    gr.Markdown("""
---
**Tech Stack:** Python • LangChain • FAISS • Groq LLaMA 3.1 • Gradio | **Dataset:** 5.000 artikel berita Indonesia (2024-2025)
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