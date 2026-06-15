---
title: RAG News Agent Indonesia
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.7.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# RAG News Agent Indonesia

Chatbot berita Indonesia berbasis **RAG (Retrieval-Augmented Generation)**.

## 🔗 Links
- **Live Demo:** https://huggingface.co/spaces/saviera/rag-news-agent
- **GitHub:** https://github.com/safasavieraa/RAG-news-agent

## 📖 Tentang Proyek
Chatbot ini dapat menjawab pertanyaan seputar berita Indonesia berdasarkan artikel dari Kompas, Tempo, dan Detik (2024-2025) menggunakan teknik RAG.

## ⚙️ Tech Stack
- **LangChain** — RAG framework
- **FAISS** — Vector database untuk semantic search
- **Sentence Transformers** — Embedding model (all-MiniLM-L6-v2)
- **Groq API** — LLM inference (LLaMA 3.1 8B)
- **Gradio** — Chatbot UI
- **Python** — Bahasa pemrograman

## Arsitektur RAG
## 📊 Dataset
- 1.000 artikel berita Indonesia dari Kompas, Tempo, Detik
- Periode: 2024-2025

## Cara Menjalankan Lokal
1. Clone repo ini
2. Install dependencies: `pip install -r requirements.txt`
3. Buat file `.env` dan isi `GROQ_API_KEY=your_key`
4. Jalankan: `python app.py`