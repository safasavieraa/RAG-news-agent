import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def load_articles(csv_path):
    """Load artikel dari CSV dan ubah ke format Document"""
    print("1. Loading artikel...")
    df = pd.read_csv(csv_path)
    
    documents = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Memproses"):
        # Setiap artikel dijadikan 1 Document
        # page_content = teks utama yang akan di-embedding
        # metadata = info tambahan (tidak di-embedding, tapi ikut tersimpan)
        doc = Document(
            page_content=str(row['text']),
            metadata={
                'judul': str(row['Judul']),
                'waktu': str(row['Waktu']),
                'source': str(row['source']),
            }
        )
        documents.append(doc)
    
    print(f"   {len(documents)} artikel berhasil diload")
    return documents

def build_vector_store(csv_path, save_path):
    """Buat dan simpan FAISS vector store"""
    print("=== MEMBANGUN VECTOR DATABASE ===\n")
    
    # 1. Load artikel
    documents = load_articles(csv_path)
    
    # 2. Siapkan embedding model
    # Model ini yang mengubah teks → vektor angka
    # all-MiniLM-L6-v2 = model kecil, cepat, gratis, cukup akurat
    print("\n2. Loading embedding model...")
    print("   (Download model ~90MB, hanya sekali)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("   Embedding model siap!")
    
    # 3. Buat FAISS vector store
    # Di sinilah semua artikel diubah jadi vektor dan disimpan ke FAISS
    print("\n3. Membuat vector database...")
    print("   (Proses embedding 5000 artikel, 3-10 menit...)")
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    print("   Vector database berhasil dibuat!")
    
    # 4. Simpan ke disk
    print(f"\n4. Menyimpan ke {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"   Berhasil disimpan!")
    
    # 5. Test pencarian
    print("\n5. Test pencarian...")
    test_query = "berita ekonomi Indonesia"
    results = vector_store.similarity_search(test_query, k=3)
    print(f"   Query: '{test_query}'")
    print(f"   Hasil teratas:")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] {doc.metadata['judul'][:60]}...")
    
    print("\n✅ Vector database selesai dibuat dan disimpan!")
    return vector_store

if __name__ == "__main__":
    build_vector_store(
        csv_path='data/clean_articles.csv',
        save_path='faiss_index'
    )