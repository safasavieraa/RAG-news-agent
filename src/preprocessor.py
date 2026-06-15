import pandas as pd
import re
from tqdm import tqdm

def clean_text(text):
    """Bersihkan teks dari karakter tidak perlu"""
    if not isinstance(text, str):
        return ""
    # Hapus link URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # Hapus karakter spesial, sisakan huruf, angka, spasi, tanda baca dasar
    text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(input_path, output_path, max_articles=5000):
    """
    Load, bersihkan, dan simpan dataset
    """
    print("=== MEMULAI PREPROCESSING ===\n")

    # 1. Load data
    print("1. Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"   Jumlah awal: {len(df)} artikel")

    # 2. Hapus artikel dengan Content kosong
    print("\n2. Menghapus artikel kosong...")
    df = df.dropna(subset=['Content'])
    df = df[df['Content'].str.strip() != '']
    print(f"   Jumlah setelah hapus kosong: {len(df)} artikel")

    # 3. Ambil kolom yang diperlukan saja
    print("\n3. Memilih kolom penting...")
    df = df[['Judul', 'Content', 'Waktu', 'source']].copy()

    # 4. Bersihkan teks
    print("\n4. Membersihkan teks...")
    tqdm.pandas(desc="   Cleaning")
    df['Judul'] = df['Judul'].progress_apply(clean_text)
    df['Content'] = df['Content'].progress_apply(clean_text)

    # 5. Gabungkan Judul + Content jadi satu kolom
    print("\n5. Menggabungkan Judul + Content...")
    df['text'] = df['Judul'] + '. ' + df['Content']

    # 6. Hapus artikel dengan teks terlalu pendek (kurang dari 100 karakter)
    print("\n6. Hapus artikel terlalu pendek...")
    df = df[df['text'].str.len() >= 100]
    print(f"   Jumlah setelah filter pendek: {len(df)} artikel")

    # 7. Ambil 5000 artikel saja
    print(f"\n7. Mengambil {max_articles} artikel...")
    df = df.sample(n=max_articles, random_state=42).reset_index(drop=True)
    print(f"   Jumlah final: {len(df)} artikel")

    # 8. Simpan hasil
    print(f"\n8. Menyimpan ke {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"   Berhasil disimpan!")

    # 9. Tampilkan contoh hasil
    print("\n=== CONTOH HASIL PREPROCESSING ===")
    print(f"Judul  : {df['Judul'].iloc[0]}")
    print(f"Source : {df['source'].iloc[0]}")
    print(f"Waktu  : {df['Waktu'].iloc[0]}")
    print(f"Text   : {df['text'].iloc[0][:200]}...")
    print(f"\n✅ Preprocessing selesai! {len(df)} artikel siap digunakan.")

    return df

if __name__ == "__main__":
    preprocess_dataset(
        input_path='data/final_merge_dataset.csv',
        output_path='data/clean_articles.csv',
        max_articles=5000
    )