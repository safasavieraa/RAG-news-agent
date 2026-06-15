import pandas as pd

# Load dataset CSV
print("Loading dataset...")
df = pd.read_csv('data/final_merge_dataset.csv')

# Ukuran dataset
print("\n=== UKURAN DATASET ===")
print(f"Jumlah artikel : {len(df)}")
print(f"Kolom yang ada : {list(df.columns)}")

# 3 baris pertama
print("\n=== 3 BARIS PERTAMA ===")
print(df.head(3).to_string())

# Cek data kosong
print("\n=== CEK DATA KOSONG ===")
print(df.isnull().sum())

# Contoh isi 1 artikel
print("\n=== CONTOH ISI 1 ARTIKEL ===")
for col in df.columns:
    val = str(df[col].iloc[0])
    print(f"{col}: {val[:200]}")