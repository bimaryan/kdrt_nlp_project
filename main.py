from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Definisikan variabel
model_pipeline = None
encoder = None
model_load_error = None
dataset = None


try:
    print("Mencoba meload model dan dataset...")
    # Load Model
    model_pipeline = joblib.load("models/random_forest_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    
    # Load Dataset untuk mengambil 'Tanggapan'
    # Pastikan path ini sesuai dengan struktur folder Anda
    dataset = pd.read_csv("data/processed/dataset_labeled.csv")
    
    print("✅ Model, Encoder, dan Dataset berhasil dimuat!")
except Exception as e:
    model_load_error = str(e)
    print(f"❌ ERROR SAAT LOAD MODEL/DATASET: {e}")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_kdrt(data: TextInput):
    if model_pipeline is None or dataset is None:
        return {"status": "error", "message": "Server error."}

    try:
        # Prediksi
        prediction_numeric = model_pipeline.predict([data.text])
        predicted_label = encoder.inverse_transform(prediction_numeric)[0]
        
        # --- LOGIKA MENCARI TANGGAPAN ---
        # 1. Filter dataset sesuai kategori tebakan model
        df_filtered = dataset[dataset['label_auto'] == predicted_label].copy()
        
        if df_filtered.empty:
            tanggapan_terpilih = "Mohon maaf, tanggapan spesifik tidak tersedia untuk kategori ini."
        else:
            # 2. Reset index agar pencarian tidak error
            df_filtered = df_filtered.reset_index(drop=True)
            
            # 3. Ubah semua Konteks di dataset dan input pengguna menjadi vektor
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(df_filtered['Konteks'])
            user_tfidf = tfidf.transform([data.text])
            
            # 4. Hitung tingkat kemiripan kalimat (Cosine Similarity)
            similarities = cosine_similarity(user_tfidf, tfidf_matrix)
            
            # 5. Cari index kalimat Konteks yang paling mirip
            best_match_idx = similarities.argmax()
            
            # 6. Ambil Tanggapan dari index yang paling mirip tersebut
            tanggapan_terpilih = df_filtered.iloc[best_match_idx]['Tanggapan']
        
        return {
            "status": "success",
            "predicted_label": predicted_label,
            "user_input": data.text,
            "tanggapan_ai": tanggapan_terpilih
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}