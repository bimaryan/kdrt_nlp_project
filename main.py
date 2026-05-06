from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import re
from groq import Groq
from dotenv import load_dotenv

# ==========================================
# INISIALISASI ENVIRONMENT & API KEY GROQ
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = None
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY tidak ditemukan di environment Docker!")
else:
    groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(
    title="SafeTalk AI Backend",
    description="API Konsultasi Berbasis NLP untuk Klasifikasi Teks Laporan KDRT (DP3A Kab. Indramayu) dengan integrasi Groq LLM",
    version="2.1.0"
)

# ==========================================
# SETUP PATH & LOAD MODEL AI (.pkl) & DATASET
# ==========================================
MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
DATASET_PATH = os.path.join("data", "raw", "dataset.csv")

rf_model = None
label_encoder = None
pesan_error_asli = ""
TANGGAPAN_KDRT = {}

try:
    rf_model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ MANTAP! Model & Label Encoder sukses di-load!")
    
    df = pd.read_csv(DATASET_PATH)
    kolom_kategori = 'Label' 
    kolom_tanggapan = 'Tanggapan'
    
    if kolom_kategori in df.columns and kolom_tanggapan in df.columns:
        df_mapping = df.dropna(subset=[kolom_kategori, kolom_tanggapan]).drop_duplicates(subset=[kolom_kategori])
        TANGGAPAN_KDRT = dict(zip(df_mapping[kolom_kategori], df_mapping[kolom_tanggapan]))
    else:
        raise Exception("Struktur kolom CSV tidak cocok.")
    
except Exception as e:
    pesan_error_asli = str(e)
    print(f"⚠️ Gagal load file: {e}")
    TANGGAPAN_KDRT = {
        "K5": "🚨 DARURAT: Nyawa terancam. Segera hubungi Polisi (110) atau hotline darurat (112) dan amankan diri Anda."
    }

KATEGORI_KDRT = {
    "NON_KDRT": "Bukan KDRT",
    "K1": "Keluhan Ringan",
    "K2": "Kekerasan Verbal / Emosional",
    "K3": "Tekanan Psikologis / Kontrol",
    "K4": "Kekerasan Fisik",
    "K5": "Kekerasan Berat / Darurat"
}

class ChatRequest(BaseModel):
    pesan_teks: str

# ==========================================
# FUNGSI PREPROCESSING
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿáéíóúàèìòùâêîôûçñ\s]", " ", text)
    
    # NORMALISASI KATA (Perbaiki typo atau imbuhan yang salah)
    kamus_perbaikan = {
        "di ancam": "diancam",
        "di pukul": "dipukul",
        "di tampar": "ditampar",
        "di cekik": "dicekik",
        "di bunuh": "dibunuh",
        "di tendang": "ditendang"
    }
    
    for salah, benar in kamus_perbaikan.items():
        text = text.replace(salah, benar)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================================
# ENDPOINT
# ==========================================
@app.get("/")
def cek_status():
    return {"status": "Online"}

@app.post("/klasifikasi-chat")
def proses_klasifikasi(request: ChatRequest):
    if rf_model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail=f"Gagal Load PKL bray! Error: {pesan_error_asli}")

    teks_masuk = request.pesan_teks.strip()
    if not teks_masuk:
        raise HTTPException(status_code=400, detail="Pesan kosong.")

    # ==================================================
    # 1. FITUR BALASAN SAPAAN (GREETING)
    # ==================================================
    kata_sapaan = ["halo", "hai", "p", "ping", "tes", "test", "selamat pagi", "selamat siang", "selamat sore", "selamat malam", "assalamualaikum"]
    
    if teks_masuk.lower() in kata_sapaan or len(teks_masuk.split()) < 3:
        return {
            "teks_laporan": teks_masuk,
            "teks_dianalisis": teks_masuk,
            "kode_kategori": "SAPAAN",
            "hasil_klasifikasi": "Sapaan / Pesan Pendek",
            "rekomendasi_sistem_raw": "Halo! Selamat datang di SafeTalk AI DP3A Kabupaten Indramayu. Silakan ceritakan detail kejadian atau keluhan yang Anda alami.",
            "tanggapan_llm": "Halo! Saya adalah asisten virtual SafeTalk AI dari DP3A Kabupaten Indramayu. Ada yang bisa saya bantu? Silakan ceritakan keluhan atau kejadian yang Anda alami dengan tenang. Kami siap mendengarkan dan membantu Anda."
        }

    # ==================================================
    # 2. PREPROCESSING NLP
    # ==================================================
    teks_bersih = clean_text(teks_masuk)
    
    try:
        # 3. PREDIKSI
        probabilitas = rf_model.predict_proba([teks_bersih])[0]
        probabilitas_max = max(probabilitas)
        hasil_prediksi_mentah = rf_model.predict([teks_bersih])[0]
        
        try:
            kode_kategori = label_encoder.inverse_transform([hasil_prediksi_mentah])[0]
        except:
            kode_kategori = str(hasil_prediksi_mentah)

        hasil_deskripsi = KATEGORI_KDRT.get(kode_kategori, "Tidak Dikenali")
        rekomendasi_sop = TANGGAPAN_KDRT.get(kode_kategori, "Kami akan segera merespon.")

        # 4. LLM GENERATION MENGGUNAKAN GROQ
        tanggapan_ai_final = rekomendasi_sop 
        
        if groq_client:
            prompt_user = f"""
            Anda adalah konselor virtual dari layanan SafeTalk AI milik DP3A Kabupaten Indramayu.
            Tugas Anda adalah memberikan balasan yang sangat empatik, menenangkan, dan solutif kepada pelapor kasus KDRT.
            
            Informasi Laporan:
            - Cerita Pelapor: "{teks_masuk}"
            - Kategori Deteksi Sistem: {hasil_deskripsi}
            - Arahan SOP DP3A Kabupaten Indramayu: "{rekomendasi_sop}"
            
            Instruksi:
            1. Validasi perasaan korban dengan empati.
            2. Masukkan arahan dari "Arahan SOP DP3A Kabupaten Indramayu" dengan bahasa halus.
            3. JANGAN mengarang nomor telepon/kontak selain dari "Arahan SOP DP3A Kabupaten Indramayu".
            4. Maksimal 1 paragraf singkat.
            """
            try:
                # Memanggil API Groq dengan model Llama 3.1 8B
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "Anda adalah konselor virtual dari layanan SafeTalk AI milik DP3A Kabupaten Indramayu."
                        },
                        {
                            "role": "user",
                            "content": prompt_user
                        }
                    ],
                    model="llama-3.1-8b-instant", 
                    temperature=0.4,
                    max_tokens=512,
                    top_p=0.9,
                )
                tanggapan_ai_final = chat_completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ Gagal generate LLM Groq: {e}")

        return {
            "teks_laporan": teks_masuk,  
            "teks_dianalisis": teks_bersih, 
            "kode_kategori": kode_kategori,
            "hasil_klasifikasi": hasil_deskripsi,
            "rekomendasi_sistem_raw": rekomendasi_sop, 
            "tanggapan_llm": tanggapan_ai_final 
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error AI: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8111, reload=True)