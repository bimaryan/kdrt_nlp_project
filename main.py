from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import re  # TAMBAHAN: Dibutuhkan untuk fungsi clean_text

# ==========================================
# INISIALISASI APP
# ==========================================
app = FastAPI(
    title="SafeTalk AI Backend",
    description="API Konsultasi Berbasis NLP untuk Klasifikasi Teks Laporan KDRT (DP3A Kab. Indramayu)",
    version="1.0.0"
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
    # 1. Load model hasil dari Google Colab pakai joblib
    rf_model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ MANTAP BRAY! Model Pipeline & Label Encoder sukses di-load!")
    
    # 2. Load Tanggapan dari Dataset CSV
    # Asumsi pemisah kolomnya koma (,). Kalau error coba ganti sep=';'
    df = pd.read_csv(DATASET_PATH)
    
    # GANTI NAMA KOLOM DI BAWAH INI KALAU BEDA SAMA YANG DI CSV LU YA BRAY
    kolom_kategori = 'Label' 
    kolom_tanggapan = 'Tanggapan'
    
    # Validasi jika kolom tersedia di dataset
    if kolom_kategori in df.columns and kolom_tanggapan in df.columns:
        # Buang data duplikat biar kita dapet mapping unik dari Kategori -> Tanggapan
        df_mapping = df.dropna(subset=[kolom_kategori, kolom_tanggapan]).drop_duplicates(subset=[kolom_kategori])
        TANGGAPAN_KDRT = dict(zip(df_mapping[kolom_kategori], df_mapping[kolom_tanggapan]))
        print("✅ MANTAP BRAY! Rekomendasi sistem sukses di-load dari dataset.csv!")
    else:
        print("⚠️ Kolom kategori/tanggapan tidak ditemukan, pakai fallback!")
        raise Exception("Struktur kolom CSV tidak cocok.")
    
except Exception as e:
    pesan_error_asli = str(e)
    print(f"⚠️ Gagal nge-load file (Model atau Dataset)! Error aslinya: {e}")
    # Fallback kalau CSV gagal ke-baca (jaga-jaga aja)
    TANGGAPAN_KDRT = {
        "K5": "🚨 DARURAT: Nyawa terancam. Segera hubungi Polisi (110) atau hotline darurat (112) dan amankan diri Anda.",
        "K4": "Segera hubungi atau datangi kantor DP3A Kabupaten Indramayu untuk perlindungan fisik.",
        "K3": "Segera hubungi DP3A untuk pendampingan psikologis dan perlindungan.",
        "K2": "Segera hubungi DP3A untuk pendampingan.",
        "K1": "Disarankan untuk melakukan konseling mediasi keluarga/pernikahan melalui layanan DP3A.",
        "NON_KDRT": "Diarahkan ke layanan konseling umum atau pendampingan psikolog untuk pemulihan kesehatan mental."
    }

# ==========================================
# KAMUS KATEGORI KDRT (Buat deskripsi)
# ==========================================
KATEGORI_KDRT = {
    "NON_KDRT": "Bukan KDRT (Perasaan sedih, stres, depresi tanpa unsur kekerasan)",
    "K1": "Keluhan Ringan (Terkait relasi rumah tangga, belum jelas ada kekerasan)",
    "K2": "Kekerasan Verbal / Emosional (Dibentak, dihina, direndahkan, dimaki)",
    "K3": "Tekanan Psikologis / Kontrol (Intimidasi, ancaman, pengurungan, larangan)",
    "K4": "Kekerasan Fisik (Dipukul, ditampar, ditendang, didorong, dijambak)",
    "K5": "Kekerasan Berat / Darurat (Dicekik, diancam dibunuh, pakai senjata, luka parah)"
}

class ChatRequest(BaseModel):
    pesan_teks: str

# ==========================================
# FUNGSI PREPROCESSING (SINKRON DARI COLAB)
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
# ENDPOINT / ROUTES
# ==========================================

@app.get("/", tags=["Status"])
def cek_status():
    return {
        "status": "Online", 
        "project": "SafeTalk AI - DP3A Indramayu",
        "message": "Server backend siap menerima laporan!"
    }

@app.post("/klasifikasi-chat", tags=["AI Processing"])
def proses_klasifikasi(request: ChatRequest):
    if rf_model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail=f"Gagal Load PKL bray! Error Asli: {pesan_error_asli}")

    teks_masuk = request.pesan_teks.strip()

    if not teks_masuk:
        raise HTTPException(status_code=400, detail="Pesan teks nggak boleh kosong.")

    # ==================================================
    # 1. FITUR BALASAN SAPAAN (GREETING)
    # ==================================================
    kata_sapaan = ["halo", "hai", "p", "ping", "tes", "test", "selamat pagi", "selamat siang", "selamat sore", "selamat malam", "assalamualaikum"]
    
    if teks_masuk.lower() in kata_sapaan or len(teks_masuk.split()) < 3:
        return {
            "teks_laporan": teks_masuk,
            "kode_kategori": "SAPAAN",
            "hasil_klasifikasi": "Sapaan / Pesan Pendek",
            "rekomendasi_sistem": "Halo! Selamat datang di SafeTalk AI DP3A Kabupaten Indramayu. Silakan ceritakan detail kejadian atau keluhan yang Anda alami, kami siap membantu dan melindungi Anda."
        }

    try:
        # ==================================================
        # 2. PREPROCESSING
        # ==================================================
        # Wajib di-clean dulu biar persis kayak data training di Colab
        teks_bersih = clean_text(teks_masuk)

        # ==================================================
        # 3. PREDIKSI LANGSUNG DARI PIPELINE (KEMBAR DENGAN COLAB)
        # ==================================================
        # Catatan: rf_model ini sebenarnya adalah Pipeline (TF-IDF + Random Forest)
        # Jadi kita tinggal masukkan teks string-nya saja, otomatis di-TF-IDF-kan oleh pipeline
        
        probabilitas = rf_model.predict_proba([teks_bersih])[0]
        probabilitas_max = max(probabilitas)
        
        # Ambil prediksi mentah (angka)
        hasil_prediksi_mentah = rf_model.predict([teks_bersih])[0]
        
        try:
            # Ubah angka kembali jadi label ("K5", "NON_KDRT", dll)
            kode_kategori = label_encoder.inverse_transform([hasil_prediksi_mentah])[0]
        except:
            kode_kategori = str(hasil_prediksi_mentah)

        # ==================================================
        # 4. HASIL & REKOMENDASI SOP (Narik dari CSV)
        # ==================================================
        hasil_deskripsi = KATEGORI_KDRT.get(kode_kategori, "Kategori Tidak Dikenali")

        # Narik balasan dari dictionary dataset
        rekomendasi = TANGGAPAN_KDRT.get(kode_kategori, "Sistem telah menerima pesan Anda. Kami akan segera merespon.")

        return {
            "teks_laporan": teks_masuk,  
            "teks_dianalisis": teks_bersih, 
            "kode_kategori": kode_kategori,
            "hasil_klasifikasi": hasil_deskripsi,
            "rekomendasi_sistem": rekomendasi
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Otak AI lagi error pas ngeproses bray: {str(e)}")

# ==========================================
# RUN SERVER LOKAL
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)