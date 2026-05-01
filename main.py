from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

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
MODEL_PATH = os.path.join("models", "random_forest_model2.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder2.pkl")
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
    
    # Buang data duplikat biar kita dapet mapping unik dari Kategori -> Tanggapan
    df_mapping = df.dropna(subset=[kolom_kategori, kolom_tanggapan]).drop_duplicates(subset=[kolom_kategori])
    TANGGAPAN_KDRT = dict(zip(df_mapping[kolom_kategori], df_mapping[kolom_tanggapan]))
    
    print("✅ MANTAP BRAY! Rekomendasi sistem sukses di-load dari dataset.csv!")
    
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
            "tingkat_keyakinan_ai": 100.0,
            "rekomendasi_sistem": "Halo! Selamat datang di SafeTalk AI DP3A Kabupaten Indramayu. Silakan ceritakan detail kejadian atau keluhan yang Anda alami, kami siap membantu dan melindungi Anda."
        }

    try:
        # ==================================================
        # 2. FILTER KATA ASING (Out-Of-Vocabulary)
        # ==================================================
        tfidf_step = rf_model.named_steps['tfidf']
        vektor_teks = tfidf_step.transform([teks_masuk])
        probabilitas_max = 0.0
        
        # Kalau teksnya nggak dikenali sama sekali sama mesin TF-IDF
        if vektor_teks.nnz == 0:
            kode_kategori = "NON_KDRT"
        else:
            # ==================================================
            # 3. FILTER PROBABILITAS
            # ==================================================
            probabilitas = rf_model.predict_proba([teks_masuk])[0]
            probabilitas_max = max(probabilitas)
            
            # Kalau mesin gak yakin (di bawah 40%)
            if probabilitas_max < 0.40: 
                kode_kategori = "NON_KDRT"
            else:
                # ==================================================
                # 4. PREDIKSI NORMAL RANDOM FOREST
                # ==================================================
                hasil_prediksi_mentah = rf_model.predict([teks_masuk])[0]
                try:
                    kode_kategori = label_encoder.inverse_transform([hasil_prediksi_mentah])[0]
                except:
                    kode_kategori = str(hasil_prediksi_mentah)

        # ==================================================
        # 5. HASIL & REKOMENDASI SOP (Narik dari CSV)
        # ==================================================
        hasil_deskripsi = KATEGORI_KDRT.get(kode_kategori, "Kategori Tidak Dikenali")

        # Ini kuncinya bray! Narik balasan dari dictionary yang udah ngebaca dataset
        rekomendasi = TANGGAPAN_KDRT.get(kode_kategori, "Sistem telah menerima pesan Anda. Kami akan segera merespon.")

        return {
            "teks_laporan": teks_masuk,
            "kode_kategori": kode_kategori,
            "hasil_klasifikasi": hasil_deskripsi,
            "tingkat_keyakinan_ai": round(probabilitas_max * 100, 2) if probabilitas_max > 0 else 100.0,
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