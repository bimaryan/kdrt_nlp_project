from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
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
# SETUP PATH & LOAD MODEL AI (.pkl)
# ==========================================
# Pastiin nama file di folder "models" udah sesuai sama ini ya bray!
MODEL_PATH = os.path.join("models", "random_forest_model2.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder2.pkl")

rf_model = None
label_encoder = None
pesan_error_asli = ""

try:
    # Load model hasil dari Google Colab pakai joblib
    rf_model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
        
    print("✅ MANTAP BRAY! Model Pipeline & Label Encoder sukses di-load!")
except Exception as e:
    pesan_error_asli = str(e)
    print(f"⚠️ Gagal nge-load model! Error aslinya: {e}")

# ==========================================
# KAMUS KATEGORI KDRT
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
        # 5. HASIL & REKOMENDASI SOP
        # ==================================================
        hasil_deskripsi = KATEGORI_KDRT.get(kode_kategori, "Kategori Tidak Dikenali")

        if kode_kategori == "K5":
            rekomendasi = "🚨 DARURAT: Nyawa terancam. Segera hubungi Polisi (110) atau hotline darurat (112) dan amankan diri Anda."
        elif kode_kategori in ["K2", "K3", "K4"]:
            rekomendasi = "Segera hubungi atau datangi kantor DP3A Kabupaten Indramayu untuk perlindungan dan pendampingan hukum/psikologis."
        elif kode_kategori == "K1":
            rekomendasi = "Disarankan untuk melakukan konseling mediasi keluarga/pernikahan melalui layanan DP3A."
        elif kode_kategori == "NON_KDRT":
            rekomendasi = "Diarahkan ke layanan konseling umum atau pendampingan psikolog untuk pemulihan kesehatan mental."
        else:
            rekomendasi = "Sistem telah menerima pesan Anda."

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