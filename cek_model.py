import pickle
import joblib

file_path = "models/random_forest_model.pkl"

print("--- UJI COBA LOAD MODEL ---")

# Tes 1: Pake Joblib (Paling Sering Jadi Solusi)
try:
    print("Mencoba load dengan joblib...")
    model_joblib = joblib.load(file_path)
    print("✅ BERHASIL! Model ini di-save pakai joblib.")
    print("Tipe model:", type(model_joblib))
except Exception as e:
    print("❌ Gagal pakai joblib:", e)

print("\n---------------------------\n")

# Tes 2: Pake Pickle standar (Buat perbandingan aja)
try:
    print("Mencoba load dengan pickle standar...")
    with open(file_path, "rb") as f:
        model_pickle = pickle.load(f)
    print("✅ BERHASIL! Model ini di-save pakai pickle biasa.")
except Exception as e:
    print("❌ Gagal pakai pickle:", e)