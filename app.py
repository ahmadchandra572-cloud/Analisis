import streamlit as st
import joblib
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Jika ini digunakan untuk stemming

# ==========================================
# 0️⃣ PREPROCESSING FUNCTION (Wajib Ada)
# ==========================================
# (Asumsi Anda memasukkan fungsi preprocessing lengkap Anda di sini,
# seperti yang kita bahas sebelumnya, untuk memastikan akurasi)
@st.cache_data
def text_preprocessing(text):
    # --- Tempatkan seluruh logika cleaning, case folding, dan stemming di sini ---
    # Contoh implementasi sederhana:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus simbol/angka
    # Asumsi StemmerFactory sudah didefinisikan jika menggunakan stemming
    # text = stemmer.stem(text) 
    return text.strip()
# ---------------------------------------------------------------------------------


# ==========================================
# 1️⃣ LOAD SEMUA MODEL DAN VECTORIZER (DENGAN NAMA FILE YANG BENAR)
# ==========================================
# Gunakan st.cache_resource agar loading hanya dilakukan sekali
@st.cache_resource
def load_all_models():
    # Mengoreksi semua nama file
    try:
        models = {
            "Random Forest (GAM-GWO)": joblib.load("model_RF_GamGwo.pkl"),
            "Logistic Regression (GAM-GWO)": joblib.load("model_LR_GamGwo.pkl"),
            "Support Vector Machine (GAM-GWO)": joblib.load("model_SVM_GamGwo.pkl"),
        }
        # MEMUAT VECTORIZER DENGAN NAMA FILE YANG BENAR
        vectorizer = joblib.load("tfidf_vectorizer.pkl") 
        return models, vectorizer
    except Exception as e:
        st.error(f"Error memuat file: Pastikan semua .pkl dan tfidf_vectorizer.pkl ada. Detail: {e}")
        return None, None

# Panggil fungsi load
MODELS, VECTORIZER = load_all_models()

# ==========================================
# 2️⃣ ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("Aplikasi Analisis Sentimen 🇮🇩")
st.write("Prediksi Akurasi: RF > LR > SVM (sesuai hasil optimasi)")

# Pilihan Model menggunakan Selectbox
model_choice = st.selectbox(
    "Pilih Model untuk Prediksi:",
    list(MODELS.keys())
)

input_text = st.text_area("Teks input")

if st.button("Analisis"):
    if MODELS is None or VECTORIZER is None:
        st.warning("Model belum dimuat karena ada error FileNotFoundError.")
        
    elif input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
        
    else:
        # 1. Preprocessing Input
        clean_text = text_preprocessing(input_text)
        
        # 2. Transformasi ke Angka (Menggunakan Vectorizer yang benar)
        X = VECTORIZER.transform([clean_text])
        
        # 3. Prediksi menggunakan model yang dipilih
        selected_model = MODELS[model_choice]
        prediction = selected_model.predict(X)[0]

        # 4. Tampilkan Hasil
        st.info(f"Teks Bersih: {clean_text}")

        if prediction == "Positif":
            st.success("Sentimen: POSITIF 👍")
        elif prediction == "Negatif":
            st.error("Sentimen: NEGATIF 👎")
        else:
            st.warning("Sentimen: NETRAL 😐")
