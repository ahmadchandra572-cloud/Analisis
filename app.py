import streamlit as st
import joblib
import re
import string
# PENTING: Sastrawi harus diimpor untuk Stemming (perlu di list requirements.txt)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 

# ==========================================
# 0️⃣ PREPROCESSING FUNCTION (Wajib Sama dengan Training)
# ==========================================
@st.cache_data
def text_preprocessing(text):
    if not isinstance(text, str): return ""
    
    # 1. Case Folding
    text = text.lower()
    
    # 2. Cleaning (Hapus Simbol/Angka/URL/Username)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+','', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 3. Stemming (Sastrawi)
    # Ini sering menyebabkan error di cloud, pastikan Sastrawi ada di requirements.txt
    try:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = stemmer.stem(text)
    except:
        pass # Lewati jika Sastrawi gagal
        
    return text.strip()

# ==========================================
# 1️⃣ LOAD SEMUA MODEL DAN VECTORIZER
# ==========================================
# Menggunakan st.cache_resource agar loading file .pkl hanya 1x
@st.cache_resource
def load_resources():
    try:
        # Load 3 MODEL OPTIMASI (Menggunakan NAMA FILE YANG BENAR di repo)
        models = {
            "Random Forest (RF)": joblib.load("model_RF_GamGwo.pkl"),
            "Logistic Regression (LR)": joblib.load("model_LR_GamGwo.pkl"),
            "Support Vector Machine (SVM)": joblib.load("model_SVM_GamGwo.pkl"),
        }
        # Load Vectorizer TF-IDF (PENTING untuk konversi teks ke angka)
        vectorizer = joblib.load("tfidf_vectorizer.pkl") 
        return models, vectorizer
    except Exception as e:
        # Menampilkan error di console Streamlit jika ada masalah loading file
        st.error(f"FATAL ERROR: Gagal memuat file .pkl. Pastikan semua model dan vectorizer terupload. Error: {e}")
        return None, None
            
MODELS, VECTORIZER = load_resources()


# ==========================================
# 2️⃣ ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("Aplikasi Analisis Sentimen DPR")
st.subheader("Model Optimasi GAM-GWO (RF | LR | SVM)")

# Pilihan Model menggunakan Selectbox
model_options = list(MODELS.keys()) if MODELS else ["(Error Loading Models)"]
model_choice = st.selectbox("Pilih Algoritma Prediksi:", model_options)

input_text = st.text_area("Masukkan Komentar YouTube di sini:", height=100)

if st.button("Analisis"):
    if MODELS is None:
        st.stop()
        
    elif input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
        
    else:
        # --- PREDICTIVE LOGIC ---
        
        # 1. Preprocessing Input
        clean_text = text_preprocessing(input_text)
        
        # 2. Vectorization (Menggunakan Vectorizer yang benar)
        # HIDE TF-IDF complexity from user, but it's essential here.
        X = VECTORIZER.transform([clean_text])
        
        # 3. Prediksi
        selected_model = MODELS[model_choice]
        prediction = selected_model.predict(X)[0]

        # 4. Tampilkan Hasil
        st.info(f"Teks Bersih (Preprocessed): {clean_text}")

        if prediction == "Positif":
            st.success(f"Sentimen: POSITIF 👍 (Model: {model_choice})")
        elif prediction == "Negatif":
            st.error(f"Sentimen: NEGATIF 👎 (Model: {model_choice})")
        else:
            st.warning(f"Sentimen: NETRAL 😐 (Model: {model_choice})")

### **Catatan Penting (Untuk Memperbaiki ModuleNotFoundError):**

Pastikan file **`requirements.txt`** Anda sudah mencakup library **`Sastrawi`** dan **`joblib`** agar Streamlit tidak *crash* lagi:
