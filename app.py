import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load("model_LR_GamGwo.pkl")
vectorizer = joblib.load("model_SVM_GamGwo.pkl")

st.title("Aplikasi Analisis Sentimen 🇮🇩")
st.write("Masukkan teks yang ingin dianalisis:")

input_text = st.text_area("Teks input")

if st.button("Analisis"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        X = vectorizer.transform([input_text])
        prediction = model.predict(X)[0]

        if prediction == "positif":
            st.success("Sentimen: POSITIF 👍")
        elif prediction == "negatif":
            st.error("Sentimen: NEGATIF 👎")
        else:
            st.info("Sentimen: NETRAL 😐")
