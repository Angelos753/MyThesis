import streamlit as st
from model import predict

st.set_page_config(page_title="Rumor Classifier", layout="centered")

st.title("🕵️ Rumor Detection using BERT")
st.write("Enter a text to classify whether it is a **rumor** or **not**.")

user_input = st.text_area("Enter text here:", height=150)

if st.button("Predict"):
    if user_input.strip():
        prediction = predict(user_input)
        st.success(f"🧠 Prediction: **{prediction}**")
    else:
        st.warning("Please enter some text.")

