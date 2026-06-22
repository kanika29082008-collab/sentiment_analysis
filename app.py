import streamlit as st
import joblib
import re
import string

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text.strip()

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Sentiment Analysis on Social Media")
st.write("Type any sentence or tweet below and see if it's Positive 😀 or Negative 😞")

user_input = st.text_area("Enter text:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        sentiment = "Positive 😀" if pred == 1 else "Negative 😞"
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")
