# 📊 Sentiment Analysis on Social Media

A machine learning project that classifies social media text (tweets/reviews) into **Positive 😀** or **Negative 😞** sentiment using NLP techniques.

---

## 🔄 Project Workflow

```mermaid
flowchart TD
    A[📥 Data Collection\nSentiment140 Dataset] --> B[🧹 Preprocessing\nClean text, remove URLs & punctuation]
    B --> C[🔠 Feature Extraction\nTF-IDF Vectorization]
    C --> D[🤖 Model Training\nSGDClassifier (Logistic Regression)]
    D --> E[📈 Evaluation\nAccuracy & Loss Curves]
    E --> F[💾 Save Artifacts\nModel + Vectorizer]
    F --> G[🌐 Streamlit App\nInteractive Sentiment Prediction]
