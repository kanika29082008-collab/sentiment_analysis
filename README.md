# 📊 Sentiment Analysis on Social Media

A machine learning project that classifies social media text (tweets/reviews) into **Positive 😀** or **Negative 😞** sentiment using NLP techniques.

---

## 🚀 Features
- Clean text preprocessing (URLs, punctuation, case normalization)
- TF‑IDF vectorization
- Logistic regression (SGDClassifier) model
- Training progress plots (accuracy & loss curves)
- Saved model + vectorizer for reuse
- Interactive **Streamlit app** for live predictions

---

## 📂 Dataset
This project uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).  
⚠️ Note: The dataset is **not included in this repo** due to size limits. Please download it separately and place `sentiment140.csv` in your project folder.

---

## 🛠️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/sentiment_analysis.git
cd sentiment_analysis
pip install -r requirements.txt
