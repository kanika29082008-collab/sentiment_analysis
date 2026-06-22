# 📊 Sentiment Analysis on Social Media

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview
This project implements **Sentiment Analysis** on social media text using the [Sentiment140 dataset](http://help.sentiment140.com/for-students).  
It classifies tweets into **Positive 😀** or **Negative 😞** sentiments using machine learning (SGDClassifier with incremental training).

---

## 🚀 Features
- Preprocessing pipeline (lowercasing, URL removal, punctuation cleaning).
- Balanced sampling of dataset for efficient training.
- Real‑time progress bars for preprocessing and vectorization.
- Live accuracy and loss tracking per epoch.
- Loss & accuracy curve visualization with `matplotlib`.
- Model persistence with `joblib` (save & reload without retraining).
- Demo prediction for quick testing.

---

## 📂 Project Structure

---

## 📥 Dataset
This project uses the **Sentiment140 dataset (1.6M tweets)**.  
Due to GitHub’s 100 MB file limit, the dataset is excluded from the repo.  

👉 Download it here: [Sentiment140 Dataset](http://help.sentiment140.com/for-students)  
Place the file as `sentiment140.csv` in the project root.

---

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/kanika29082008-collab/sentiment_analysis.git
cd sentiment_analysis
pip install -r requirements.txt
