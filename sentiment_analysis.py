import pandas as pd
import re
import string
import joblib
import matplotlib
matplotlib.use("Agg")   # non-GUI backend (no blocking window)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text.strip()

# -----------------------------
# Training function
# -----------------------------
def train_model(X_train, y_train, X_test, y_test, epochs=5):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(tqdm(X_train, desc="Vectorizing train data"))
    X_test_vec = vectorizer.transform(tqdm(X_test, desc="Vectorizing test data"))

    model = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)

    acc_list, loss_list = [], []

    for epoch in range(1, epochs + 1):
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        acc_list.append(acc)
        loss_list.append(loss)

        print(f"📈 Epoch {epoch}/{epochs}\n   Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # Save plot instead of showing
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), acc_list, label="Accuracy", marker="o")
    plt.plot(range(1, epochs + 1), loss_list, label="Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.close()

    # Save model and vectorizer
    joblib.dump(model, "sentiment_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

    return model, vectorizer

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    # Load dataset (replace with your path)
    df = pd.read_csv("sentiment140.csv", encoding="latin-1", usecols=[0, 5], names=["target", "text"], skiprows=1)
    df["text"] = df["text"].apply(clean_text)
    df["target"] = df["target"].map({0: 0, 4: 1})  # 0=negative, 4=positive

    # Sample smaller dataset for faster training
    df_sample = df.sample(n=20000, random_state=42)
    X = df_sample["text"]
    y = df_sample["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model, vectorizer = train_model(X_train, y_train, X_test, y_test, epochs=5)

    # Demo prediction
    sample = "I love this product, it works great!"
    sample_vec = vectorizer.transform([clean_text(sample)])
    pred = model.predict(sample_vec)[0]
    print(f"\nSample: {sample}\nSentiment: {'Positive 😀' if pred == 1 else 'Negative 😞'}")
