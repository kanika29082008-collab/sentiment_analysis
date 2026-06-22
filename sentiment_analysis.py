# sentiment_analysis.py

import os
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

# Step 1: Load dataset safely with balanced sampling
def load_data(path, sample_size=50000):
    folder = os.path.dirname(path)
    print("📂 Debug: Files in", folder)
    for f in os.listdir(folder):
        print(" -", f)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, encoding="latin-1", header=None)
    df = df[[5, 0]]
    df.columns = ["text", "label"]
    df["label"] = df["label"].apply(lambda x: 1 if x == 4 else 0)

    # Balanced sampling: half negative, half positive
    neg = df[df["label"] == 0].sample(sample_size // 2, random_state=42)
    pos = df[df["label"] == 1].sample(sample_size // 2, random_state=42)
    df_balanced = pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

# Step 2: Preprocess text with progress bar
def preprocess_texts(texts):
    return [preprocess_text(t) for t in tqdm(texts, desc="🔄 Preprocessing texts")]

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text

# Step 3: Train model with live accuracy and loss curve
def train_model(X_train, y_train, X_test, y_test, epochs=5):
    print("⚙️ Training model with SGDClassifier...")
    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(tqdm(X_train, desc="📊 Vectorizing training data"))
    X_test_vec = vectorizer.transform(X_test)

    model = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True, random_state=42)
    classes = [0, 1]

    losses = []
    accuracies = []

    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        model.partial_fit(X_train_vec, y_train, classes=classes)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Calculate log loss
        y_proba = model.predict_proba(X_test_vec)
        loss = log_loss(y_test, y_proba)
        losses.append(loss)

        print(f"   Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), losses, marker='o', label="Loss")
    plt.plot(range(1, epochs+1), accuracies, marker='s', label="Accuracy")
    plt.title("Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("✅ Training complete")
    return model, vectorizer

# Step 4: Predict sentiment
def predict_sentiment(model, vectorizer, text):
    text_clean = preprocess_text(text)
    text_vec = vectorizer.transform([text_clean])
    prediction = model.predict(text_vec)[0]
    return "Positive 😀" if prediction == 1 else "Negative 😞"

if __name__ == "__main__":
    dataset_path = r"C:\Users\Kanika\sentiment_analysis\sentiment140.csv"

    try:
        # Load and preprocess dataset with balanced sampling
        df = load_data(dataset_path, sample_size=50000)
        df["text"] = preprocess_texts(df["text"])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42
        )

        # Train model with live accuracy and loss curve
        model, vectorizer = train_model(X_train, y_train, X_test, y_test, epochs=5)

        # Final evaluation
        print("\n📊 Final Classification Report:")
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        print(classification_report(y_test, y_pred))

        # Demo prediction
        sample = "I love this product, it works great!"
        print(f"\nSample: {sample}")
        print("Sentiment:", predict_sentiment(model, vectorizer, sample))

    except FileNotFoundError as e:
        print("❌ Error:", e)
        print("Make sure the file name matches exactly.")
