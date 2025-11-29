import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

MODEL_FILE = "sentiment_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

def train_model(df):
    # Use LikesCount as proxy sentiment label (you can modify later)
    df["sentiment"] = df["LikesCount"].apply(lambda x: 1 if x > 1 else 0)

    X = df["Text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    report = classification_report(y_test, preds)

    # Save model & vectorizer
    pickle.dump(model, open(MODEL_FILE, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_FILE, "wb"))

    return report, MODEL_FILE

def predict_sentiment(text):
    vectorizer = pickle.load(open(VECTORIZER_FILE, "rb"))
    model = pickle.load(open(MODEL_FILE, "rb"))

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return "Positive 👍" if pred == 1 else "Negative 👎"
