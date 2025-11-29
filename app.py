import streamlit as st
import pandas as pd
from model import train_model, predict_sentiment

st.set_page_config(page_title="Climate Change Modeling", layout="wide")

st.title("🌍 Climate Change Modeling – NLP & Sentiment Analysis")

uploaded_file = st.file_uploader("Upload NASA Comments Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📌 Dataset Preview", df.head())

    if st.button("Train Machine Learning Model"):
        report, model_path = train_model(df)
        st.success("Training Completed!")
        st.write("### 🔍 Classification Report")
        st.text(report)

    st.subheader("🔮 Predict Sentiment for New Comment")
    user_text = st.text_area("Enter a climate-change related comment")

    if st.button("Predict"):
        result = predict_sentiment(user_text)
        st.write("### Sentiment Result:")
        st.success(result)
