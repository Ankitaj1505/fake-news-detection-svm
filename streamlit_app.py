import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Title
st.title("Fake News Detection App")

st.write("Enter a news article or headline below:")

# User input
news = st.text_area("News Text")

# Prediction button
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Transform text
        news_vector = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(news_vector)

        # Result
        if prediction[0] == 0:
            st.error("This News is FAKE")
        else:
            st.success("This News is REAL")