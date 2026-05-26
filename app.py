import joblib

# Load saved files
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# User input
news = input("Enter News Text: ")

# Convert text
news_vector = vectorizer.transform([news])

# Prediction
prediction = model.predict(news_vector)

# Result
if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")