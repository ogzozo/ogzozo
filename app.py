import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords (only needed first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🎬 Sentiment Analysis App")
st.write("✅ App loaded successfully")
st.write("Enter a movie review and get its sentiment prediction.")

review = st.text_area("Your Review:")
if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        clean_review = preprocess_text(review)
        review_tfidf = vectorizer.transform([clean_review])
        prediction = model.predict(review_tfidf)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.subheader(f"Sentiment: {sentiment}")