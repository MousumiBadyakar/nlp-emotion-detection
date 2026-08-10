import streamlit as st
import joblib
import string
import nltk

from nltk.corpus import stopwords


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="💭",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL AND VECTORIZER
# --------------------------------------------------

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# --------------------------------------------------
# STOPWORDS
# --------------------------------------------------

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(txt):

    # Convert text to lowercase
    txt = txt.lower()

    # Remove punctuation
    txt = txt.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    new = ""

    for i in txt:
        if not i.isdigit():
            new += i

    # Remove non-ASCII characters
    new = "".join(i for i in new if i.isascii())

    # Split text into words
    words = new.split()

    # Remove stopwords
    cleaned = []

    for word in words:
        if word not in stop_words:
            cleaned.append(word)

    return " ".join(cleaned)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💭 Emotion Detection")

st.write("How are you feeling? Write it down below.")


# --------------------------------------------------
# TEXT INPUT
# --------------------------------------------------

user_text = st.text_area(
    "Your thoughts",
    placeholder="Write something here...",
    height=150,
    label_visibility="collapsed"
)


# --------------------------------------------------
# BUTTON STYLE
# --------------------------------------------------

st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #8B7E9E;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1rem;
        font-size: 16px;
        font-weight: 500;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #746684;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("✨ Discover Emotion", use_container_width=True):

    if user_text.strip() == "":
        st.warning("Please enter something first.")

    else:

        # Preprocess user input
        cleaned_text = preprocess_text(user_text)

        # Convert text into TF-IDF features
        text_vector = vectorizer.transform([cleaned_text])

        # Predict emotion
        prediction = model.predict(text_vector)[0]

        # Convert numerical prediction to emotion
        emotion_mapping = {
            0: "Sadness 😢",
            1: "Anger 😠",
            2: "Love ❤️",
            3: "Surprise 😲",
            4: "Fear 😨",
            5: "Joy 😊"
        }

        emotion = emotion_mapping[prediction]

        # Display result
        st.success(f"Detected emotion: {emotion}")