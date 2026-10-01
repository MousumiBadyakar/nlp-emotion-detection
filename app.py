import streamlit as st
import joblib
import string
import nltk

from nltk.corpus import stopwords

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="✦",
    layout="centered"
)

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

def preprocess_text(txt):
    txt = txt.lower()

    txt = txt.translate(
        str.maketrans("", "", string.punctuation)
    )

    new = ""

    for i in txt:
        if not i.isdigit():
            new += i

    new = "".join(i for i in new if i.isascii())

    words = new.split()

    cleaned = []

    for word in words:
        if word not in stop_words:
            cleaned.append(word)

    return " ".join(cleaned)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');

    .stApp {
        background: #f3f0e9;
    }

    .block-container {
        max-width: 780px;
        padding-top: 5rem;
        padding-bottom: 4rem;
    }

    .eyebrow {
        text-align: center;
        color: #7a806d;
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .main-title {
        text-align: center;
        color: #252522;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 58px;
        font-weight: 600;
        line-height: 1.1;
        margin-bottom: 14px;
    }

    .subtitle {
        text-align: center;
        color: #77766f;
        font-family: 'DM Sans', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        margin: 0 auto 42px auto;
        max-width: 520px;
    }

    .input-card {
        background: #faf9f6;
        border: 1px solid #ddd9cf;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 12px 35px rgba(55, 53, 45, 0.06);
    }

    .input-label {
        color: #35352f;
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    textarea {
        background-color: #f7f5f0 !important;
        color: #292925 !important;
        border: 1px solid #d8d4ca !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 14px !important;
    }

    textarea:focus {
        border: 1px solid #858b73 !important;
        box-shadow: 0 0 0 1px #858b73 !important;
    }

    textarea::placeholder {
        color: #aaa79f !important;
    }

    div.stButton > button {
        background: #35382f;
        color: #f8f7f3;
        border: none;
        border-radius: 11px;
        height: 48px;
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.2px;
        margin-top: 16px;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #555a48;
        color: #ffffff;
        border: none;
        transform: translateY(-1px);
    }

    .result-card {
        background: #e4e6d8;
        border: 1px solid #d1d4c1;
        border-radius: 18px;
        padding: 28px;
        margin-top: 28px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(55, 53, 45, 0.05);
    }

    .result-label {
        color: #707462;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .result-emotion {
        color: #292b25;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 30px;
        font-weight: 600;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        font-family: 'DM Sans', sans-serif;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="eyebrow">Natural Language Processing</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Emotion Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Put your thoughts into words and discover the emotion expressed through your text.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="input-label">Your thoughts</div>',
    unsafe_allow_html=True
)

user_text = st.text_area(
    "",
    placeholder="Write something here...",
    height=170,
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

if st.button("Analyze Emotion", use_container_width=True):

    if user_text.strip() == "":
        st.warning("Please enter some text first.")

    else:
        cleaned_text = preprocess_text(user_text)

        text_vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(text_vector)[0]

        emotion_mapping = {
            0: "Sadness",
            1: "Anger",
            2: "Love",
            3: "Surprise",
            4: "Fear",
            5: "Joy"
        }

        emotion = emotion_mapping[prediction]

        emotion_symbols = {
            "Sadness": "☾",
            "Anger": "◈",
            "Love": "♡",
            "Surprise": "✦",
            "Fear": "◌",
            "Joy": "☀"
        }

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Detected Emotion</div>
                <div class="result-emotion">
                    {emotion_symbols[emotion]}&nbsp;&nbsp;{emotion}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )