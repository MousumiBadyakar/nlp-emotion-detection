# 💭 NLP-Based Emotion Detection

A machine learning project that uses Natural Language Processing (NLP) to detect the emotion expressed in a given text.

The application allows users to enter a sentence and predicts one of six emotions using a trained Linear Support Vector Machine (SVM) model.

---

## ✨ Features

- Emotion detection from text
- NLP-based text preprocessing
- TF-IDF feature extraction
- Unigram and bigram features
- Linear SVM classification
- Six emotion categories
- Interactive Streamlit web application
- Approximately 90.19% test accuracy

---

## 🎭 Emotion Categories

The model predicts one of the following emotions:

- 😢 Sadness
- 😠 Anger
- ❤️ Love
- 😲 Surprise
- 😨 Fear
- 😊 Joy

---

## 🧠 Machine Learning Pipeline

The project follows this pipeline:

```text
Raw Text
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Unigrams + Bigrams
   ↓
Linear SVM
   ↓
Emotion Prediction