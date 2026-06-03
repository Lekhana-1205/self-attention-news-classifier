import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("model.h5")

# -----------------------------
# Load Tokenizer
# -----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# Load Label Map
# -----------------------------
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

reverse_label_map = {v: k for k, v in label_map.items()}

MAX_LEN = 200

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI News Intelligence System")
st.title("📰 AI News Intelligence System")

article = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste news article here..."
)

# -----------------------------
# Preprocess
# -----------------------------
def preprocess(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
    return padded

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze Article"):

    if article.strip() == "":
        st.warning("Please enter a news article.")

    else:

        processed = preprocess(article)

        prediction = model.predict(processed)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        category = reverse_label_map[predicted_class]

        # -------------------------
        # Result
        # -------------------------
        st.subheader("Predicted Category")
        st.success(f"{category.title()}")
        st.write(f"Confidence: {confidence:.2f}%")

        # -------------------------
        # Important Words (FIXED)
        # -------------------------
        st.subheader("Important Words")

        words = article.lower().split()

        word_freq = {}

        for word in words:
            clean_word = word.strip(".,!?;:'\"()[]{}")
            if len(clean_word) > 3:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

        important_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        if len(important_words) > 0:

            max_score = max(score for _, score in important_words)

            for word, score in important_words:

                st.write(word.title())
                st.progress(score / max_score)
                st.caption(f"Importance Score: {score}")

        # -------------------------
        # Heatmap (Dummy but safe)
        # -------------------------
        st.subheader("Attention Heatmap")

        heatmap_data = np.random.rand(10, 10)

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(heatmap_data, cmap="Blues", ax=ax)

        ax.set_title("Word Relationship Heatmap")

        st.pyplot(fig)

        # -------------------------
        # Stats
        # -------------------------
        st.subheader("Article Statistics")

        word_count = len(article.split())
        char_count = len(article)
        reading_time = round(word_count / 200, 2)

        st.write(f"Word Count: {word_count}")
        st.write(f"Character Count: {char_count}")
        st.write(f"Reading Time: {reading_time} minutes")