import streamlit as st
import joblib
import nltk

from nltk.corpus import stopwords


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🎭",
    layout="centered"
)


# =========================================
# DOWNLOAD NLTK STOPWORDS
# =========================================

nltk.download("stopwords", quiet=True)


# =========================================
# LOAD TRAINED MODEL FILES
# =========================================

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
emotion_labels = joblib.load("emotion_labels.pkl")


# =========================================
# STOPWORDS
# =========================================

stop_words = set(stopwords.words("english"))


# =========================================
# TEXT PREPROCESSING
# =========================================

def preprocess_text(txt):

    # Convert text to lowercase
    txt = txt.lower()

    # Remove numbers
    txt = ''.join(i for i in txt if not i.isdigit())

    # Remove emojis / non-ASCII characters
    txt = ''.join(i for i in txt if i.isascii())

    # Remove stopwords
    words = txt.split()

    cleaned = []

    for word in words:

        if word not in stop_words:
            cleaned.append(word)

    return ' '.join(cleaned)


# =========================================
# STREAMLIT UI
# =========================================

st.title("🎭 Emotion Detection")

st.write(
    "Enter a sentence and let the NLP model detect the emotion."
)


# =========================================
# TEXT INPUT
# =========================================

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am really happy today!",
    height=150
)


# =========================================
# DETECT EMOTION BUTTON
# =========================================

if st.button("🔍 Detect Emotion"):

    # Check if user entered anything
    if text.strip() == "":
        
        st.warning("⚠️ Please enter some text.")

    else:

        # -------------------------------------
        # STEP 1: Preprocess text
        # -------------------------------------

        cleaned_text = preprocess_text(text)


        # -------------------------------------
        # STEP 2: Convert text using TF-IDF
        # -------------------------------------

        transformed_text = vectorizer.transform(
            [cleaned_text]
        )


        # -------------------------------------
        # STEP 3: Predict emotion
        # -------------------------------------

        prediction = model.predict(
            transformed_text
        )[0]


        # -------------------------------------
        # STEP 4: Get probabilities
        # -------------------------------------

        probabilities = model.predict_proba(
            transformed_text
        )[0]


        # -------------------------------------
        # STEP 5: Convert number → emotion
        # -------------------------------------

        emotion = emotion_labels[prediction]


        # -------------------------------------
        # STEP 6: Calculate confidence
        # -------------------------------------

        confidence = max(probabilities) * 100


        # =====================================
        # DISPLAY RESULT
        # =====================================

        st.success(
            f"🎭 Detected Emotion: {emotion.upper()}"
        )


        # =====================================
        # DISPLAY CONFIDENCE
        # =====================================

        st.write(
            f"### Confidence: {confidence:.2f}%"
        )


        # =====================================
        # DISPLAY ALL EMOTION PROBABILITIES
        # =====================================

        st.write("### 📊 Emotion Probabilities")


        for i, probability in enumerate(probabilities):

            emotion_name = emotion_labels[i]

            percentage = probability * 100


            st.write(
                f"**{emotion_name.capitalize()} — "
                f"{percentage:.2f}%**"
            )


            st.progress(
                float(probability)
            )