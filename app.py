import re
import string

import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# IMPORTANT: spam_mail.pkl stores a sklearn Pipeline whose CountVectorizer
# was built with `preprocessor=wordopt`. Pickle only stores a *reference*
# to that function by name, so the exact same function must exist in this
# file (with this exact name) before the model is unpickled, or loading
# will fail.
# ---------------------------------------------------------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


st.set_page_config(page_title="Spam Mail Classifier", page_icon="📧", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("spam_mail.pkl")


model = load_model()

st.title("📧 Spam Mail Classifier")
st.write("Paste an email or message below and check whether it's spam or not.")

text_input = st.text_area("Email / message text", height=200, placeholder="Type or paste the message here...")

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Check", type="primary", use_container_width=True)

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        prediction = model.predict([text_input])[0]
        label = "🚨 Spam" if int(prediction) == 1 else "✅ Not Spam (Ham)"

        try:
            proba = model.predict_proba([text_input])[0]
            confidence = max(proba)
            st.subheader(label)
            st.progress(float(confidence))
            st.caption(f"Confidence: {confidence * 100:.1f}%")
        except AttributeError:
            # Fallback if the underlying classifier has no predict_proba
            st.subheader(label)

        with st.expander("See processed text used by the model"):
            st.code(wordopt(text_input))

st.divider()
st.caption("Model: scikit-learn Pipeline (CountVectorizer + RandomForestClassifier)")
