import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load model and vectorizer
with open("phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="PhishGuard 🛡️", layout="centered")
st.title("🛡️ PhishGuard – Phishing Email Detector")
st.write("Paste an email message below to check if it's phishing or legitimate.")

email_input = st.text_area("📧 Email Text", height=250)

if st.button("🔍 Detect"):
    if not email_input.strip():
        st.warning("Please enter some email content.")
    else:
        input_vector = vectorizer.transform([email_input])
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)[0]

        phishing_prob = round(prediction_proba[1] * 100, 2)
        legit_prob = round(prediction_proba[0] * 100, 2)
        label = "⚠️ Phishing Email" if prediction == 1 else "✅ Legitimate Email"

        st.subheader(f"Result: {label}")
        st.info(f"🔢 Confidence: {max(phishing_prob, legit_prob)}%")

        # 📊 Add a confidence bar chart
        st.markdown("### 🔎 Confidence Distribution")
        fig, ax = plt.subplots()
        classes = ["Legitimate", "Phishing"]
        probs = [legit_prob, phishing_prob]
        colors = ["green", "red"]

        ax.bar(classes, probs, color=colors)
        ax.set_ylim([0, 100])
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)
