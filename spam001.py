import streamlit as st
import pickle

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

# Load model and vectorizer
try:
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

st.title("📧 Spam Email Detection")
st.write("Detect whether an email/message is **Spam** or **Ham (Not Spam)** using Machine Learning.")
st.markdown("---")

email = st.text_area(
    "✉️ Enter Email/Message",
    height=200,
    placeholder="Type or paste a message here...\n\nExample: Congratulations! You won 5000 rupees"
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if predict_btn:
    if email.strip() == "":
        st.warning("⚠️ Please enter an email message.")
    else:
        with st.spinner("Analyzing..."):
            email_vector = vectorizer.transform([email])
            prediction = model.predict(email_vector)[0]
            probability = model.predict_proba(email_vector)
            confidence = probability.max() * 100

        st.markdown("---")
        st.subheader("📊 Result")

        if prediction == "spam":
            st.error("🚨 This Message is SPAM")
            st.progress(confidence/100)
        else:
            st.success("✅ This Message is NOT SPAM - It's HAM")
            st.progress(confidence/100)

        st.write(f"**Confidence: {confidence:.2f}%**")

st.sidebar.title("About Project")
st.sidebar.info("""
### Spam Email Detection

This app predicts whether an email is:

✅ **Ham** = Not Spam - Safe
🚨 **Spam** = Unwanted/Promotional

**Machine Learning Algorithm:**
- Multinomial Naive Bayes

**Feature Extraction:**
- TF-IDF Vectorizer

**Tech Stack:**
- Python
- Scikit-learn
- Streamlit
""")
st.sidebar.markdown("---")
st.sidebar.write("Made for B.Tech ML Project")
st.sidebar.write("📍 Turki, Bihar")