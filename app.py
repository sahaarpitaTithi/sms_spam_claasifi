import streamlit as st
import pickle

# Load the model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.markdown("Enter a message to classify it as **Spam** or **Ham**.")

user_input = st.text_area("Message", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        vect_msg = vectorizer.transform([user_input])
        prediction = model.predict(vect_msg)[0]
        prob = model.predict_proba(vect_msg)[0]

        if prediction == "spam":
            st.error(f"🚨 Spam ({round(prob[1]*100, 2)}% confidence)")
        else:
            st.success(f"✅ Ham ({round(prob[0]*100, 2)}% confidence)")
