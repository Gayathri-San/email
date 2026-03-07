import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("email_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Max length used during training
MAX_LEN = 50

st.title("📧 Email Spam Detection")

st.write("Enter an email message to check whether it is Spam or Not Spam")

# Input box
email_text = st.text_area("Enter Email Text")

if st.button("Predict"):

    if email_text.strip() == "":
        st.warning("Please enter some text")

    else:
        # Text preprocessing
        seq = tokenizer.texts_to_sequences([email_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Prediction
        prediction = model.predict(padded)

        if prediction[0][0] > 0.5:
            st.error("Not spam")
        else:
            st.success("Spam")