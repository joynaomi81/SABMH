import streamlit as st
import openai
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Set the page configuration as the first command
st.set_page_config(page_title="Mental Health AI", page_icon="💬")

# Add OpenAI API key input field
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# Prompt user to input their OpenAI API key if not set
if st.session_state.openai_api_key == "":
    st.session_state.openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if st.session_state.openai_api_key:
        st.success("API Key successfully entered!")

# Set OpenAI API key
openai.api_key = st.session_state.openai_api_key

# Load the trained deep learning model
model = tf.keras.models.load_model("mental_health_model.h5")

# Load tokenizer
with open("tokenizer.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)  # Use the same maxlen as during training
    return padded

# Predict sentiment
def predict_sentiment(text):
    padded_input = preprocess_text(text)
    prediction = model.predict(padded_input)[0]
    sentiment_index = np.argmax(prediction)

    labels = ['Positive', 'Neutral', 'Negative']  # Adjust based on your model's output
    return labels[sentiment_index]

# Generate empathetic response using OpenAI
def get_openai_response(user_input, sentiment):
    prompt = f"""
    You're an empathetic mental health support chatbot.
    The user seems to be feeling {sentiment.lower()}. Respond with kindness, support, and encouragement.

    User said: "{user_input}"
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a kind and empathetic mental health chatbot."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("💬 Mental Health Companion (English Only)")
st.markdown("Type how you're feeling. The AI will listen and respond with support and care 💛")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("How are you feeling today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Step 1: Predict sentiment
    sentiment = predict_sentiment(user_input)

    # Step 2: Get empathetic response
    bot_response = get_openai_response(user_input, sentiment)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.chat_message("assistant").write(bot_response)

    # Optional: show sentiment type
    st.info(f"Detected Sentiment: **{sentiment}**")
