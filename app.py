import openai
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Set up OpenAI API key (you can input it directly here or use Streamlit input)
openai.api_key = st.text_input("Enter OpenAI API Key", type="password")

# Load the mental health model (assuming it is an .h5 file)
model = tf.keras.models.load_model('mental_health_model.h5')

# Load the tokenizer used during training
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the sentiment encoder
with open('sentiment_encoder.pkl', 'rb') as handle:
    sentiment_encoder = pickle.load(handle)

# Function to predict sentiment (you can use other methods depending on your model)
def predict_sentiment(user_input):
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)
    prediction = model.predict(padded_sequences)
    sentiment = np.argmax(prediction, axis=1)
    return sentiment_encoder.inverse_transform(sentiment)[0]

# Function to get OpenAI response (now using the updated OpenAI API)
def get_openai_response(user_input, sentiment):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" depending on your preference
            messages=[
                {"role": "system", "content": "You are a helpful mental health assistant."},
                {"role": "user", "content": user_input},
                {"role": "system", "content": f"User seems to be feeling {sentiment}."}
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app layout
st.title("Mental Health AI Companion")
st.write("This chatbot uses sentiment analysis to understand your emotions and suggests mental wellness tips in your language.")

# Take user input
user_input = st.text_input("How are you feeling today?")

if user_input:
    # Predict the sentiment of the input text
    sentiment = predict_sentiment(user_input)
    
    # Get a response from OpenAI based on the sentiment and user input
    bot_response = get_openai_response(user_input, sentiment)
    
    # Display the response in Streamlit
    st.write("Bot Response: " + bot_response)

