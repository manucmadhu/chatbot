from flask import Flask, request, jsonify, render_template
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import re
import random  # Import random to select random responses
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# Load the intents file
with open("chatbot/intents.json") as file:
    intents = json.load(file)

# Load the trained model
model = load_model("chatbot_model.h5")

# Load tokenizer and label encoder
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pickle", "rb") as enc:
    lbl_encoder = pickle.load(enc)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('user_input')
    if not user_input:
        return jsonify({"error": "No user_input provided"}), 400
    
    try:
        response = chatbot_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def chatbot_response(user_input):
    # First, try pattern matching
    response = match_intent(user_input, intents)
    
    if response is None:
        # If no match found, fall back to ML model prediction
        predictions = predict_class(user_input)
        tag = get_response(predictions, lbl_encoder)
        response = get_response_message(tag)
    
    return response

def preprocess_input(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

def match_intent(user_input, intents):
    user_input = preprocess_input(user_input)  # Preprocess the input
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if re.search(r"\b" + pattern.lower() + r"\b", user_input):
                return random.choice(intent["responses"])  # Select a random response
    return None

def predict_class(user_input):
    padded_sequences = preprocess_input_for_model(user_input)
    predictions = model.predict(padded_sequences)
    return predictions

def preprocess_input_for_model(user_input):
    # Convert user input into sequences for the ML model
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = pad_sequences(sequences, maxlen=94)  # Adjust maxlen according to model training
    return padded_sequences

def get_response(predictions, lbl_encoder):
    try:
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class = np.squeeze(predicted_class)
        tag = lbl_encoder.inverse_transform([predicted_class])[0]
        return tag
    except Exception as e:
        print(f"Error in get_response: {e}")
        return 'default'

def get_response_message(tag):
    # Updated response set with multiple responses for each tag
    responses = {
        'greeting': ['Hello! How can I assist you today?', 'Hi there! How can I help?', 'Hey! What can I do for you?'],
        'goodbye': ['Goodbye! Have a great day!', 'See you later!', 'Take care!'],
        'power_saving_tip': ['To save power, consider using energy-efficient appliances.', 
                             'Turn off lights when not in use to save electricity.', 
                             'Use a smart thermostat to manage your AC usage.'],
        'default': ['I am not sure how to respond to that.', 'Sorry, I didn’t understand that. Can you try again?']
    }
    return random.choice(responses.get(tag, responses['default']))  # Select a random response for the tag

if __name__ == "__main__":
    app.run(debug=True)
