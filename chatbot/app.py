from flask import Flask, request, jsonify, render_template
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import pickle
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
    predictions = predict_class(user_input)
    tag = get_response(predictions, lbl_encoder)
    response = get_response_message(tag)
    return response
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    print(f"User input: {user_input}")
    print(f"Matched intent: {detected_intent}")
    return text
def match_intent(user_input, intents):
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if re.search(r"\b" + pattern.lower() + r"\b", user_input.lower()):
                return intent["responses"]
    return ["Sorry, I don't understand."]


# def preprocess_input(user_input):
#     padded_sequences = tokenizer.texts_to_sequences([user_input])
#     padded_sequences = pad_sequences(padded_sequences, maxlen=94)  # Adjust maxlen as needed
#     return padded_sequences

def predict_class(user_input):
    padded_sequences = preprocess_input(user_input)
    predictions = model.predict(padded_sequences)
    return predictions

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
def get_response(predictions, lbl_encoder):
    try:
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class = np.squeeze(predicted_class)
        print("Predicted Class:", predicted_class)  # Debug print to check class index
        tag = lbl_encoder.inverse_transform([predicted_class])[0]
        print("Predicted Tag:", tag)  # Debug print to check the predicted tag
        return tag
    except Exception as e:
        print(f"Error in get_response: {e}")
        return 'default'

# def get_response(predictions, lbl_encoder):
#     try:
#         predicted_class = np.argmax(predictions, axis=1)
#         predicted_class = np.squeeze(predicted_class)
#         tag = lbl_encoder.inverse_transform([predicted_class])[0]
#         return tag
#     except Exception as e:
#         print(f"Error in get_response: {e}")
#         return 'default'  # Return default response on error

def get_response_message(tag):
    responses = {
        'greeting': 'Hello! How can I help you today?',
        'goodbye': 'Goodbye! Have a great day!',
        'power_saving_tip': 'To save power, consider using energy-efficient appliances and turning off lights when not in use.',
        'default': 'I am not sure how to respond to that.'
    }
    return responses.get(tag, responses['default'])

if __name__ == "__main__":
    app.run(debug=True)
