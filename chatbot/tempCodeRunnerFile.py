from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import json
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

# Initialize stemmer
stemmer = LancasterStemmer()

# Load or create data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Process intents file
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = [0 for _ in range(len(words))]
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag[words.index(w)] = 1

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Build model
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load or train model
try:
    model.load_weights("model.weights.h5")
except (FileNotFoundError, ValueError):
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
    model.save_weights("model.weights.h5")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        if se in words:
            bag[words.index(se)] = 1
    return np.array(bag)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    try:
        # Predict the response
        results = model.predict(np.array([bag_of_words(user_input, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        # Find the response
        responses = next((tg['responses'] for tg in data["intents"] if tg['tag'] == tag), ["Sorry, I don't understand."])

        return jsonify({"response": random.choice(responses)})

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"response": f"An error occurred: {str(e)}"})
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."


if __name__ == "__main__":
    app.run(debug=True)
