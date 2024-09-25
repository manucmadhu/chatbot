from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import pickle

# Create and fit the Tokenizer
tokenizer = Tokenizer()
texts = ["sample text", "another example"]
tokenizer.fit_on_texts(texts)

# Save the Tokenizer
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create and fit the LabelEncoder
lbl_encoder = LabelEncoder()
import json

# Load intents file
with open("chatbot\intents.json") as file:
    intents = json.load(file)

# Extract classes (tags)
classes = [intent["tag"] for intent in intents["intents"]]
lbl_encoder.fit(classes)

# Save the LabelEncoder
with open("label_encoder.pickle", "wb") as enc:
    pickle.dump(lbl_encoder, enc, protocol=pickle.HIGHEST_PROTOCOL)
