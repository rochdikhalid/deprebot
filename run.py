import pickle, json, random
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Instances
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

# Setting up the data
with open('../data/root.json') as file:
    intents = json.load(file)

# To import the preprocessed dependencies
data = pickle.load(open( "../data/training_data", "rb" ) )
words = data['tokenized_words']
classes = data['categories']
x_train = data['x_train']
y_train = data['y_train']

# To load the model
model = load_model('../models/model.h5')

