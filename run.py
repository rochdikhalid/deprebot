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

# To clean up the pattern using some NLP techniques
def clean_up(pattern):
    pattern_of_words = tokenizer.tokenize(pattern)
    pattern_of_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_of_words]
    return pattern_of_words

# Encoding: "Bag of words" method
def encode(pattern):
    pattern_of_words = clean_up(pattern)
    bag = [0] * len(words)
    for w in pattern_of_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
