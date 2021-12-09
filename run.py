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

# To predict the pattern category
def predict_category(pattern):
    bow = encode(pattern)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, j] for i, j in enumerate(res) if j > ERROR_THRESHOLD]
    result.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for item in result:
        return_list.append({'intent': classes[item[0]], 'probability': str(item[1])})
    return return_list

# To execute the response
def get_response(intents_list, intents_json):
    category = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for item in list_of_intents:
        if item['tag'] == category:
            result = random.choice(item['responses'])
            break
    return result

print("I'm Astro, I'm here to help you boost your mental health ...")

while True:
    message = input('')
    ints = predict_category(message)
    response = get_response(ints, intents)
    print(response)
