from flask import Flask, render_template, url_for, request
import os
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import time

app = Flask(__name__)

# Path to the model file
MODEL_PATH = '/shared/models/linear_regression_model.pkl'

def remove_pattern(input_txt, pattern):
    '''
    Removes pattern from input_txt using regex
    '''
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    # Removes punctuations
    input_txt = re.sub(r'[^\w\s]', ' ', input_txt)

    return input_txt.strip().lower()

@app.route('/')
def home():
    global cv, clf

    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        return render_template('waiting.html')  # Display waiting page

    # Load the model if not already loaded
    if 'cv' not in globals() or 'clf' not in globals():
        with open(MODEL_PATH, 'rb') as f:
            cv, clf = pickle.load(f)

    return render_template('home.html')  # Load the home page

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        clean_test = remove_pattern(message, "@[\w]*")
        tokenized_clean_test = clean_test.split()
        stem_tokenized_clean_test = [stemmer.stem(i) for i in tokenized_clean_test]
        message = ' '.join(stem_tokenized_clean_test)
        data = [message]
        data = cv.transform(data)
        my_prediction = clf.predict(data)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':

    # Initialize stemmer
    stemmer = PorterStemmer()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000)
