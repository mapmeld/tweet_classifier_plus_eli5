import csv, json
from random import shuffle

from flask import Flask, request
from flask_cors import CORS

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.metrics import classification_report

from eli5 import explain_prediction, format_as_text

positives = []
negatives = []
rowcutoff = 10000

with open('bset_automl_2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    index = -1
    for line in csv_reader:
        # skipping header row
        index += 1
        if index > 0:
            if line[1] == 'True':
                positives.append(line)
            else:
                negatives.append(line)

datarows = positives[:int(rowcutoff / 2)]
datarows += negatives[:(rowcutoff - len(datarows))]

shuffle(datarows)
print(datarows[0])

x = []
y = []
testcutoff = int(0.85 * len(datarows))

for line in datarows:
    x.append(line[0])
    if line[1] == 'True':
        y.append(0)
    else:
        y.append(1)

x = np.array(x)
y = np.array(y)

tfid = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
x = tfid.fit_transform(x).todense()

gnb = RidgeClassifierCV()
gnb.fit(x[:testcutoff], y[:testcutoff])

y_predicted = gnb.predict(x[testcutoff:])
print(classification_report(y[testcutoff:], y_predicted, target_names=['known weird', 'less weird']))


app = Flask(__name__)
CORS(app)

@app.route('/tweet', methods=['POST'])
def result():
    tweet = str(request.data)
    explain = explain_prediction(gnb, tweet, vec=tfid, target_names=['known weird', 'less weird'])
    return str(format_as_text(explain))

if __name__ == '__main__':
    app.run(debug=True)
