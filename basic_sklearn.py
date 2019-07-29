import csv
from random import shuffle

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

import eli5

datarows = []
with open('../aoc_reply_set/all_tweets/bset_automl_2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    index = -1
    for line in csv_reader:
        # skipping header row
        index += 1
        if index > 0:
            datarows.append(line)

shuffle(datarows)

x = []
y = []
x_test = []
y_test = []

# memory plz
datarows = datarows[:10000]

test_cutoff = int(0.85 * len(datarows))

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

#gnb = LogisticRegressionCV()
gnb = RidgeClassifierCV()
gnb.fit(x[:testcutoff], y[:testcutoff])

y_predicted = gnb.predict(x[testcutoff:])
print(classification_report(y[testcutoff:], y_predicted, target_names=['known weird', 'less weird']))

explained = eli5.sklearn.explain_prediction.explain_prediction_sklearn(gnb, datarows[0][0], vec=tfid, target_names=['known weird', 'less weird'])
print(explained)
