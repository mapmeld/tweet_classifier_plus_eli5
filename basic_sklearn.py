import csv
from random import shuffle

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
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
test_cutoff = int(0.85 * len(datarows))

index = 0
for line in datarows:
    index += 1
    if index < test_cutoff:
        x.append(line[0])
        if line[1] == 'True':
            y.append(0)
        else:
            y.append(1)
    else:
        x_test.append(line[0])
        if line[1] == 'True':
            y_test.append(0)
        else:
            y_test.append(1)

x = np.array(x)
x_test = np.array(x_test)
y = np.array(y)
y_test = np.array(y_test)

tfid = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
x = tfid.fit_transform(x).todense()
#print(x)
gnb = LogisticRegressionCV()
gnb.fit(x, y)

y_predicted = gnb.predict(x_test)
print(classification_report(y_test, y_predicted, target_names=['known weird', 'less weird']))

explained = eli5.sklearn.explain_prediction.explain_prediction_sklearn(gnb, x[0], vec=tfid, target_names=y)
print(explained)
