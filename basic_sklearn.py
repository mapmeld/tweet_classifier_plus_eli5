import csv
from random import shuffle

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, ElasticNetCV, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

import eli5

positives = []
negatives = []
rowcutoff = 3000

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

# even numbers of positives and negatives
# if we don't have enough for 50% positive, negatives will fill to rowcutoff
datarows = positives[:int(rowcutoff / 2)]
datarows += negatives[:(rowcutoff - len(datarows))]

shuffle(datarows)

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

classifiers = [
    LogisticRegressionCV,
    RidgeClassifierCV,
    #Perceptron
    #PassiveAggressiveClassifier,
    # SGDClassifier,
    #LinearSVC, SVC, NuSVC

    #LinearDiscriminantAnalysis, # 0.50
#    KNeighborsClassifier, # 0.51
#    RandomForestClassifier, # 0.58
    #GaussianNB, # 0.575
#    MultinomialNB, # 0.62
#    BernoulliNB, # 0.62
    XGBClassifier, # 0.62
    #SVC # 0.5
    ]

# low accuracy is expected because of how I collected any Tweets by trolls
# my own classification is poor and I hope the model picks up on larger trends
# logistic regression: 0.61
# ridge classifier CV: 0.60
# elastic net: ValueError: Classification metrics can't handle a mix of binary and continuous targets
# lars
# lasso
# lassolars
# orthogonal
# perceptron        0.55
# passiveaggressive  0.555
# ridgeCV
# sgdclassifier  0.58
# linear SVC     0.58
# SVC
# NuSVC 0.59

for classifier in classifiers:
    print(classifier)
    gnb = classifier()
    gnb.fit(x[:testcutoff], y[:testcutoff])

    y_predicted = gnb.predict(x[testcutoff:])
    print(classification_report(y[testcutoff:], y_predicted, target_names=['known weird', 'less weird']))

    if classifier == XGBClassifier:
        explained = eli5.xgboost.explain_prediction(gnb, datarows[0][0], vec=tfid, target_names=['known weird', 'less weird'])
    else:
        explained = eli5.sklearn.explain_prediction.explain_prediction_sklearn(gnb, datarows[0][0], vec=tfid, target_names=['known weird', 'less weird'])
    print(explained)
