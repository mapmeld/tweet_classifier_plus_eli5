import csv
from random import shuffle

import numpy as np

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from xgboost import XGBClassifier

from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import VectorizerMixin

import torch
from pytorch_transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium')

import eli5
from eli5.lime import TextExplainer

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
    if (len(x) % 100 == 0):
        print(len(x))
    if line[1] == 'True':
        y.append(0)
    else:
        y.append(1)

x = np.array(x)
y = np.array(y)

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

class V(VectorizerMixin):
  def fit (self, X, y=None):
    return self

  def transform (self, X):
    xout = []
    for row in X:
        input_ids = torch.tensor([tokenizer.encode(row)])
        words = model(input_ids)[0][0]
        average_word_vector = []
        for word in words:
            index = 0
            for word_block in word:
                if len(average_word_vector) == index:
                    average_word_vector.append(0)
                average_word_vector[index] += float(word_block)
                index += 1
        index = 0
        for word_block in average_word_vector:
            average_word_vector[index] /= float(len(words))
            index += 1
        xout.append(average_word_vector)
    return np.array(xout)
vectorizer = V()

for classifier in classifiers:
    print(classifier)
    gnb = classifier()

    pipe = make_pipeline(vectorizer, gnb)
    pipe.fit(x[:testcutoff], y[:testcutoff])



    y_predicted = pipe.predict_proba(x[testcutoff:])
    #print(classification_report(y[testcutoff:], y_predicted, target_names=['known weird', 'less weird']))

    te = TextExplainer(random_state=101, n_samples=500)
    te.fit('Green new deal is the best bro, bring it on', pipe.predict_proba)
    te.show_prediction(target_names=['known weird', 'less weird'])
