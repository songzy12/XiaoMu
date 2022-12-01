#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build a simple Question Classifier using TF-IDF or Bag of Words Model
"""

import io
import json
import joblib
import os

from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

from lightgbm import LGBMClassifier

import jieba
import jieba.posseg as pseg
import gensim

from constant import id2category


def _tokenize(text):
    return list(text)
    # return jieba.lcut(text)


def save_model(grid, filename='model/svm.pkl'):
    joblib.dump(grid, filename, compress=1)


def load_model(filename='model/svm.pkl'):
    grid = joblib.load(filename)
    return grid


def train(data_train='data/svm/train', data_test='data/svm/test'):
    # the training data folder must be passed as first argument
    dataset_train = load_files(data_train, shuffle=False)
    dataset_test = load_files(data_test, shuffle=False)
    print("train: %d, test: %d" %
          (len(dataset_train.data), len(dataset_test.data)))

    docs_train, docs_test = dataset_train.data, dataset_test.data
    y_train, y_test = dataset_train.target, dataset_test.target

    # split the dataset in training and test set:
    models = [SVC(probability=True, gamma='scale'),
              SGDClassifier(),
              KNeighborsClassifier(),
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              MultinomialNB(),
              XGBClassifier(objective='multi:softprob'),
              LGBMClassifier()
              ]

    for model in models:
        print(model)

        text_clf = Pipeline([('vect', CountVectorizer(tokenizer=_tokenize, ngram_range=(1, 3))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', model)])

        text_clf.fit(docs_train, y_train)

        y_predicted = text_clf.predict(docs_test)

        confusion = {}
        for i, q in enumerate(docs_test):
            if y_predicted[i] != y_test[i]:
                confusion[q.decode('utf8')] = {
                    'label': id2category[y_test[i]], 'prediction': id2category[y_predicted[i]]}
        with io.open('debug/confusion.json', 'w', encoding='utf8') as f:
            f.write(json.dumps(confusion, ensure_ascii=False, indent=4))

        # Print the classification report
        print(metrics.classification_report(y_test, y_predicted,
                                            target_names=dataset_test.target_names))

        # Print and plot the confusion matrix
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)

        save_model(text_clf)


def predict(clf, question):
    return clf.predict([question])[0]


def predict_proba(clf, question):
    return clf.predict_proba([question])


if __name__ == "__main__":
    train()
    grid = load_model()
    question = u'字符串的用法'
    print(predict(grid, question))
    print(predict_proba(grid, question))
