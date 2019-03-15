import io
import json
from collections import defaultdict

import gensim
import numpy as np
import pandas as pd

from metric import _map, mrr, ndcg


model = gensim.models.Word2Vec.load(
    '../../data/wiki.zh.text.list.128.model')


def get_wmd(doc1, doc2):
    distance = model.wmdistance(doc1, doc2)
    return -distance


def get_dot(doc1, doc2):
    doc1 = filter(lambda x: x in model, list(doc1))
    doc2 = filter(lambda x: x in model, list(doc2))
    doc1 = list(map(lambda x: model[x], doc1))
    doc2 = list(map(lambda x: model[x], doc2))

    if not doc1 or not doc2:
        return 0

    vec1 = sum(doc1) / len(doc1)
    vec2 = sum(doc2) / len(doc2)
    return float(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def get_lcs(doc1, doc2):
    lengths = [[0 for j in range(len(doc2)+1)] for i in range(len(doc1)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(doc1):
        for j, y in enumerate(doc2):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(doc1), len(doc2)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert doc1[x-1] == doc2[y-1]
            result = doc1[x-1] + result
            x -= 1
            y -= 1
    return len(result)

# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python


def get_edit_distance(doc1, doc2):
    if len(doc1) < len(doc2):
        doc1, doc2 = doc2, doc1

    # So now we have len(doc1) >= len(doc2).
    if len(doc2) == 0:
        return -len(doc1)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    doc1 = np.array(tuple(doc1))
    doc2 = np.array(tuple(doc2))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(doc2.size + 1)
    for s in doc1:
        # Insertion (doc2 grows longer than doc1):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and doc1 items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
            current_row[1:],
            np.add(previous_row[:-1], doc2 != s))

        # Deletion (doc2 grows shorter than doc1):
        current_row[1:] = np.minimum(
            current_row[1:],
            current_row[0:-1] + 1)

        previous_row = current_row

    return -previous_row[-1]


def run(test_file):
    df = pd.read_csv(test_file)

    for dist_func in [get_lcs, get_edit_distance, get_dot, get_wmd]:
        metrics = defaultdict(list)
        for query in set(df['text_left']):
            candidates = df[df['text_left'] == query]['text_right']
            y_true = df[df['text_left'] == query]['label']
            y_pred = [dist_func(query, candidate)
                      for candidate in candidates]
            for metric in [mrr, _map, ndcg]:
                metrics[str(metric)].append(metric(y_true, y_pred))
        for metric in [mrr, _map, ndcg]:
            print("%s\t%s: %f" % (str(dist_func), str(metric), sum(
                metrics[str(metric)])/len(metrics[str(metric)])))


if __name__ == '__main__':
    test_file = '../data/test.csv'
    run(test_file)
