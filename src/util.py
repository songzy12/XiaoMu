import gensim
import io
import json
import jieba
import requests
import urllib

import pandas as pd
from matchzoo import DataPack


# model = gensim.models.Word2Vec.load('data/wiki.zh.text.jieba.model')


def es_reply(query):
    url = 'http://127.0.0.1:9200/robot_common/_search?q=question:{}'.format(
        urllib.quote(query.encode('utf8')))
    response = json.loads(requests.get(url).text)
    hits = response['hits'] if 'hits' in response else None
    if not hits or not hits['hits']:
        return []
    return [[x['_source']['question'], x['_score']] for x in hits['hits']]


def save(json_data, filename):
    with io.open(filename, 'w', encoding='utf8') as f:
        f.write(json.dumps(json_data, ensure_ascii=False, sort_keys=True, indent=4))


def prepare_data():
    text2id = {}

    label_file = '../data/label/match.json'
    candidate_file = '../data/label/candidate.json'

    with io.open(label_file, encoding='utf8') as f:
        m_match = json.loads(f.read())

    with io.open(candidate_file, encoding='utf8') as f:
        m_candidate = json.loads(f.read())

    left = []
    right = []
    train_relation = []
    test_relation = []

    num_candidate = 5

    test_portion = 0.2
    cnt = int((1-test_portion)*len(m_candidate))
    cur_cnt = 0

    for k, v in m_candidate.items():
        cur_cnt += 1

        if k not in m_match or not m_match[k]:
            continue
        if k not in text2id:
            text2id[k] = 't_%d' % (len(text2id))
            left.append([text2id[k], k])

        for _candidate in v[:num_candidate]:
            candidate = _candidate[0]

            if candidate not in text2id:
                text2id[candidate] = 't_%d' % (len(text2id))
                right.append([text2id[candidate], candidate])

            if candidate in m_match[k]:
                if cur_cnt < cnt:
                    train_relation.append([text2id[k], text2id[candidate], 1])
                else:
                    test_relation.append([text2id[k], text2id[candidate], 1])
            else:
                if cur_cnt < cnt:
                    train_relation.append([text2id[k], text2id[candidate], 0])
                else:
                    test_relation.append([text2id[k], text2id[candidate], 0])

        for _candidate in v[num_candidate:]:
            candidate = _candidate[0]
            if candidate not in text2id:
                text2id[candidate] = 't_%d' % (len(text2id))
                right.append([text2id[candidate], candidate])

            if candidate in m_match[k]:
                if cur_cnt < cnt:
                    train_relation.append([text2id[k], text2id[candidate], 1])
                else:
                    test_relation.append([text2id[k], text2id[candidate], 1])

    train_relation = pd.DataFrame(train_relation, columns=[
                                  'id_left', 'id_right', 'label'])
    test_relation = pd.DataFrame(test_relation, columns=[
                                 'id_left', 'id_right', 'label'])
    left = pd.DataFrame(left, columns=['id_left', 'text_left'])
    left.set_index('id_left', inplace=True)
    right = pd.DataFrame(right, columns=['id_right', 'text_right'])
    right.set_index('id_right', inplace=True)

    train = pd.merge(train_relation, left, on=['id_left'])
    train = pd.merge(train, right, on=['id_right'])
    train.to_csv('../data/train.csv')

    test = pd.merge(test_relation, left, on=['id_left'])
    test = pd.merge(test, right, on=['id_right'])
    test.to_csv('../data/test.csv')


def load_data(stage='train'):
    path = '../data/%s.csv' % stage
    data_pack = mz.pack(pd.read_csv(path, index_col=0))
    data_pack.relation['label'] = data_pack.relation['label'].astype('float32')


if __name__ == '__main__':
    prepare_data()
