# coding=utf8
import code
import gensim
import io
import json
import jieba
import requests
import urllib

model = gensim.models.Word2Vec.load('data/wiki.zh.text.jieba.model')


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


if __name__ == '__main__':
    query = u'证书'
    print(es_reply(query))
