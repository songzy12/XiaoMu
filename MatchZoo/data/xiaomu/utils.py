import code
import jieba
import json
import gensim
import math

data_dir = 'classification/'

def get_embed(word_dict):
    embed_dir = '/home/songzy/word2vec/wiki.zh.text.jieba.100.model'
    embed_dir = './texts.100.model'
    model = gensim.models.Word2Vec.load(embed_dir)
    
    with open(data_dir+'embed_word2vec_d100', 'w') as f:
        for k, v in word_dict.items():
            if k not in model:
                continue
            temp = model[k] / math.sqrt(model[k].dot(model[k]))
            f.write(str(v)+' '+' '.join(map(str, temp))+'\n')

def get_tfidf_(text_files):
    texts0 = []
    texts1 = []
    for text_file in text_files:
        with open(text_file) as f:
            temp = json.loads(f.read())
            for k, v in temp.items():
                if v != 0:
                    if k in texts1:
                        continue
                    texts1.append(k)
                else:
                    if k in texts0:
                        continue
                    texts0.append(k) 

    tfidf = {}
    texts0 = list(map(jieba.lcut, texts0))
    texts1 = list(map(jieba.lcut, texts1))
    texts = texts0 + texts1

    for text in texts0:
        for word in text:
            tf = len([x for x in texts0 if word in x]) + 1
            idf = 1.0 / (len([x for x in texts1 if word in x]) + 1)
            if word not in tfidf:
                # we make it to be between (1, 2)
                tfidf[word] = 1 + (1.0 / (1 + math.exp(-tf * idf)) - 0.5) / 0.5 
            else:
                continue

    import operator
    tfidf = sorted(tfidf.items(), key=operator.itemgetter(1))

    with open(data_dir+'category-tfidf_.txt', 'w') as f:
        for k, v in tfidf:
            f.write(k+' '+str(v)+'\n')

    return tfidf

def get_tfidf():
    m = {}
    with open(data_dir+'category-tfidf_.txt') as f:
        for line in f.readlines():
            try:
                word, weight = line.strip().split()
            except:
                continue
                code.interact(local=locals())
            m[word] = weight
    m_ = {}
    with open(data_dir+'word_dict.txt') as f:
        for line in f.readlines():
            try:
                word, id_ = line.split()
            except:
                continue
            try:
                m_[id_] = m[word]
            except:
                m_[id_] = 1
    with open(data_dir+'category-tfidf.txt','w') as f:
        for k, v in m_.items():
            f.write(k+" "+str(v)+'\n')
    
def get_corpus(texts, output=data_dir+'corpus.txt'):
    # text to id
    corpus = {}
    for text in texts:
        if text not in corpus:
            corpus[text] = len(corpus)
    with open(output, 'w') as f:
        for k, v in corpus.items():
            f.write('T'+str(v) + ' '+' '.join(jieba.lcut(k))+'\n')
    return corpus

def get_corpus_preprocessed(corpus, word_dict, output=data_dir+'corpus_preprocessed.txt'):
    # text id to bunch of word ids
    with open(output, 'w')  as f:
        for k, v in corpus.items():
            f.write('T'+str(v)+' '+str(len(k))+' '+' '.join([str(word_dict[x]) for x in jieba.lcut(k)])+'\n')

def get_relation(queries, match, corpus, output=data_dir+'relation_train.txt'):
    with open(output, 'w')  as f:
        for k, v in match.items():
            if k[0] not in queries:
                continue
            f.write(str(v)+' T'+str(corpus[k[0]])+' T'+str(corpus[k[1]])+'\n')

def get_sample(match, output=data_dir+'sample.txt'):
    # text sample of pos case and neg case
    with open(output, 'w') as f:
        for k, v in match.items():
            f.write(str(v)+'\t'+k[0]+'\t'+k[1]+'\n')

def get_text(text_files):
    texts = []
    for text_file in text_files:
        with open(text_file) as f:
            temp = json.loads(f.read())
            for k, v in temp.items():
                if v != 0:
                    continue
                #if k not in match:
                #    print(text_file)
                texts.append(k)
    return texts

def get_candidates():
    # returns a dict with {question: answer} of es common
    candidate_path = '/home/songzy/mu_qa/xiaomu_data/es/common.json'
    m = {}
    with open(candidate_path) as f:
        line = f.readline()
        while line:
            line = f.readline()
            temp = json.loads(line)
            m[temp['question']] = temp['answer']
            line = f.readline()
    return m

def get_pos_neg():
    # pos case and neg case from match label and candidates
    with open('es_scores.json') as f:
        m_ = json.loads(f.read())
    with open('match_label.json') as f:
        m = json.loads(f.read())
    
    match = {}

    for k, vs in m_.items():
        if k not in m or not m[k]:
            continue
        for v in vs:
            if v[0] in m[k]:
                match[(k, v[0])] = 1
            else:
                match[(k, v[0])] = 0 
        for v_ in m[k]:
            match[(k, v_)] = 1 
    print('pos:', len({k:v for k,v in match.items() if v}), 'neg:', len({k:v for k,v in match.items() if not v}))
    return match

def get_word_dict(texts):
    # word to id
    word_dict = {}
    for text in texts:
        for word in jieba.cut(text):
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    with open(data_dir+'word_dict.txt', 'w') as f:
        for k, v in word_dict.items():
            f.write(k + ' ' + str(v) + '\n')
    return word_dict
    

if __name__ == '__main__':
    
    text2_corpus = ['label_common.json',
                 'label_0309-0414.json',
                 'label_0414-0510.json',
                 'label_0510-0609.json',
                 'label_0609-0622.json',
                 'label_0622-0716.json',
                 'label_0717-0728.json',
                 'label_0729-0807.json',
                 'label_0808-0903.json',
                 'label_2017-06-05_2017-07-02.json',
                 'label_2017-07-03_2017-08-06.json',
                 'label_2017-08-07_2017-09-03.json',
                 #'label_2017-09-04_2017-10-01.json',
                 #'label_2017-10-02_2017-11-05.json',
                 #'label_2017-11-06_2017-12-03.json',
                ]
    qm_dir = '/home/songzy/mu_qa/xiaomu_data/question_label/'
    text_train = get_text([qm_dir + x for x in text2_corpus[:-2]])
    text_valid = get_text([qm_dir + x for x in text2_corpus[-2:-1]])
    text_test = get_text([qm_dir + x for x in text2_corpus[-1:]])

    texts = map(jieba.lcut, text_train+text_valid+text_test)

    with open('texts', 'w') as f:
        for text in texts:
            f.write(' '.join(text)+'\n')

    print('train, valid, test:', len(text_train), len(text_valid), len(text_test))
    
    candidates = list(get_candidates().keys())
    print('candidates:', len(candidates))

    word_dict = get_word_dict(text_train+text_valid+text_test+candidates)
    corpus = get_corpus(text_train+text_valid+text_test+candidates)
    corpus_preprocessed = get_corpus_preprocessed(corpus, word_dict)

    get_embed(word_dict)

    match = get_pos_neg()
    get_sample(match)

    get_relation(text_train, match, corpus, output=data_dir+'relation_train.txt')
    get_relation(text_test, match, corpus, output=data_dir+'relation_test.txt')
    get_relation(text_valid, match, corpus, output=data_dir+'relation_valid.txt')

    get_tfidf_([qm_dir + x for x in text2_corpus])
    get_tfidf()
