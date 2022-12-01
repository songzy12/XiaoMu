from collections import defaultdict
import io
import json
import os


def merge_label_file(path_input, path_output):
    m = {}

    try:
        with open(path_output) as f:
            m.update(json.loads(f.read()))
    except:
        pass

    for file_ in path_input:
        with io.open("./data/label/%s" % file_, encoding='utf8') as f:
            m.update(json.loads(f.read()))

    print(path_output, len(m))
    with io.open(path_output, 'w', encoding='utf8') as f:
        f.write(json.dumps(m, ensure_ascii=False, indent=4))


def generate_dataset_sklearn(path_input='label/train_test.json', path_output='svm_train_test'):
    with io.open(path_input, encoding='utf8') as f:
        m = json.loads(f.read())
        m_ = defaultdict(list)

        for k, v in m.items():
            m_[v].append(k)

        for k, vs in m_.items():
            for i, v in enumerate(vs):
                filename = "%s/%s/%s.txt" % (path_output, str(k), str(i))
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except:
                        raise
                with io.open(filename, "w", encoding='utf8') as f:
                    f.write(v)


def prepare_sklearn():
    train_files = ['label_course.json',
                   'label_common.json',
                   'label_2017-07.json',
                   'label_2017-08.json',
                   'label_2017-09.json',
                   'label_2017-10.json',
                   'label_2017-11.json',
                   'label_2017-12.json',
                   'label_2018-01.json',
                   'label_2018-02.json',
                   'label_2018-03.json',
                   'label_2018-04.json',
                   'label_2018-05.json',
                   'label_2018-06.json',
                   'label_2018-07.json',
                   'label_2018-08.json',
                   'label_2018-09.json',
                   'label_2018-10.json',
                   'label_2018-11.json',
                   'label_2018-12.json',
                   ]

    test_files = ['label_2019-01.json', 'label_2019-02.json']

    merge_label_file(
        path_input=test_files + train_files, path_output='./data/svm/train_test.json')
    generate_dataset_sklearn(
        './data/svm/train_test.json', './data/svm/train_test')

    for label_file in train_files:
        print(label_file)
        with io.open("./data/label/%s" % label_file, encoding='utf8') as f:
            content = json.loads(f.read())
            m = defaultdict(list)
            for k, v in content.items():
                m[v].append(k)
            for k, v in m.items():
                print(k, len(v))

    merge_label_file(
        path_input=train_files, path_output='./data/svm/train.json')
    merge_label_file(
        path_input=test_files, path_output='./data/svm/test.json')

    generate_dataset_sklearn('./data/svm/train.json', './data/svm/train')
    generate_dataset_sklearn('./data/svm/test.json', './data/svm/test')


if __name__ == '__main__':
    prepare_sklearn()
