import io
import json

import matchzoo as mz

from util import load_data

model_classes = [
    mz.models.DUET,
    mz.models.ArcI,
    mz.models.ArcII,
    mz.models.ANMM,
    mz.models.MVLSTM,
]


train_pack = load_data('train')
valid_pack = load_data('test')

task = mz.tasks.Ranking(metrics=['mrr', 'map', 'ndcg'])
results = []
for model_class in model_classes:
    print('model_class', str(model_class))
    model, preprocessor, data_gen_builder, embedding_matrix = mz.auto.prepare(
        task=task,
        model_class=model_class,
        data_pack=train_pack
    )

    train_processed = preprocessor.fit_transform(train_pack)
    train_generator = mz.PairDataGenerator(
        train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)

    valid_processed = preprocessor.transform(valid_pack)
    valid_x, valid_y = valid_processed.unpack()
    evaluate = mz.callbacks.EvaluateAllMetrics(
        model, valid_x, valid_y,
        batch_size=len(valid_x),
    )
    history = model.fit_generator(train_generator,
                                  epochs=20, callbacks=[evaluate])
    results.append(
        {'model_class': str(model.params['model_class']), 'history': history})

for i, result in enumerate(results):
    result['history'] = result['history'].history
    for key in list(result['history'].keys()):
        result['history'][str(key)] = result['history'].pop(key)

print(json.dumps(results, ensure_ascii=False, indent=4))
with io.open('../log/history.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(results, ensure_ascii=False, indent=4))
