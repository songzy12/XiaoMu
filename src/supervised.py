import matchzoo as mz
from util import load_data

import bokeh
from bokeh.plotting import figure
from bokeh.io import export_png
from bokeh.layouts import column
from bokeh.models.tools import HoverTool

model_classes = [
    mz.models.CDSSM,
    mz.models.DSSM,
    mz.models.DUET,
    mz.models.DenseBaseline,
    mz.models.ArcI,
    mz.models.ArcII,
    mz.models.MatchPyramid,
    mz.models.DRMM,
    mz.models.ANMM,
    mz.models.MVLSTM
]


train_data_pack = load_data('train')
test_data_pack = load_data('test')

task = mz.tasks.Ranking(metrics=['map', 'mrr', 'ndcg'])
results = []
for model_class in model_classes:
    print(model_class)
    model = model_class()
    model.params['task'] = task
    model_ok, train_ok, preprocesor_ok = mz.auto.prepare(
        model=model,
        data_pack=train_data_pack,
        verbose=0
    )
    test_ok = preprocesor_ok.transform(test_data_pack, verbose=0)
    callback = mz.engine.callbacks.EvaluateAllMetrics(
        model_ok,
        *test_ok.unpack(),
        batch_size=1024,
        verbose=0
    )
    history = model_ok.fit(*train_ok.unpack(), batch_size=32,
                           epochs=32, callbacks=[callback])
    results.append({'name': model_ok.params['name'], 'history': history})

charts = {
    metric: figure(
        title=str(metric),
        sizing_mode='scale_width',
        width=800, height=400
    ) for metric in results[0]['history'].history.keys()
}
hover_tool = HoverTool(tooltips=[
    ("x", "$x"),
    ("y", "$y")
])
for metric, sub_chart in charts.items():
    lines = {}
    for result, color in zip(results, bokeh.palettes.Category10[10]):
        x = result['history'].epoch
        y = result['history'].history[metric]
        lines[result['name']] = sub_chart.line(
            x, y, color=color, line_width=2, alpha=0.5, legend=result['name'])
        sub_chart.add_tools(hover_tool)

export_png(column(*charts.values()), "quick_start_chart.png")
