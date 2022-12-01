这个是用来比较不同模型在我们标注的数据集上表现的代码。

## code

- `data/label/`: 手工标注的数据
- `sklearn_baseline`: 一些非神经网络的模型实现，如 svm, xgboost, lightgbm 等
- `util_data_sklearn.py`: 由 label 文件生成 sklearn 使用的输入格式
- `util_data_tf.py`: 由 label 文件生成 tensorflow 使用的输入格式

## sklearn_baseline

运行命令可见 `sklearn_baseline/run.sh`.

```
python baseline.py 2>&1 | tee ../log/baseline_$(date +%F).log
```

上面的命令会将各个模型的 performance 打印在 console 里，

并通过 tee 重定向至 log 文件。
