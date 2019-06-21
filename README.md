代码位置 `zhilan:/home/songzy/Xiaomu-Matching`

## code

* data/
  * label/
    * candidate.json：对于每一条用户查询，由 es 给出来的备选答案
    * match.json: 对于每一条用户查询，标注的合适的相关答案
  * common.json: es 里存储的所有客服整理的问答对
* src/
  * metric.py: 各种无监督的距离计算方法实现
  * unsupervised.py: 无监督的模型效果
  * supervised.py: 神经网络模型效果，使用了 matchzoo 这个库
  * util.py: 数据准备相关
  * run.sh: 如何运行的一个示例

## run

```
cd src
python util.py
python supervised.py
python unsupervised.py
```