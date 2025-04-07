# SpamEmailBayesClassifier

[![wakatime](https://wakatime.com/badge/user/70908aa3-b2c6-4f44-a07f-7bd45f260e48/project/06fc3170-43dc-453e-80fb-984667472444.svg)](https://wakatime.com/badge/user/70908aa3-b2c6-4f44-a07f-7bd45f260e48/project/06fc3170-43dc-453e-80fb-984667472444)

基于 TF-IDF 及 Bayes 分类器的垃圾邮件分类器

数据集使用 2006 TREC Public Spam Corpora 中的 trec06c 中文垃圾邮件数据集作为实验数据集.

下载地址：<https://plg.uwaterloo.ca/~gvcormac/treccorpus06/>

## 文件内容说明

### 文件夹结构

- `Data`: 存放部分已经经过预处理的文本数据，以及停用词表
- `Model`: 存放训练好的模型
- `Report`: 存放提交的报告的 Typst 源文件以及模板


### 代码文件说明
- `BayesClassification.py`: 分类器具体实现（使用函数式编程的思想）
    - 模型的训练和推理
    - 模型的保存和加载
- `Data.py`: 数据加载函数，加载已经预先处理好的文本数据
- `rawdata.py`: 原始数据处理脚本，将原始数据转换预先处理好的词序列
- `trian.py`: 训练脚本，训练后保存模型
- `predict.py`: 预测脚本，使用训练好的模型对输入文本进行分类
- `analysis.py`: 模型评估并绘制混淆矩阵
- `topK.py`: 输出每个分类概率最高的前 K 个词
- `topSpam.py`: 输出最具有垃圾邮件特征的垃圾邮件

## 运行说明

本项目仅需要如下库（本项目并没有直接使用 sklearn 库，而是手动实现了 TF-IDF 以及 Bayes 分类器）：

- numpy
- matplotlib
- jieba

