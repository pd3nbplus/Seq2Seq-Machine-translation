# Quick Start

环境说明：

+ python 3.6
+ torch Stable 1.11.0


```shell
pip install -r requirements.txt
```

数据集下载地址 http://www.statmt.org/wmt17/translation-task.html#download
选择 New Commentary v12进行下载即可

下载完后放在`training-parallel-nc-v12/training/`目录下即可

## Train

```shell
python train.py
```

# Machine-Translation项目

机器翻译是一个经典的Seq2Seq项目，本项目用于熟悉Pytorch Seq2Seq的Attention写法

目录结构

```shell
.
├── data  处理好的数据
│   ├── cn_datas.npy
│   ├── cn_dict.json
│   ├── en_datas.npy
│   └── en_dict.json
├── model  保存模型
│   ├── epoch_0.pth.tar
│   └── train.log
├── net.py  构建网络
├── preprocessing.py  数据预处理
├── __pycache__
│   ├── net.cpython-36.pyc
│   ├── preprocessing.cpython-36.pyc
│   └── utils.cpython-36.pyc
├── README.md
├── requirements.txt
├── seq2seq_attention_old.py 
├── training-parallel-nc-v12  数据集
│   └── training
│       ├── news-commentary-v12.cs-en.cs
│       ├── news-commentary-v12.cs-en.en
│       ├── news-commentary-v12.de-en.de
│       ├── news-commentary-v12.de-en.en
│       ├── news-commentary-v12.es-en.en
│       ├── news-commentary-v12.es-en.es
│       ├── news-commentary-v12.fr-en.en
│       ├── news-commentary-v12.fr-en.fr
│       ├── news-commentary-v12.ru-en.en
│       ├── news-commentary-v12.ru-en.ru
│       ├── news-commentary-v12.zh-en.en
│       └── news-commentary-v12.zh-en.zh
├── train.py  训练模型
├── translation.py  应用模型
└── utils.py  工具类
```


