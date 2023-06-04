首先下载数据集
https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0
解压data_lmdb_release，放置目录下

train框架已经单独写好，直接run就行。
train.py --train_data data_lmdb_release\training --valid_data data_lmdb_release\validation

