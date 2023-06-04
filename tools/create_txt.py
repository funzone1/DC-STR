import pandas as pd
import os
import glob
for path in glob.glob("D:\\dataset\\GDUFS-MSTD6000\\有序\\有文本 - 副本\\image - 副本\\*.jpg"):
    with open("D:\\dataset\\GDUFS-MSTD6000\\有序\\有文本 - 副本\\gt-train.txt", "a", encoding='utf-8') as f:
        f.write(path + '\t')
        with open(path.replace('jpg', 'txt'), "r", encoding='utf-8') as f2:  # 打开文件
            for line in f2.readlines():
                line = line.strip('\n')
                list = line.split('"')
                if len(list) >= 2:
                    f.write(list[1])
            f.write('\n')




