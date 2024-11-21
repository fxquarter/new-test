#获取数据集中各类别的名称
import os

train_dir = 'E:/classification/train/'
for filename in os.listdir(train_dir):
    print(filename)
