#用于在validation、test文件夹中分别创建5个类别的子文件夹
import os

CLASS = ('airplane', 'beach',  'forest', 'river', 'tenniscourt')  # 此处的类别名称可由get_filename.py文件的运行结果得到
num = len(CLASS)
for i in range(num):
    validation_dir = 'E:/classification/validation/'
    test_dir = 'E:/classification/test/'

    os.mkdir(os.path.join(validation_dir, CLASS[i]))
    os.mkdir(os.path.join(test_dir, CLASS[i]))
