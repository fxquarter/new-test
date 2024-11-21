#将数据集中的所有样本按比例划分为训练集、验证集和测试集
import os, random, shutil


def moveFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1  # 自定义抽取图片的比例，100张抽10张，0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.move(fileDir + '/' + name, tarDir + '/' + name)
    return


if __name__ == '__main__':
    train_dir = 'E:/classification/train/'
    validation_dir = 'E:/classification/validation/'
    test_dir = 'E:/classification/validation/'

    CLASS = ('airplane', 'beach',  'forest', 'river', 'tenniscourt')
    num = len(CLASS)
    for i in range(num):
        fileDir = os.path.join(train_dir, CLASS[i])
        tarDir1 = os.path.join(validation_dir, CLASS[i])
        moveFile(fileDir, tarDir1)
        tarDir2 = os.path.join(test_dir, CLASS[i])
        moveFile(fileDir, tarDir2)
