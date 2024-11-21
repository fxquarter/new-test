#预测最可能的索引
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

path = 'E:/classification/train/forest/forest20.tif'
model = load_model('E:/classification/model_UCMerced_LandUse.h5')
img = image.load_img(path, target_size=(256, 256))
x = image.img_to_array(img) / 255.0
# 在第0维添加维度变为1x256x256x3，和我们模型的输入数据一样
x = np.expand_dims(x, axis=0)
# 使用 predict 方法进行预测
classes1 = model.predict(x, batch_size=10)
print(classes1)
# 获取最可能的类别索引
ind = np.argmax(classes1, axis=1)[0]
print(ind)
# UCMerced_LandUse
CLASS = ('airplane', 'beach', 'forest', 'river', 'tenniscourt')
print("It is ", CLASS[ind])