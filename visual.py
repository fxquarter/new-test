#用于可视化每个卷积层的特征图
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from tensorflow.keras.models import load_model

np.seterr(divide='ignore', invalid='ignore')
model = load_model('E:/classification/model_UCMerced_LandUse.h5')
# 让我们定义一个新模型，该模型将图像作为输入，并输出先前模型中所有层的中间表示。
successive_outputs = [layer.output for layer in model.layers[1:]]
# 进行可视化模型搭建，设置输入和输出
visualization_model = tf.keras.models.Model(inputs=model.inputs, outputs=successive_outputs)
# 选取一张随机的图片
img_path = 'E:/classification/train/beach/beach15.tif'  # 确保没有前导空格
img = load_img(img_path, target_size=(256, 256))  # 以指定大小读取图片
x = img_to_array(img)  # 变成array
x = x.reshape((1,) + x.shape)  # 变成 (1, 150, 150, 3)
# 统一范围到0到1
x /= 255
# 运行我们的模型，得到我们要画的图。
successive_feature_maps = visualization_model.predict(x)
# 选取每一层的名字
layer_names = [layer.name for layer in model.layers]
# 开始画图
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        # 只绘制卷积层，池化层，不画全连接层。
        n_features = feature_map.shape[-1]  # 最后一维的大小，也就是通道数，也是我们提取的特征数
        # feature_map的形状为 (1, size, size, n_features)
        size = feature_map.shape[1]
        # 我们将在此矩阵中平铺图像
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            # 后期处理我们的特征，使其看起来更美观。
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            # 我们把图片平铺到这个大矩阵中
            display_grid[:, i * size: (i + 1) * size] = x
        # 绘制这个矩阵
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))  # 设置图片大小
        plt.title(layer_name)  # 设置题目
        plt.grid(False)  # 不绘制网格线
        # aspect='auto'自动控制轴的长宽比。 这方面对于图像特别相关，因为它可能会扭曲图像，即像素不会是正方形的。
        # cmap='viridis'设置图像的颜色变化
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
