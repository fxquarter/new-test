#用于设计卷积神经网络模型并进行训练以实现场景分类
# 运行代码后在根路径下得到最终模型model_UCMerced_LandUse.h5及精度曲线和损失曲线。
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_dir = 'E:/classification/'
# 指定每一种数据的位置
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

CLASS = ('airplane', 'beach',  'forest', 'river', 'tenniscourt')
num = len(CLASS)
for i in range(num):
    train_class_dir = os.path.join(train_dir, CLASS[i])
    validation_class_dir = os.path.join(validation_dir, CLASS[i])

# 设计模型
model = tf.keras.models.Sequential([
    # 我们的数据是150x150而且是三通道的，所以我们的输入应该设置为这样的格式。
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.5),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 'sigmoid'
])
TF_ENABLE_ONEDNN_OPTS=0
model.summary()  # 打印模型相关信息

# 进行优化方法选择和一些超参数设置
# 因为只有两个分类。所以用2分类的交叉熵，使用RMSprop，学习率为0.001.优化指标为accuracy
model.compile(optimizer=RMSprop(learning_rate=0.001),
              # loss='binary_crossentropy',
              loss='categorical_crossentropy',
              metrics=['acc'])

# 数据处理
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=6.2,
    horizontal_flip=True, )

# Note that the validation data should not be augmented!（注意，不能增强验证数据)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    # This is the target directory （目标目录)
    train_dir,
    # All images will be resized to 150x150（将所有图像的大小调整为150x150)
    target_size=(256, 256),
    batch_size=20,
    class_mode='categorical')  # 多分类

# 生成验证集带标签的数据
validation_generator = test_datagen.flow_from_directory(validation_dir,  # 验证图片的位置
                                                        batch_size=20,  # 每一个投入多少张图片训练
                                                        # class_mode='binary',  # 设置我们需要的标签类型
                                                        class_mode='categorical',  # 多分类
                                                        target_size=(256, 256))  # 将图片统一大小

# 进行训练
history = model.fit(train_generator, validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=200,
                              validation_steps=50,
                              verbose=2)
# pip install h5py
model.save('E:/classification/model_UCMerced_LandUse.h5')

# 得到精度和损失值
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))  # 得到迭代次数
# 绘制精度曲线
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.legend(('Training accuracy', 'validation accuracy'))
plt.figure()

# 绘制损失曲线
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.legend(('Training loss', 'validation loss'))
plt.title('Training and validation loss')
plt.show()
