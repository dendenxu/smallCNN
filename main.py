import glob
import os
import cv2
import re
import numpy as np

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam


def processing_data(data_path, height, width, batch_size=128, validation_split=0.1):
    """
    数据处理
    :param data_path: 带有子目录的数据集路径
    :param height: 图像形状的行数
    :param width: 图像形状的列数
    :param batch_size: batch 数据的大小，整数，默认128。
    :param validation_split: 在 0 和 1 之间浮动。用作测试集的训练数据的比例，默认0.1。
    :return: train_generator, validation_generator: 处理后的训练集数据、验证集数据
    """

    train_data = ImageDataGenerator(
        # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
        rescale=1. / 255,
        # 浮点数，剪切强度（逆时针方向的剪切变换角度）
        shear_range=0.2,
        # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        zoom_range=0.2,
        # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        width_shift_range=0.2,
        # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        height_shift_range=0.2,
        # 布尔值，进行随机水平翻转
        horizontal_flip=True,
        # 布尔值，进行随机竖直翻转
        vertical_flip=True,
        # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
        validation_split=validation_split
    )

    # 接下来生成测试集，可以参考训练集的写法
    validation_data = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=validation_split)

    img_list = img_list = glob.glob(os.path.join(data_path, '*/*.jpg'))
    x = np.array(
        [np.array([cv2.resize(cv2.imread(img_path), (width, height)), re.split("[\\\\|/]", img_path)[-2]]) for img_path in
         img_list])
    labels = set(x[:, 1])
    labels = list(labels)
    dic = {value: key for key, value in enumerate(labels)}
    y = to_categorical([dic[lis] for lis in x[:, 1]])
    x = np.stack(x[:, 0])

    train_generator = train_data.flow(x, y)
    validation_generator = validation_data.flow(x, y)

    return train_generator, validation_generator, img_list, y


def cnn_model(input_shape, learning_rate=1e-4):
    """
    该函数实现 Keras 创建深度学习模型的过程
    :param input_shape: 模型数据形状大小，比如:input_shape=(384, 512, 3)

    :return: 返回已经训练好的模型
    """
    # Input 用于实例化 Keras 张量。
    # shape: 一个尺寸元组（整数），不包含批量大小。 例如，shape=(32,) 表明期望的输入是按批次的 32 维向量。
    print(input_shape)
    inputs = Input(shape=input_shape)
    cnn = Conv2D(32, kernel_size=(3, 3), padding="same")(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPool2D()(cnn)
    cnn = Conv2D(32, kernel_size=(5, 5), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPool2D()(cnn)
    cnn = Conv2D(32, kernel_size=(5, 5), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPool2D()(cnn)
    cnn = Conv2D(64, kernel_size=(5, 5), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(64, kernel_size=(5, 5), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(64, kernel_size=(5, 5), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dropout(rate=0.1)(cnn)
    cnn = MaxPool2D()(cnn)
    cnn = Conv2D(64, kernel_size=(5, 5), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = MaxPool2D()(cnn)
    cnn = Conv2D(32, kernel_size=(3, 3))(cnn)
    cnn = Conv2D(32, kernel_size=(3, 3))(cnn)
    cnn = Conv2D(64, kernel_size=(3, 3))(cnn)
    cnn = Conv2D(128, kernel_size=(3, 3))(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPool2D()(cnn)
    cnn = Flatten()(cnn)

    # Dense 全连接层  实现以下操作：output = activation(dot(input, kernel) + bias)
    # 其中 activation 是按逐个元素计算的激活函数，kernel 是由网络层创建的权值矩阵，
    # 以及 bias 是其创建的偏置向量 (只在 use_bias 为 True 时才有用)。
    cnn = BatchNormalization()(cnn)
    cnn = Dense(64, activation="relu")(cnn)
    cnn = Dropout(0.2)(cnn)
    cnn = Dense(64, activation="relu")(cnn)
    # 批量标准化层: 在每一个批次的数据中标准化前一层的激活项， 即应用一个维持激活项平均值接近 0，标准差接近 1 的转换。
    # axis: 整数，需要标准化的轴 （通常是特征轴）。默认值是 -1
    # cnn = BatchNormalization(axis=-1)(cnn)
    # 将激活函数,输出尺寸与输入尺寸一样，激活函数可以是'softmax'、'sigmoid'等
    # cnn = Activation('sigmoid')(cnn)
    # Dropout 包括在训练中每次更新时，将输入单元的按比率随机设置为 0, 这有助于防止过拟合。
    # rate: 在 0 和 1 之间浮动。需要丢弃的输入比例。
    cnn = Dense(32, activation="relu")(cnn)
    cnn = Dense(32, activation="relu")(cnn)
    cnn = Dropout(0.2)(cnn)
    cnn = Dense(32, activation="relu")(cnn)
    cnn = Dense(32, activation="relu")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dense(6, activation="sigmoid")(cnn)

    outputs = cnn

    # 生成一个函数型模型
    model = Model(inputs=inputs, outputs=outputs)

    # 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
    model.compile(
        # 是优化器, 主要有Adam、sgd、rmsprop等方式。
        optimizer=Adam(lr=learning_rate),
        # 损失函数,多分类采用 categorical_crossentropy
        loss='categorical_crossentropy',
        # 是除了损失函数值之外的特定指标, 分类问题一般都是准确率
        metrics=['accuracy'])

    return model


def train(model, train_generator, validation_generator, train_n, val_n, epoch_n, model_save_path='results/cnn.h5',
          log_dir="results/logs/", batch_size=128):
    """
    :param model: 我们训练出的模型
    :param train_generator: 训练集
    :param validation_generator: 验证集
    :param model_save_path: 保存模型的路径
    :param log_dir: 保存模型日志路径
    :param train_n: 训练集大小
    :param val_n: 验证集大小
    :param epoch_n: 迭代次数
    :param batch_size: 批大小
    :return:
    """
    # 可视化，TensorBoard 是由 Tensorflow 提供的一个可视化工具。
    # tensorboard = TensorBoard(log_dir)

    # 训练模型, fit_generator函数:https://keras.io/models/model/#fit_generator
    # 利用Python的生成器，逐个生成数据的batch并进行训练。
    # callbacks: 实例列表。在训练时调用的一系列回调。详见 https://keras.io/callbacks/。
    history = model.fit_generator(
        # 一个生成器或 Sequence 对象的实例
        generator=train_generator,
        # epochs: 整数，数据的迭代总轮数。
        epochs=epoch_n,
        # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        steps_per_epoch=train_n // batch_size,
        # 验证集
        validation_data=validation_generator,
        # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
        validation_steps=val_n // batch_size,
        # callbacks=[tensorboard]
    )
    # 模型保存
    model.save(model_save_path)

    return history, model


def plot_training_history(res):
    """
    绘制模型的训练结果
    :param res: 模型的训练结果
    :return:
    """
    # 绘制模型训练过程的损失和平均损失
    # 绘制模型训练过程的损失值曲线，标签是 loss
    plt.plot(res.history['loss'], label='loss')

    # 绘制模型训练过程中的平均损失曲线，标签是 val_loss
    plt.plot(res.history['val_loss'], label='val_loss')

    # 绘制图例,展示出每个数据对应的图像名称和图例的放置位置
    plt.legend(loc='upper right')

    # 展示图片
    plt.show()

    # 绘制模型训练过程中的的准确率和平均准确率
    # 绘制模型训练过程中的准确率曲线，标签是 acc
    plt.plot(res.history['acc'], label='accuracy')

    # 绘制模型训练过程中的平均准确率曲线，标签是 val_acc
    plt.plot(res.history['val_acc'], label='val_accuracy')

    # 绘制图例,展示出每个数据对应的图像名称，图例的放置位置为默认值。
    plt.legend()

    # 展示图片
    plt.show()


def evaluate(history, save_model_path):
    pass


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    # This one is without the step_per_epoch
    # 获取数据名称列表
    height, width = 384, 512
    batch_size = 32
    epoch_n = 5
    validation_split = 0.05
    learning_rate = 1e-4
    # data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized"
    data_path = "./dataset-resized"
    model_save_path = "./results/cnn4.h5"  # 保存模型路径和名称
    check_point_dir = "./results/chkpt/"
    log_dir = "./results/logs"

    # 获取数据
    train_data, test_data, img_list, labels = processing_data(data_path, height, width, batch_size, validation_split)

    # 创建、训练和保存模型
    print((height, width))
    model = cnn_model((height, width, 3), learning_rate)
    model.summary()
    for i in range(20):
        history, model = train(model, train_data, test_data, len(img_list),
                               len(img_list), epoch_n, model_save_path, log_dir, batch_size)
        plot_training_history(history)
        model.save(check_point_dir + "cnn" + str(i) + ".h5")
        print("Check point model {} saved to {}".format("cnn" + str(i) + ".h5", check_point_dir))
    # 评估模型
    # evaluate(history, save_model_path)


if __name__ == '__main__':
    main()
