import glob
import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing import image
import datetime

try:
    from obs import ObsClient

    AK = 'IT0BUVSNXM64EU0XOGQ8'
    SK = 'j8dkml5YbEsiMZuqfs7CapoZQM5NxnumlfogeCpc'
    obs_endpoint = 'obs.cn-north-4.myhuaweicloud.com'
    obs_client = ObsClient(AK, SK, is_secure=True, server=obs_endpoint)

    bucket_name = "dataset.kaggle.garbage"
    source_path = "results"
    target_path = "garbage/" + source_path
except:
    pass


class SyncResultsWithOBS(Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            resp = obs_client.putFile(bucketName=bucket_name, objectKey=target_path, file_path=source_path)
            print(resp)
        except:
            pass


def processing_data(data_path, height, width, batch_size=128, validation_split=0.1, labels=None):
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
        shear_range=0.1,
        # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        zoom_range=0.1,
        # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        width_shift_range=0.1,
        # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        height_shift_range=0.1,
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
        [np.array([cv2.resize(cv2.imread(img_path), (width, height)),
                   re.split("[\\\\|/]", img_path)[-2]]) for img_path in img_list])
    if labels is None:
        labels = set(x[:, 1])
        labels = list(labels)
    dic = {value: key for key, value in enumerate(labels)}
    y = to_categorical([dic[lis] for lis in x[:, 1]])
    x = np.stack(x[:, 0])

    train_generator = train_data.flow(x, y, seed=0, subset="training", batch_size=batch_size)
    validation_generator = validation_data.flow(x, y, seed=0, subset="validation", batch_size=batch_size)

    # train_generator = train_data.flow_from_directory(
    #     # 提供的路径下面需要有子目录
    #     data_path,
    #     # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
    #     target_size=(height, width),
    #     # 一批数据的大小
    #     batch_size=batch_size,
    #     # "categorical", "binary", "sparse", "input" 或 None 之一。
    #     # 默认："categorical",返回one-hot 编码标签。
    #     class_mode='categorical',
    #     # 数据子集 ("training" 或 "validation")
    #     subset='training',
    #     seed=0)
    # validation_generator = validation_data.flow_from_directory(
    #     data_path,
    #     target_size=(height, width),
    #     batch_size=batch_size,
    #     class_mode='categorical',
    #     subset='validation',
    #     seed=0)
    # labels = train_generator.class_indices
    return train_generator, validation_generator, img_list, labels


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def cnn_model(input_shape, learning_rate=1e-4, depth=34, class_number=6):
    """
    该函数实现 Keras 创建深度学习模型的过程
    :param input_shape: 模型数据形状大小，比如:input_shape=(384, 512, 3)

    :return: 返回已经训练好的模型
    """
    # Input 用于实例化 Keras 张量。
    # shape: 一个尺寸元组（整数），不包含批量大小。 例如，shape=(32,) 表明期望的输入是按批次的 32 维向量。
    print(input_shape)
    model = resnet_v2(input_shape, depth, class_number)
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
          log_dir="results/logs/", best_path="results/best2.h5", batch_size=128):
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
    tensorboard = TensorBoard(log_dir)
    if int(keras.__version__.split(".")[1]) >= 3:
        model_chkpt = ModelCheckpoint(best_path, 'val_accuracy', 1, True, mode='max')
    else:
        model_chkpt = ModelCheckpoint(best_path, 'val_acc', 1, True, mode='max')
    callbacks = [tensorboard, model_chkpt, SyncResultsWithOBS()]
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
        callbacks=callbacks
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
    if int(keras.__version__.split(".")[1]) >= 3:
        # 绘制模型训练过程中的准确率曲线，标签是 acc
        plt.plot(res.history['accuracy'], label='accuracy')
        # 绘制模型训练过程中的平均准确率曲线，标签是 val_acc
        plt.plot(res.history['val_accuracy'], label='val_accuracy')
    else:
        # 绘制模型训练过程中的准确率曲线，标签是 acc
        plt.plot(res.history['acc'], label='accuracy')
        # 绘制模型训练过程中的平均准确率曲线，标签是 val_acc
        plt.plot(res.history['val_acc'], label='val_accuracy')

    # 绘制图例,展示出每个数据对应的图像名称，图例的放置位置为默认值。
    plt.legend()

    # 展示图片
    plt.show()


# def evaluate(model, validation_generator, labels):
#     """
#     加载模型、模型预测并展示模型预测结果等
#     :param validation_generator: 预测数据
#     :param labels: 数据标签
#     :return:
#     """
#
#     # 测试集数据与标签
#     test_x, test_y = validation_generator.__getitem__(2)
#
#     # 预测值
#     preds = model.predict(test_x)
#
#     # 绘制预测图像的预测值和真实值，定义画布
#     plt.figure(figsize=(16, 16))
#     for i in range(16):
#         # 绘制各个子图
#         plt.subplot(4, 4, i + 1)
#
#         # 图片名称
#         plt.title(
#             'pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
#
#         # 展示图片
#         plt.imshow(test_x[i])
#     plt.show()


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    # This one is without the step_per_epoch
    # 获取数据名称列表
    height, width = 256, 256
    batch_size = 32
    depth = 29  # depth should be 9n+2 (eg 56 or 110 in [b])
    class_number = 6
    epoch_n = 200
    validation_split = 0.05
    learning_rate = 1e-4
    data_path = "./dataset-resized"
    try:
        os.listdir(data_path)
    except:
        data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized"
    model_save_path = "./results/cnn10.h5"  # 保存模型路径和名称
    check_point_dir = "./results/chkpt/"
    log_dir = "./results/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    best_path = "./results/best4.h5"

    # 获取数据
    train_generator, test_generator, img_list, labels = processing_data(data_path, height, width, batch_size,
                                                                        validation_split)

    # 创建、训练和保存模型
    print((height, width))
    model = cnn_model((height, width, 3), learning_rate, depth, class_number)
    model.summary()
    history, model = train(model, train_generator, test_generator, len(img_list),
                           len(img_list), epoch_n, model_save_path, log_dir, best_path, batch_size)
    plot_training_history(history)
    print(labels)
    # 评估模型
    # evaluate(model, test_generator, labels)
    return train_generator, test_generator, img_list, labels


if __name__ == '__main__':
    train_generator, test_generator, img_list, labels = main()
