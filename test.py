from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/dnn.h5'
model_path = "results/best2.h5"

# 加载模型，如果采用keras框架训练模型，则 model=load_model(model_path)
model = load_model(model_path)
height, width = 96, 128


# ---------------------------------------------------------------------------
def predict(img):
    """
    加载模型和模型预测
    主要步骤:
        1.图片处理
        2.用加载的模型预测图片的类别
    :param img: PIL.Image 对象
    :return: string, 模型识别图片的类别,
            共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 获取图片的类别，共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    # 把图片转换成为numpy数组
    img = image.img_to_array(img)
    img = np.array(img)
    print(img.shape)
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)
    if img.shape[0] > img.shape[1] // 4 * 3:
        img = img[img[1] // 8 + 1:-img[1] // 8 - 1]
    print(img.shape)
    img = cv2.resize(img, (width, height))
    print(img.shape)
    # img = img.astype(int)
    # img = img[:, :, ::-1] * 1. / 255
    img = np.expand_dims(img, axis=0)
    labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    # labels = ['plastic', 'metal', 'glass', 'trash', 'paper', 'cardboard']
    # labels = ["paper", "glass", "plastic", "metal", "cardboard", "trash"]
    # 获取输入图片的类别
    y_predict = model.predict(img)
    y_predict = labels[np.argmax(y_predict)]

    # -------------------------------------------------------------------------

    # 返回图片的类别
    return y_predict


# 输入图片路径和名称
img_path = 'glass.jpg'

# 打印该张图片的类别
img = image.load_img(img_path)
print(predict(img))


def plot_load_and_model_prediction(test_generator, labelsP, labelsO):
    """
    加载模型、模型预测并展示模型预测结果等
    :param validation_generator: 预测数据
    :param labels: 数据标签
    :return:
    """

    # 测试集数据与标签
    test_x, test_y = test_generator.__getitem__(2)

    # 预测值
    preds = model.predict(test_x)

    # 绘制预测图像的预测值和真实值，定义画布
    plt.figure(figsize=(16, 16))
    for i in range(16):
        # 绘制各个子图
        plt.subplot(4, 4, i + 1)

        # 图片名称
        plt.title(
            'pred:%s / truth:%s' % (labelsP[np.argmax(preds[i])], labelsO[np.argmax(test_y[i])]))

        # 展示图片
        plt.imshow(test_x[i])
    plt.show()
