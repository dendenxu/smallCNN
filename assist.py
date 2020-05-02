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
model_path = "results/best9.h5"
model_assist_path = "results/best.h5"

# 加载模型，如果采用keras框架训练模型，则 model=load_model(model_path)
# model = load_model(model_path)
model_assist = load_model(model_assist_path)
height, width = 128, 128
height_a, width_a = 256, 256


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
    img_a = cv2.resize(img, (height_a, width_a))
    img = cv2.resize(img, (width, height))
    print(img.shape)
    # img = img.astype(int)
    # img = img[:, :, ::-1] * 1. / 255
    img = img * 1. / 255
    img_a = img_a[:, :, ::-1] * 1. / 255
    img = np.expand_dims(img, axis=0)
    img_a = np.expand_dims(img_a, axis=0)
    labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    # labels = ['plastic', 'metal', 'glass', 'trash', 'paper', 'cardboard']
    # labels = ["paper", "glass", "plastic", "metal", "cardboard", "trash"]
    # 获取输入图片的类别
    # y_predict = model.predict(img)
    y_predict_a = model_assist.predict(img_a)
    y_predict_a[:, 0] = y_predict_a[:, 5]
    y_predict_a[:, 2] = y_predict_a[:, 3]
    y_predict_a[:, 3] = y_predict_a[:, 0]
    y_predict_a[:, 4] = y_predict_a[:, 2]
    # y_predict = labels[np.argmax(y_predict+y_predict_a)]
    y_predict = labels[np.argmax(y_predict_a)]

    # -------------------------------------------------------------------------

    # 返回图片的类别
    return y_predict
