import cv2
import numpy as np


# 这个API用于处理用户自己拍摄的图片
# 使之符合模型的要求
def process_image(image_path, reverse=False, threshold=60, show_origin=False):
    # 从用户给定的路径读取图片
    image = cv2.imread(image_path)
    w, h, c = image.shape
    # 如果需要可以显示图中心800*800大小的图片
    if show_origin:
        _image = image[int(w / 2 - 400):int(w / 2 + 400), int(h / 2 - 400):int(h / 2 + 400), :]
        _image = cv2.resize(_image, (280, 280), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('origin iamge', _image)
    # 将图片转换到28*28大小
    image = cv2.resize(
        # 取原图中心800*800大小的图片
        image[int(w / 2 - 400):int(w / 2 + 400), int(h / 2 - 400):int(h / 2 + 400), :],
        (28, 28),
        interpolation=cv2.INTER_AREA)
    # 将图像转为灰度图
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    print('二值化阈值', ret)
    # 如果resverse为True则对图片进行像素灰度反转
    # 因为实际的手写数字是白纸黑字, 但是mnist数据集中不是
    # 在mnist数据集中, 我们可以看到训练用的数据集中背景是0,
    # 笔画部分是0 ~ 255之间, 所以白纸黑字的图片需要取反
    if reverse:
        image = cv2.bitwise_not(image)
    # 图片简单预处理, 这里最好和mnist数据集的处理方式相同
    image = np.expand_dims(image, -1)
    return image
