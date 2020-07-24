from core import create_dataset
import cv2


# 将mnist数据集中图片保存成.png格式，存放在mnist文件夹下
(train_x, train_y), (test_x, test_y) = create_dataset()

for index, pic in enumerate(test_x):
    print(pic.shape)
    cv2.imwrite('./mnist/%d.png'%index, pic*255)
