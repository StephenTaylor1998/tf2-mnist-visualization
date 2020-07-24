import cv2
import os
# 导入项目下的其他需要的包
from core import create_model
from core.predict import predict
from core import create_dataset
from core.photo_utils import process_image


# 导入数据集
(train_x, train_y), (test_x, test_y) = create_dataset()
# 导入模型结构
model = create_model()
# 导入模型数据
model.load_weights('./data/weights.h5')
# 用测试集验证模型, 验证权重是否正常导入
out = model.evaluate(test_x, test_y)

# 指定你想要进行预测的图片路径

# file_name = './data/my_images/5.jpg'
# file_name = './data/my_images/6.jpg'
file_name = './data/my_images/7.jpg'

# 读取图片文件并进行简单的预处理, 函数定义在core.photo_utils中
# 注意： mnist中的图片是黑底白字的, 而我们拍摄的一般是白底黑字
# 所以需要转换成黑底白字 函数中 reverse 用来控制是否取反
# 将图片二值化可以更好的进行预测, threshold 用来控制二值化时的阈值
# picture = process_image(file_name, reverse=False)
# 如果需要查看二值化前的原图可以使用 show_origin=True
picture = process_image(file_name, reverse=True, threshold=83, show_origin=True)
# 把读出来的图片放进模型里进行预测, 因为这个操作十分常见且通用, 所以建议封装
# 这里我们使用了一个封装, 输入图片和模型就能得到最后结果
# 函数的定义在core.predict文件中
out = predict(picture=picture, model=model, draw_in_console=False, print_confidence_rate=True)
# 把原来28*28大小的图片放大到280*280, 觉得小可以自己调大一点
large = cv2.resize(picture, (280, 280))
# 打印结果
print('输出结果', out)
# 创建窗口并显示出来
cv2.imshow('Result: %d(Next:press Space/Exit:press Esc)' % out, large)
# 这里用于获取键盘输入, 按下空格切换到下一张, 按下Esc键关闭程序
cv2.waitKey(0)
# 用于关闭前一个窗口, 防止创建过多的窗口
cv2.destroyAllWindows()
