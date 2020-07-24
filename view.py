import cv2
import os
# 导入项目下的其他需要的包
from core import create_model
from core.predict import predict
from core import create_dataset

# 导入数据集
(train_x, train_y), (test_x, test_y) = create_dataset()
# 导入模型结构
model = create_model()
# 导入模型数据
model.load_weights('./data/weights.h5')
# 用测试集验证模型
out = model.evaluate(test_x,test_y)

# 遍历data/mnist下的图片文件名, 并保存成列表
file_list = os.listdir('./data/mnist/')
# 将列表中文件名排序, 方便之后操作, 不做也可以hhh
file_list.sort(key=lambda x: int(x[:-4]))
# 打印文件名列表的内容, 程序出错的时候可以看看是不是没有找到文件
print('file_list: ', file_list)

# 遍历这个文件列表依次取出里面的文件名来放入模型预测
for file_name in file_list:
    # 因为file_name只是文件名, 读取文件的时候需要完整的文件路径, 这里拼接一下文件路径
    file_name = os.path.join('./data/mnist/', file_name)
    # 打印文件路径方便对照和调试, 不想加可以去掉
    print(file_name)
    # 读取图片文件
    picture = cv2.imread(file_name)
    # 把读出来的图片放进模型里进行预测, 因为这个操作十分常见且通用, 所以建议封装
    # 这里我们使用了一个封装, 输入图片和模型就能得到最后结果
    # 函数的定义在core.predict文件中
    out = predict(picture=picture, model=model)
    # 把原来28*28大小的图片放大到280*280, 觉得小可以自己调大一点
    large = cv2.resize(picture, (280, 280))
    # 打印结果
    print(out)
    # 创建窗口并显示出来
    cv2.imshow('Result: %d(Next:press Space/Exit:press Esc)' % out, large)
    # 这里用于获取键盘输入, 按下空格切换到下一张, 按下Esc键关闭程序
    if cv2.waitKey(0) == 27:
        break
    # 用于关闭前一个窗口, 防止创建过多的窗口
    cv2.destroyAllWindows()

# 关闭窗口, 多余的操作, 但建议加上，防止CV2运行出错时无法释放资源
cv2.destroyAllWindows()