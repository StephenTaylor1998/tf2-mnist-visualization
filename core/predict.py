import tensorflow as tf


# 封装了模型的预测操作
def predict(picture, model, draw_in_console=True, print_confidence_rate=False):
    # print(picture[:, :, 0].shape)
    # 将输入的图片进行预处理
    # 原图是三通道灰度图, 这里取出其中任意一个通道即可
    # 输入的图片需要是四维的, 分别为(batch * width * height * channel)
    # 这里缺少batch这一个维度所以用tf.expand_dims补上
    input_tensor = tf.expand_dims(picture[:, :, 0] * 1.0, 0) / 255.0
    # 这里是在命令行中打印出图片 亮度 > 0.1 的位置画 '*', 否则画空格
    # 当draw_in_console=False 时即可取消绘制
    if draw_in_console:
        for i in input_tensor[0]:
            for j in i:
                # print(int(j+0.5)*5, end=' ')
                if int(j + 0.9) == 0:
                    print(' ', end=' ')
                else:
                    print('*', end=' ')
            print()
    # 将处理过的图片送入模型, 得到结果
    out = model.predict(input_tensor)
    # 将输出的one hot转换成具体的数字
    index = out.argmax()
    # 如果需要可以打印模型对这张图片的置信度
    if print_confidence_rate:
        print('模型置信度(可信度): ', out[0, index])
    return index
