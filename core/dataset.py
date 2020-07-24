from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical


# 这个API用于创建并处理数据集
def create_dataset():
    # 导入TF自带的原始数据集
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    # 简单归一化，并转换成float
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    # 转换成one hot编码形式
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return (train_x, train_y), (test_x, test_y)
