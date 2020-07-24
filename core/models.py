import tensorflow as tf
from tensorflow.keras import layers, optimizers


# 这个API用于创建模型, 并输出模型的结构
def create_model():
    # 用keras.Sequential构建一个模型，并从keras.optimizers实例化一个随机梯度下降优化器。
    model = tf.keras.Sequential([
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(10, activation='softmax')
    ])
    # 打印模型结构
    model.summary()
    # 优化器
    optimizer = optimizers.Adam(lr=1e-3)
    # loss函数
    loss_function = tf.losses.categorical_crossentropy
    # 编译模型, 使用Adam优化器, 二次交叉熵loss, 评判标准为正确率acc(Accuracy)
    model.compile(optimizer=optimizer, metrics=['acc'], loss=loss_function)
    return model
