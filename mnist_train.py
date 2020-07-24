# 导入项目下其他自己封装的功能
from core import create_model
from core import create_dataset

# 生成模型
model = create_model()
# 导入数据集
(train_x, train_y), (test_x, test_y) = create_dataset()
# 训练，使模型拟合(fit)数据集
model.fit(train_x, train_y, epochs=5)
# 验证，查看模型在测试集上的效果
out = model.evaluate(test_x,test_y)
# 打印结果
print('loss in test set:', out[0], '  accuracy:', out[1])
# 保存权重
model.save_weights('./data/weights.h5')
# 加载保存的权重
model.load_weights('./data/weights.h5')
# 再次验证, 用于验证保存的模型是否出错
out = model.evaluate(test_x,test_y)