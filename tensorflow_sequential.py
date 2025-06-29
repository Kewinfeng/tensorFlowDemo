# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np

# 创建训练数据
X_train = np.array([1, 2, 3, 4, 5], dtype=float)
y_train = np.array([2, 3, 4, 5, 6], dtype=float)

# 创建模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# 编译模型
model.compile(optimizer='sgd', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=1000)

# 输出模型权重
weights = model.weights
print(f"模型权重: {weights[0].numpy()}, 偏置: {weights[1].numpy()}")

# 进行预测
new_data = np.array([6, 7], dtype=float)
predictions = model.predict(new_data)
print(f"预测结果: {predictions}")



