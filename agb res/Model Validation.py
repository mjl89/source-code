import csv
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers.experimental import preprocessing

# 加载模型
model_path = r"D:\Ruanjian\result\标准化\DNN\DNNKF10Model"
# model = load_model(model_path)
# model = joblib.load(model_path)
model = tf.keras.models.load_model(model_path)
# 读取数据
data_path = r'D:\Ruanjian\result\标准化\yanzhengmath归一化后的数据xin.csv'

# 读取数据
MyData = pd.read_csv(data_path, encoding='gbk', error_bad_lines=False)

# 提取自变量和因变量
X = MyData.copy(deep=True)
y = X.pop('XCH4')
print(X)
print(y)
# # 数据标准化
# normalizer = preprocessing.Normalization()
# normalizer.adapt(np.array(X))
#
# # 标准化输入数据
# X_normalized = normalizer(X).numpy()
# print(X_normalized)
# 使用模型进行预测
# y_predict = model.predict(X_normalized)
y_predict = model.predict(X)
print(y_predict)
# 将预测结果和真实值保存到CSV文件中
output_path = r'D:\Ruanjian\result\标准化\yanzhengmath归一化后的数据xinresult.csv'
with open(output_path, 'w', encoding='utf-8', newline="") as f:
    csv_writer = csv.writer(f)
    # 写入列名
    csv_writer.writerow(['y_predict', 'y'] + list(X.columns))
    # 写入每一行的数据
    for i in range(len(y)):
        csv_writer.writerow([y_predict[i][0], y.iloc[i]] + list(X.iloc[i]))

# 打印评估指标
print('成功')
print('MAE:', metrics.mean_absolute_error(y, y_predict))
rmse = np.sqrt(metrics.mean_squared_error(y, y_predict))
print('RMSE: {:.8f}'.format(rmse))
print('R^2:', metrics.r2_score(y, y_predict))

# 绘制散点图和趋势线
plt.scatter(y, y_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs. Predicted Values')
# 设置横纵坐标范围一样
min_val = min(1.70, 2.20)
max_val = max(1.70, 2.20)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
# 绘制趋势线
p = np.polyfit(y, y_predict.flatten(), 1)
plt.plot(y, np.polyval(p, y), 'r-')

plt.show()
