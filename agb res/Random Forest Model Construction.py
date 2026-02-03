import csv
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=42, shuffle=True)  # 10折

path = r"D:\Ruanjian\Pycharm\pythonProject\liyao\wfl\wfldatatraindata.csv"
dataset = pd.read_csv(path)
dataset.describe()
X = dataset.iloc[:, 1:32].values
y = dataset.iloc[:, 0].values
# 结果输出
output_path = r'D:\Ruanjian\Pycharm\pythonProject\liyao\wfl\RFKF10.csv'
f = open(output_path, 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(['Field_Biomass', 'Field_Biomass_pred'])

# 划分训练数据集，训练模型
# 显示具体划分情况
for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Validation:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

i = 1
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = RandomForestRegressor(n_estimators=460, max_depth=45, random_state=0)
    # 训练模型
    model.fit(X_train, y_train)
    # joblib.dump(model, r'E:\vippython\model\XGBoostTOA_PM25.m')  # 保存模型
    y_predict = model.predict(X_test)
    for j in range(len(y_test)):
        array = []
        array.append(y_test[j])
        array.append(y_predict[j])
        csv_writer.writerow(array)
    print('成功')
    print('MAE:', metrics.mean_absolute_error(y_test, y_predict))
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    print('RMSE: {:.3f}'.format(rmse))
    # print('MSE: ' + str(mean_squared_error(y_test, y_predict)))
    print('R^2: ' + str(r2_score(y_test, y_predict)))
    # shijian_test(model, i)
    i += 1
f.close()

# 将训练好的模型进行保存
model = RandomForestRegressor(n_estimators=521, max_depth=39, random_state=0)
model.fit(X, y)
joblib.dump(model, r'D:\Ruanjian\Pycharm\pythonProject\liyao\wfl\RFmodel.m')  # 保存模型
y_predict = model.predict(X)
# for j in range(len(y)):
#     array = []
#     array.append(y[j])
#     array.append(y_predict[j])
#     csv_writer.writerow(array)
print('成功')
print('MAE:', metrics.mean_absolute_error(y, y_predict))
rmse = np.sqrt(mean_squared_error(y, y_predict))
print('RMSE: {:.8f}'.format(rmse))
print('R^2: ' + str(r2_score(y, y_predict)))
