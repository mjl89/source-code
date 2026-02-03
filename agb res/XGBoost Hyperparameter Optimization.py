import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor

path = r'E:\DATA\data\traindata-43.csv'
dataset = pd.read_csv(path)
dataset.describe()

X = dataset.iloc[:, 1:18].values
Y = dataset.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# #构造要优化的函数
# XGBOOST
def black_box_function(n_estimators, learning_rate, max_depth,min_child_weight):
    res = XGBRegressor(n_estimators=int(n_estimators),
                        learning_rate=min(learning_rate, 0.999),
                        # min_samples_split=int(min_samples_split),
                        # max_features=min(max_features, 0.999),  # float
                        max_depth=int(max_depth),
                        min_child_weight=int(min_child_weight),
                        # random_state=0,
                        ).fit(x_train, y_train).score(x_test, y_test)

    return res
#XGB
pbounds = {
                'n_estimators': (10,300),
                'learning_rate':(0.01, 0.999),
                'max_depth': (1, 30),
                'min_child_weight': (2, 40)
                }

optimizer = BayesianOptimization(
    f=black_box_function,  # 目标函数
    pbounds=pbounds,  # 取值空间
    verbose=2,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
    random_state=1,
)

optimizer.maximize(
    init_points=5,  # 随机搜索的步数
    n_iter=10,  # 执行贝叶斯优化迭代次数
    # acq='ei'
)

print(optimizer.max)
res = optimizer.max
params_max = res['params']

print(res)  # 打印所有优化的结果
print(params_max)  # 最好的结果与对应的参数