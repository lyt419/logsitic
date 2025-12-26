import numpy as np
from sklearn.linear_model import LogisticRegression
filename=r"c:\Users\Administrator\Desktop\机器学习\lesson4\testSet.txt"
#=====================
# 1. 数据读取函数
#=====================
def load_dataset(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]   # 特征
    y = data[:, -1]    # 标签
    return X, y
#=====================
# 2. 缺失值处理函数
#   （缺失值替换为该列均值）
#=====================
def replace_nan_with_mean(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        # 选择非0的数作为有效特征
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X[:, i] = col
    return X
#=====================
# 3. 主流程
#=====================
# 读取训练集


X_train, y_train = load_dataset(train_filename)
# 读取测试集
X_test, y_test = load_dataset(test_filename)

# 处理训练集和测试集的缺失值
X_train_processed = replace_nan_with_mean(X_train)
X_test_processed = replace_nan_with_mean(X_test)

#=====================
# 4. 构建并训练逻辑回归模型
#=====================
# 创建逻辑回归模型（可根据需要调整参数）
lr_model = LogisticRegression(
    random_state=42,    # 随机种子保证结果可复现
    max_iter=1000,      # 最大迭代次数
    solver='lbfgs'     # 优化器
)

# 训练模型
lr_model.fit(X_train_processed, y_train)

#=====================
# 5. 测试集预测
#=====================

# 预测测试集标签（概率/类别）
y_test_pred_prob = lr_model.predict_proba(X_test_processed)  # 预测概率
y_test_pred = lr_model.predict(X_test_processed)             # 预测类别

#=====================
# 6. 计算准确率
#=====================
accuracy = accuracy_score(y_test, y_test_pred)
