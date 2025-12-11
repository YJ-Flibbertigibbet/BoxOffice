#========================导入库=========================
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
other_folder_path = os.path.join(project_root, "res_fc")
sys.path.insert(0, other_folder_path)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
#=======================config==========================
dataFile="box_office/data/raw_ceshi.xlsx"


#=======================catboost========================
df = pd.read_excel(dataFile)
data=df.iloc[:,1:]
# 写清楚分类变量在第几列，模型会自适应
# 例如前四列是分类变量
cat_features_indices = [0,1,2,3]

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostRegressor(
    iterations=800,           # 增加迭代次数
    learning_rate=0.04,       # 降低学习率
    depth=6,                  # 树深度
    loss_function='RMSE',     # 回归损失函数
    eval_metric='RMSE',       # 评估指标
    verbose=100,              # 显示训练过程
    random_seed=42,
    cat_features=cat_features_indices,
    l2_leaf_reg=3,            # 正则化
    early_stopping_rounds=50  # 早停
)


model.fit(X_train, y_train, cat_features=cat_features_indices)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  



print(f"预测值前10个: {y_pred[:10]}")
print(f"真实值前10个: {y_test.values[:10]}")
print(f"\n回归性能指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
print(f"R²分数: {r2:.4f}")


#==================可视化======================

plt.figure(figsize=(15, 10))

# 1. 预测值 vs 真实值散点图
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('real_box')
plt.ylabel('pre_box')
plt.title('real_box vs pre_box')
plt.grid(True, alpha=0.3)

# 2. 残差图
plt.subplot(2, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('pre_box')
plt.ylabel('SSE')
plt.title('SSE')
plt.grid(True, alpha=0.3)

# 3. 残差分布
plt.subplot(2, 3, 3)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('SSE')
plt.ylabel('freq')
plt.title('SSE')
plt.grid(True, alpha=0.3)

# 4. 真实值和预测值对比（前50个样本）
plt.subplot(2, 3, 4)
sample_indices = np.arange(min(50, len(y_test)))
plt.plot(sample_indices, y_test.values[:50], 'b-', label='real_box', alpha=0.7, marker='o')
plt.plot(sample_indices, y_pred[:50], 'r--', label='pre_box', alpha=0.7, marker='s')
plt.xlabel('idx')
plt.ylabel('box')
plt.title('real_box vs pre_box（first50）')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. 误差分布
plt.subplot(2, 3, 5)
absolute_errors = np.abs(residuals)
plt.boxplot(absolute_errors)
plt.ylabel('AE')
plt.title('AE')
plt.grid(True, alpha=0.3)

# 6. 特征重要性
plt.subplot(2, 3, 6)
feature_importance = model.get_feature_importance()
sorted_idx = np.argsort(feature_importance)[-10:]  # 取最重要的10个特征
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
plt.xlabel('importance')
plt.title('Top 10 importance')
plt.tight_layout()
plt.savefig("box_office/catboost/catboost_result.png")