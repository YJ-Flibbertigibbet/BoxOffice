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
train_dataFile = "box_office/data/train_ceshi.xlsx"
test_dataFile = "box_office/data/test_ceshi.xlsx"
output_dir = "box_office/catboost/catboost_info"
os.makedirs(output_dir, exist_ok=True)

#=======================加载数据=======================
print("正在加载训练数据...")
train_df = pd.read_excel(train_dataFile)
print(f"训练数据形状: {train_df.shape}")

print("正在加载测试数据...")
test_df = pd.read_excel(test_dataFile)
print(f"测试数据形状: {test_df.shape}")


X_train = train_df.iloc[:, 1:-1]  
y_train = train_df.iloc[:, -1]   

X_test = test_df.iloc[:, 1:-1]   
y_test = test_df.iloc[:, -1]     

print(f"训练特征形状: {X_train.shape}, 训练标签形状: {y_train.shape}")
print(f"测试特征形状: {X_test.shape}, 测试标签形状: {y_test.shape}")

if list(X_train.columns) != list(X_test.columns):
    print("警告: 训练和测试数据特征列不一致!")

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    print(f"共同特征数量: {len(common_cols)}")
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

# 写清楚分类变量在第几列，模型会自适应
cat_features_indices = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16]  

print(f"分类特征索引: {cat_features_indices}")
print(f"分类特征名称: {X_train.columns[cat_features_indices].tolist()}")

# 保存处理后的数据
X_train.to_csv(f"{output_dir}/X_train_processed.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test_processed.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

#=======================catboost模型训练========================
model = CatBoostRegressor(
    iterations=800,           # 增加迭代次数
    learning_rate=0.025,       # 降低学习率
    depth=6,                  # 树深度
    loss_function='RMSE',     # 回归损失函数
    eval_metric='RMSE',       # 评估指标
    verbose=100,              # 显示训练过程
    random_seed=42,
    cat_features=cat_features_indices,
    l2_leaf_reg=3,            # 正则化
    early_stopping_rounds=50  # 早停
)

print("\n开始训练CatBoost模型...")
model.fit(X_train, y_train, cat_features=cat_features_indices)


model.save_model(f"{output_dir}/catboost_model.cbm")
print(f"模型已保存至: {output_dir}/catboost_model.cbm")

#=======================预测和评估========================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


with np.errstate(divide='ignore', invalid='ignore'):
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    mape = np.nan_to_num(mape, nan=0.0)

print(f"\n{'='*50}")
print("模型性能评估结果:")
print(f"{'='*50}")
print(f"预测值前10个: {y_pred[:10]}")
print(f"真实值前10个: {y_test.values[:10]}")
print(f"\n回归性能指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
print(f"R²分数: {r2:.4f}")


results_df = pd.DataFrame({
    'Real': y_test.values,
    'Predicted': y_pred,
    'Residual': y_test.values - y_pred
})
results_df.to_csv(f"{output_dir}/predictions.csv", index=False)
print(f"预测结果已保存至: {output_dir}/predictions.csv")


feature_importance = model.get_feature_importance()
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)
feature_importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
print(f"特征重要性已保存至: {output_dir}/feature_importance.csv")

#==================可视化======================
plt.figure(figsize=(15, 10))
residuals = y_test.values - y_pred

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
sorted_idx = np.argsort(feature_importance)[-10:]  # 取最重要的10个特征
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.xlabel('importance')
plt.title('Top 10 importance')
plt.tight_layout()

# 保存可视化结果
plt.savefig(f"{output_dir}/catboost_result.png", dpi=300, bbox_inches='tight')
print(f"可视化结果已保存至: {output_dir}/catboost_result.png")


print(f"\n{'='*50}")
print("训练和评估完成!")
print(f"所有结果已保存至: {output_dir}")
print(f"{'='*50}")