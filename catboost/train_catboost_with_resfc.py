'''
将训练的res_fc模型得到的输出作为一个特征 - 主训练文件
'''
#========================导入库=========================
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
other_folder_path = os.path.join(project_root, "res_fc")
sys.path.insert(0, other_folder_path)

import config
import model
import data

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

#=======================config==========================
dataFile = "box_office/data/train_ceshi.xlsx"
test_dataFile = "box_office/data/test_ceshi.xlsx"
model_path = "box_office/res_fc/trainpth/model_res_img.pth"
output_dir = "box_office/catboost/catboost_info_with_resfc"
os.makedirs(output_dir, exist_ok=True)

#=======================加载数据=======================

train_df = pd.read_excel(dataFile)
test_df = pd.read_excel(test_dataFile)


X_train_original = train_df.iloc[:, 2:-1]  
y_train = train_df.iloc[:, -1]

X_test_original = test_df.iloc[:, 2:-1] 
y_test = test_df.iloc[:, -1]

cat_features_indices = list(range(0, 17))  

#=======================定义res_fc特征提取器=======================
class ResFcFeatureExtractor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.imgResTrain().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def extract_features(self, dataFile):
        """从数据文件中提取res_fc特征"""
        Dataset = data.dataset(boxDataFile=dataFile)
        dataloader = DataLoader(Dataset,
                              batch_size=config.batchSize,
                              shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch_idx, (input_batch, label_batch) in enumerate(dataloader):
                input_batch = input_batch.to(self.device)
                predict_batch = self.model(input_batch)
                predictions.extend(predict_batch.cpu().numpy())
                
                print(f"Processed batch {batch_idx+1}/{len(dataloader)}")
        
        result = pd.DataFrame(predictions, columns=["res_fc_feature"])
        return result

#=======================提取res_fc特征=======================
print("正在提取res_fc特征...")
feature_extractor = ResFcFeatureExtractor(model_path)

train_res_fc_features = feature_extractor.extract_features(dataFile)
print(f"训练数据res_fc特征形状: {train_res_fc_features.shape}")

test_res_fc_features = feature_extractor.extract_features(test_dataFile)
print(f"测试数据res_fc特征形状: {test_res_fc_features.shape}")

#=======================合并特征=======================
X_train = pd.concat([X_train_original.reset_index(drop=True), 
                    train_res_fc_features.reset_index(drop=True)], axis=1)

X_test = pd.concat([X_test_original.reset_index(drop=True), 
                   test_res_fc_features.reset_index(drop=True)], axis=1)


print(f"合并后训练特征形状: {X_train.shape}")
print(f"合并后测试特征形状: {X_test.shape}")

# 保存处理后的特征
X_train.to_csv(f"{output_dir}/X_train_processed.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test_processed.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

#=======================训练CatBoost模型=======================
catboost_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.02,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=100,
    random_seed=42,
    cat_features=cat_features_indices,  
    l2_leaf_reg=3,
    early_stopping_rounds=50,
    task_type='GPU' if torch.cuda.is_available() else 'CPU' 
)

print("\n开始训练CatBoost模型...")
catboost_model.fit(X_train, y_train)

# 保存模型
catboost_model.save_model(f"{output_dir}/catboost_model.cbm")
print(f"模型已保存至: {output_dir}/catboost_model.cbm")

#=======================预测和评估=======================
y_pred = catboost_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with np.errstate(divide='ignore', invalid='ignore'):
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    mape = np.nan_to_num(mape, nan=0.0)

print("\n" + "="*50)
print("模型性能评估结果:")
print("="*50)
print(f"预测值前10个: {y_pred[:10]}")
print(f"真实值前10个: {y_test.values[:10]}")

print(f"\n回归性能指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
print(f"R²分数: {r2:.4f}")

# 保存基础评估结果
basic_metrics = {
    "MSE": float(mse),
    "RMSE": float(rmse),
    "MAE": float(mae),
    "MAPE": float(mape),
    "R2": float(r2)
}

import json
with open(f"{output_dir}/basic_metrics.json", 'w') as f:
    json.dump(basic_metrics, f, indent=4)

#=======================可视化=======================
plt.figure(figsize=(15, 10))

# 1. 预测值 vs 真实值散点图
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Real Box Office')
plt.ylabel('Predicted Box Office')
plt.title('Real vs Predicted Box Office')
plt.grid(True, alpha=0.3)

# 2. 残差图
plt.subplot(2, 3, 2)
residuals = y_test.values - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Box Office')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True, alpha=0.3)

# 3. 残差分布
plt.subplot(2, 3, 3)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.grid(True, alpha=0.3)

# 4. 真实值和预测值对比（前50个样本）
plt.subplot(2, 3, 4)
sample_indices = np.arange(min(50, len(y_test)))
plt.plot(sample_indices, y_test.values[:50], 'b-', label='Real', alpha=0.7, marker='o')
plt.plot(sample_indices, y_pred[:50], 'r--', label='Predicted', alpha=0.7, marker='s')
plt.xlabel('Sample Index')
plt.ylabel('Box Office')
plt.title('Real vs Predicted (First 50 Samples)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. 绝对误差箱线图
plt.subplot(2, 3, 5)
absolute_errors = np.abs(residuals)
plt.boxplot(absolute_errors)
plt.ylabel('Absolute Error')
plt.title('Absolute Error Distribution')
plt.grid(True, alpha=0.3)

# 6. 特征重要性
plt.subplot(2, 3, 6)
feature_importance = catboost_model.get_feature_importance()

feature_names = list(X_train_original.columns) + ['res_fc_feature']
sorted_idx = np.argsort(feature_importance)[-10:]  # 取最重要的10个特征
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.tight_layout()

plt.savefig(f"{output_dir}/catboost_with_res_fc_results.png")
print(f"\n可视化结果已保存至: {output_dir}/catboost_with_res_fc_results.png")

# 保存预测结果
results_df = pd.DataFrame({
    'Real': y_test.values,
    'Predicted': y_pred,
    'Residual': residuals
})
results_df.to_csv(f"{output_dir}/predictions.csv", index=False)
print(f"详细预测结果已保存至: {output_dir}/predictions.csv")

# 保存特征重要性
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)
feature_importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)

plt.show()

print("\n" + "="*50)
print("训练完成！")
print(f"所有文件已保存至: {output_dir}")
print("="*50)