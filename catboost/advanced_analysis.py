'''
高级分析和评估报告生成
'''

#========================导入库=========================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import shap
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

#=======================配置===========================
output_dir = "box_office/catboost/catboost_info_with_resfc"

# 检查必要文件是否存在
required_files = [
    f"{output_dir}/X_test_processed.csv",
    f"{output_dir}/y_test.csv",
    f"{output_dir}/predictions.csv",
    f"{output_dir}/catboost_model.cbm"
]

for file in required_files:
    if not os.path.exists(file):
        print(f"错误: 找不到文件 {file}")
        print("请先运行 train_catboost_with_resfc.py")
        sys.exit(1)

print("正在加载数据和模型...")

X_test = pd.read_csv(f"{output_dir}/X_test_processed.csv")
y_test = pd.read_csv(f"{output_dir}/y_test.csv").iloc[:, 0]  
predictions_df = pd.read_csv(f"{output_dir}/predictions.csv")

catboost_model = CatBoostRegressor()
catboost_model.load_model(f"{output_dir}/catboost_model.cbm")

y_pred = predictions_df['Predicted'].values
y_test_values = predictions_df['Real'].values
residuals = predictions_df['Residual'].values

mse = mean_squared_error(y_test_values, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_values, y_pred)
r2 = r2_score(y_test_values, y_pred)

with np.errstate(divide='ignore', invalid='ignore'):
    mape = np.mean(np.abs((y_test_values - y_pred) / y_test_values)) * 100
    mape = np.nan_to_num(mape, nan=0.0)

#=======================SHAP值分析=======================
print("\n" + "="*50)
print("SHAP值分析")
print("="*50)

try:

    explainer = shap.TreeExplainer(catboost_model)
    
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    
    print(f"使用 {sample_size} 个样本计算SHAP值...")
    
    # 1. SHAP汇总图
    plt.figure(figsize=(15, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP汇总图已保存")
    
    # 2. SHAP条形图（特征重要性）
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP特征重要性图已保存")
    
    # 3. SHAP依赖图（分析重要特征的影响）
    print("生成SHAP依赖图...")
    
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(shap_importance)[-5:]  
    top_features = [X_test.columns[i] for i in top_features_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features[:6]):
        if feature in X_test_sample.columns:
            shap.dependence_plot(feature, shap_values, X_test_sample, 
                                ax=axes[idx], show=False)
            axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=12)
    

    for idx in range(len(top_features[:6]), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_dependence_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP依赖图已保存")
    
except Exception as e:
    print(f"SHAP分析出错: {e}")
    print("跳过SHAP分析...")

#=======================扩展评估指标=======================
print("\n" + "="*50)
print("扩展评估指标计算")
print("="*50)


try:
    # 1. 相关系数
    pearson_corr, pearson_p = stats.pearsonr(y_test_values, y_pred)
    spearman_corr, spearman_p = stats.spearmanr(y_test_values, y_pred)
    
    # 2. 计算R² adjusted
    n = len(y_test_values)
    p = X_test.shape[1]
    r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # 3. 计算AIC和BIC（信息准则）
    aic = n * np.log(mse) + 2 * p
    bic = n * np.log(mse) + p * np.log(n)
    
    # 4. 计算Explained Variance Score
    explained_variance = 1 - np.var(residuals) / np.var(y_test_values)
    
    # 5. 计算Max Error
    max_error = np.max(np.abs(residuals))
    
    # 6. 分位数损失函数
    def quantile_loss(y_true, y_pred, quantile=0.5):
        errors = y_true - y_pred
        return np.maximum(quantile * errors, (quantile - 1) * errors).mean()
    
    median_absolute_error = np.median(np.abs(residuals))
    quantile_50_loss = quantile_loss(y_test_values, y_pred, 0.5)
    quantile_90_loss = quantile_loss(y_test_values, y_pred, 0.9)
    quantile_10_loss = quantile_loss(y_test_values, y_pred, 0.1)
    
    # 7. 对称平均绝对百分比误差 (sMAPE)
    smape = 200 * np.mean(np.abs(residuals) / (np.abs(y_test_values) + np.abs(y_pred)))
    
    # 8. 计算回归置信区间
    confidence = 0.95
    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
    std_error = np.std(residuals) / np.sqrt(n)
    ci_lower = np.mean(residuals) - t_value * std_error
    ci_upper = np.mean(residuals) + t_value * std_error
    
    # 9. 误差分布统计
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    jarque_bera_stat, jarque_bera_p = stats.jarque_bera(residuals)
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000]) if len(residuals) > 5000 else stats.shapiro(residuals)
    
    print("✓ 扩展评估指标计算完成")
    
except Exception as e:
    print(f"扩展指标计算出错: {e}")

#=======================扩展可视化=======================
print("\n" + "="*50)
print("生成扩展可视化图表")
print("="*50)

# 1. Q-Q图（检验残差正态性）
plt.figure(figsize=(10, 8))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals (Normality Test)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/qq_plot_residuals.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Q-Q图已保存")

# 2. 预测误差分布与正态分布对比
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=50, density=True, alpha=0.6, color='b', label='Residuals')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.title('Residual Distribution vs Normal', fontsize=14)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. 累计误差分布
plt.subplot(1, 2, 2)
sorted_residuals = np.sort(np.abs(residuals))
cdf = np.arange(1, len(sorted_residuals)+1) / len(sorted_residuals)
plt.plot(sorted_residuals, cdf, 'b-', linewidth=2)
plt.xlabel('Absolute Error')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution of Absolute Errors', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/error_distribution_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ 误差分布分析图已保存")

# 4. 预测误差趋势分析
plt.figure(figsize=(14, 10))

# 按样本顺序的误差变化
plt.subplot(2, 2, 1)
plt.plot(range(len(residuals)), residuals, 'b-', alpha=0.6, linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
plt.xlabel('Sample Index')
plt.ylabel('Residual')
plt.title('Residuals by Sample Order', fontsize=14)
plt.grid(True, alpha=0.3)

# 误差与预测值的关系
plt.subplot(2, 2, 2)
plt.scatter(y_pred, np.abs(residuals), alpha=0.5, s=10)
plt.xlabel('Predicted Value')
plt.ylabel('Absolute Error')
plt.title('Absolute Error vs Predicted Value', fontsize=14)
plt.grid(True, alpha=0.3)

# 误差与实际值的关系
plt.subplot(2, 2, 3)
plt.scatter(y_test_values, np.abs(residuals), alpha=0.5, s=10)
plt.xlabel('Actual Value')
plt.ylabel('Absolute Error')
plt.title('Absolute Error vs Actual Value', fontsize=14)
plt.grid(True, alpha=0.3)

# 误差分布箱线图（按预测值分位数）
plt.subplot(2, 2, 4)
num_bins = 5
bin_edges = np.percentile(y_pred, np.linspace(0, 100, num_bins + 1))
bin_indices = np.digitize(y_pred, bin_edges) - 1
bin_indices[bin_indices == num_bins] = num_bins - 1

boxplot_data = []
box_labels = []
for i in range(num_bins):
    mask = bin_indices == i
    if np.sum(mask) > 0:
        boxplot_data.append(np.abs(residuals[mask]))
        box_labels.append(f'Q{i+1}')

plt.boxplot(boxplot_data, labels=box_labels)
plt.xlabel('Predicted Value Quantiles')
plt.ylabel('Absolute Error')
plt.title('Error Distribution by Prediction Quantiles', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/error_trend_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ 误差趋势分析图已保存")

# 5. 残差自相关图
plt.figure(figsize=(10, 6))
pd.plotting.autocorrelation_plot(residuals)
plt.title('Residuals Autocorrelation', fontsize=16)
plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/residuals_autocorrelation.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ 残差自相关图已保存")

# 6. 学习曲线（如果训练历史可用）
try:
    # 尝试获取训练历史
    cv_results = catboost_model.get_best_score()
    if cv_results:
        plt.figure(figsize=(10, 6))
        # 这里可以根据实际数据结构调整
        iterations = range(1, len(catboost_model.get_evals_result()['learn']['RMSE']) + 1)
        plt.plot(iterations, catboost_model.get_evals_result()['learn']['RMSE'], 
                label='Training RMSE', linewidth=2)
        plt.plot(iterations, catboost_model.get_evals_result()['validation']['RMSE'], 
                label='Validation RMSE', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/learning_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 学习曲线图已保存")
except:
    print("⚠ 无法生成学习曲线")

#=======================生成详细评估报告=======================
print("\n" + "="*50)
print("生成详细评估报告")
print("="*50)

# 创建评估报告字典
evaluation_report = {
    "模型信息": {
        "模型类型": "CatBoostRegressor",
        "使用res_fc特征": True,
        "训练样本数": "从train_catboost_with_resfc.py获取",
        "测试样本数": len(X_test),
        "特征数量": X_test.shape[1],
        "分类特征数量": "从train_catboost_with_resfc.py获取"
    },
    "基础指标": {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
        "调整R2": float(r2_adjusted)
    },
    "相关性与分布": {
        "皮尔逊相关系数": float(pearson_corr),
        "斯皮尔曼相关系数": float(spearman_corr),
        "解释方差分数": float(explained_variance),
        "残差偏度": float(skewness),
        "残差峰度": float(kurtosis)
    },
    "误差分析": {
        "最大绝对误差": float(max_error),
        "中位数绝对误差": float(median_absolute_error),
        "对称MAPE": float(smape),
        "残差95%置信区间": [float(ci_lower), float(ci_upper)],
        "残差均值": float(np.mean(residuals)),
        "残差标准差": float(np.std(residuals))
    },
    "统计检验": {
        "Jarque_Bera_p值": float(jarque_bera_p),
        "Shapiro_Wilk_p值": float(shapiro_p),
        "正态性检验": "通过" if jarque_bera_p > 0.05 else "未通过"
    },
    "分位数性能": {
        "0.5分位数损失": float(quantile_50_loss),
        "0.9分位数损失": float(quantile_90_loss),
        "0.1分位数损失": float(quantile_10_loss)
    },
    "信息准则": {
        "AIC": float(aic),
        "BIC": float(bic)
    }
}

# 读取特征重要性
try:
    feature_importance_df = pd.read_csv(f"{output_dir}/feature_importance.csv")
    top_5_features = feature_importance_df.head(5).to_dict('records')
    evaluation_report["特征重要性"] = {
        "Top_5_特征": [
            {"特征": row['Feature'], "重要性": float(row['Importance'])} 
            for row in top_5_features
        ]
    }
except:
    print("⚠ 无法读取特征重要性文件")

# 保存评估报告为JSON
report_path = f"{output_dir}/detailed_evaluation_report.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(evaluation_report, f, indent=4, ensure_ascii=False)
print(f"✓ 详细评估报告已保存至: {report_path}")

# 生成HTML报告
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>模型评估报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .metric-name {{ font-weight: bold; }}
        .metric-value {{ color: #4CAF50; }}
        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .image {{ width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; }}
        h2 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CatBoost模型评估报告</h1>
            <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>基础指标</h2>
            <div class="grid">
                <div class="metric"><span class="metric-name">R²分数:</span><span class="metric-value">{r2:.4f}</span></div>
                <div class="metric"><span class="metric-name">RMSE:</span><span class="metric-value">{rmse:.4f}</span></div>
                <div class="metric"><span class="metric-name">MAE:</span><span class="metric-value">{mae:.4f}</span></div>
                <div class="metric"><span class="metric-name">MAPE:</span><span class="metric-value">{mape:.2f}%</span></div>
            </div>
        </div>
        
        <div class="section">
            <h2>可视化图表</h2>
            <div class="grid">
                <div>
                    <h3>预测 vs 真实值</h3>
                    <img class="image" src="catboost_with_res_fc_results.png" alt="预测结果">
                </div>
                <div>
                    <h3>SHAP特征重要性</h3>
                    <img class="image" src="shap_summary_plot.png" alt="SHAP分析">
                </div>
                <div>
                    <h3>误差分布分析</h3>
                    <img class="image" src="error_distribution_analysis.png" alt="误差分布">
                </div>
                <div>
                    <h3>残差正态性检验</h3>
                    <img class="image" src="qq_plot_residuals.png" alt="Q-Q图">
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>性能总结</h2>
            <p>模型在测试集上表现 {'良好' if r2 > 0.7 else '一般' if r2 > 0.5 else '较差'}</p>
            <p>预测精度: {'高' if mape < 10 else '中等' if mape < 20 else '较低'}</p>
            <p>残差正态性: {'符合正态分布' if jarque_bera_p > 0.05 else '不符合正态分布'}</p>
        </div>
    </div>
</body>
</html>
"""

html_path = f"{output_dir}/evaluation_report.html"
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_report)
print(f"✓ HTML评估报告已保存至: {html_path}")

#=======================生成总结报告=======================
print("\n" + "="*50)
print("模型评估总结")
print("="*50)

# 模型性能总结
performance_summary = f"""
模型性能总结报告
{'='*60}

一、模型基本信息
   - 模型类型: CatBoost回归 (集成res_fc特征)
   - 测试样本数: {len(X_test):,}
   - 特征总数: {X_test.shape[1]}

二、主要性能指标
   - R²分数: {r2:.4f} {'✓' if r2 > 0.7 else '⚠' if r2 > 0.5 else '✗'}
   - RMSE: {rmse:.4f}
   - MAE: {mae:.4f}
   - MAPE: {mape:.2f}%

三、统计检验结果
   - 残差正态性检验: {'通过' if jarque_bera_p > 0.05 else '未通过'}
   - 皮尔逊相关性: {pearson_corr:.4f}
   - 解释方差: {explained_variance:.4f}

四、模型建议
"""

if r2 > 0.8:
    performance_summary += "   ✓ 模型解释能力很强，可以用于生产环境\n"
elif r2 > 0.6:
    performance_summary += "   ⚠ 模型解释能力中等，可以考虑优化特征或参数\n"
else:
    performance_summary += "   ✗ 模型解释能力较弱，需要重新设计特征或模型\n"

if mape < 10:
    performance_summary += "   ✓ 预测精度很高\n"
elif mape < 20:
    performance_summary += "   ⚠ 预测精度可以接受\n"
else:
    performance_summary += "   ✗ 预测精度有待提高\n"

performance_summary += f"""
五、文件输出
   - 详细评估报告: {report_path}
   - HTML报告: {html_path}
   - 所有图表文件: {output_dir}/*.png
{'='*60}
"""

print(performance_summary)

# 保存总结报告
summary_path = f"{output_dir}/model_performance_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(performance_summary)

print(f"✓ 模型总结报告已保存至: {summary_path}")
print("\n" + "="*50)
print("高级分析完成！")
print(f"所有分析结果已保存至: {output_dir}")
print("="*50)

# 显示所有生成的图像文件
import glob
image_files = glob.glob(f"{output_dir}/*.png")
print(f"\n生成的图像文件 ({len(image_files)}个):")
for img_file in image_files:
    print(f"  - {os.path.basename(img_file)}")