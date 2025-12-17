# 电影票房预测项目 - CatBoost&ResNet18

具体代码数据详见box_office.zip

## 📁 项目结构

```
box_office/
├── catboost/                    # CatBoost模型相关文件
│   ├── __pycache__/            # Python缓存文件
│   ├── results_example/        # 结果示例
│   ├── advanced_analysis.py    # 高级分析和评估报告生成
│   ├── Catboost.py             # 基础CatBoost模型（无res_fc特征）
│   └── train_catboost_with_resfc.py  # 带res_fc特征的CatBoost模型
├── data/                       # 数据文件夹
│   ├── photos/                 # 图片数据
│   ├── pic/                    # 图片文件夹
│   ├── raw_data/               # 原始数据
│   ├── test_ceshi.xlsx         # 测试数据集
│   └── train_ceshi.xlsx        # 训练数据集
├── res_fc/                     # Res-FC神经网络模型
│   ├── __pycache__/
│   ├── result/
│   ├── trainpth/               # 训练好的模型权重
│   ├── __init__.py
│   ├── config.py               # 配置文件
│   ├── data.py                 # 数据加载和预处理
│   ├── main.py                 # 主程序
│   ├── model.py                # 模型定义
│   ├── test.py                 # 测试脚本
│   └── train.py                # 训练脚本
├── baseline/                   # 基线模型文件夹
└── README.md                   # 本文件
```

## 🎯 项目概述

本项目使用两种方法来预测电影票房：
1. **纯CatBoost模型**：使用传统的特征工程和CatBoost回归模型
2. **CatBoost + Res-FC模型**：将Res-FC神经网络的输出作为特征，与原始特征结合后使用CatBoost进行预测

## 🚀 快速开始

### 环境要求

```bash
# 安装依赖
pip install catboost pandas numpy scikit-learn matplotlib torch
# 可选：安装shap用于高级分析
pip install shap scipy
```

### 方法一：使用纯CatBoost模型

1. **准备数据**
   - 训练数据：`data/train_ceshi.xlsx`
   - 测试数据：`data/test_ceshi.xlsx`

2. **运行基础模型**
   ```bash
   python catboost/Catboost.py
   ```

3. **查看结果**
   - 预测结果：`catboost/predictions.csv`
   - 特征重要性：`catboost/feature_importance.csv`
   - 可视化图表：`catboost/catboost_result.png`

### 方法二：使用CatBoost + Res-FC模型

1. **首先训练Res-FC模型（如果尚未训练）**
   ```bash
   cd res_fc
   python main.py
   ```

2. **运行带res_fc特征的CatBoost模型**
   ```bash
   python catboost/train_catboost_with_resfc.py
   ```

3. **运行高级分析**
   ```bash
   python catboost/advanced_analysis.py
   ```

## 📊 文件说明

### 核心脚本

| 文件名 | 功能描述 | 输入 | 输出 |
|--------|----------|------|------|
| `Catboost.py` | 基础CatBoost模型训练和预测 | train_ceshi.xlsx, test_ceshi.xlsx | 基础预测结果和图表 |
| `train_catboost_with_resfc.py` | 带res_fc特征的CatBoost模型 | train_ceshi.xlsx, test_ceshi.xlsx, res_fc模型 | 模型文件、预测结果 |
| `advanced_analysis.py` | 高级分析和报告生成 | 上一脚本的输出文件 | SHAP分析、详细报告 |

### 数据文件

| 文件名 | 描述 | 格式 |
|--------|------|------|
| `train_ceshi.xlsx` | 训练数据集 | Excel文件 |
| `test_ceshi.xlsx` | 测试数据集 | Excel文件 |
| `photos/` | 图片特征数据 | 图片文件 |
| `raw_data/` | 原始数据文件 | 多种格式 |

### 模型文件

| 目录 | 描述 |
|------|------|
| `res_fc/` | Res-FC神经网络模型实现 |
| `catboost/` | CatBoost模型和相关分析脚本 |

## 🔧 配置说明

### 数据格式要求

1. **Excel数据格式**：
   - 第0列：ID（可选）
   - 第x列开始：特征列
   - 前y列为分类特征（可配置）
   - 最后一列：票房标签

2. **图片数据**：
   - 存储在`data/photos/`目录下
   - 用于Res-FC模型的特征提取

### 分类特征配置

在代码中修改以下配置：
```python
cat_features_indices = []  # 根据实际数据调整
```

## 📈 模型性能评估

### 基础评估指标
- 均方误差 (MSE)
- 均方根误差 (RMSE)
- 平均绝对误差 (MAE)
- 平均绝对百分比误差 (MAPE)
- R²分数

### 高级评估（使用advanced_analysis.py）
- SHAP值分析
- 残差正态性检验
- 特征重要性分析
- 统计检验
- 分位数性能评估
- 模型诊断报告

## 📊 输出文件说明

### 纯CatBoost模型输出 (`catboost_info/`)
```
catboost_info/
├── X_train_processed.csv        # 处理后的训练特征
├── X_test_processed.csv         # 处理后的测试特征
├── y_train.csv                  # 训练标签
├── y_test.csv                   # 测试标签
├── catboost_model.cbm           # 训练好的模型文件
├── predictions.csv              # 预测结果（真实值、预测值、残差）
├── feature_importance.csv       # 特征重要性排名
└── catboost_result.png          # 6个子图的综合可视化
```

### CatBoost+Res-FC模型输出 (`catboost_info_with_resfc/`)
```
catboost_info_with_resfc/
├── 基本文件（同上）
├── basic_metrics.json           # 基础指标JSON文件
├── detailed_evaluation_report.json  # 详细评估报告
├── evaluation_report.html       # HTML格式报告
├── model_performance_summary.txt    # 模型总结报告
├── catboost_with_res_fc_results.png # 基础可视化
├── shap_summary_plot.png        # SHAP特征重要性图
├── shap_feature_importance.png  # SHAP条形图
├── shap_dependence_plots.png    # SHAP依赖图
├── qq_plot_residuals.png        # 残差Q-Q图
├── error_distribution_analysis.png # 误差分布分析
├── error_trend_analysis.png     # 误差趋势分析
└── residuals_autocorrelation.png # 残差自相关图
```

## 📊 数据集来源说明

### 数据整合概述

本数据集是一个综合性的中国电影票房预测数据集，通过融合多个权威平台的电影相关信息构建而成。数据集旨在为电影票房预测提供多维度的特征支持。

### 主要数据来源

#### 1. 基础电影信息（Kaggle原始数据集）
**来源**：Kaggle平台的"CMM Chinese Multi-Modal Movie"数据集  
**包含内容**：
- 电影基本信息（名称、ID、类型、上映地区等等）
- 电影海报图片
- 用户评论数据
- 部分票房信息

#### 2. 票房数据补充
**来源**：中国电影票房数据网 & 猫眼专业版  
**补充内容**：
- 精确的电影总票房数据（单位：万元）
- 上映日期及相关时间信息
- 电影持续上映天数

#### 3. 演职人员影响力数据
**来源**：猫眼平台"主演票房榜单TOP100"  
**采集方式**：
- 通过自动化脚本采集当前市场头部演员的累计票房数据
- 将演员票房影响力映射到具体电影
- 采用对数变换处理构建"演员票房影响力"特征

#### 4. 用户评论深度分析
**来源**：原始数据集中的用户评论  
**处理方式**：
- **平均评论长度**：计算每部电影评论的平均字符数
- **加权情感得分**：基于评论点赞数进行对数平滑加权的情感评分
- **口碑趋势斜率**：分析电影上映首周评分变化趋势


### 特征构建过程

#### 结构化特征提取
1. **电影类型特征**：从复合字段中解析出11种电影类型，采用独热编码表示
2. **时间特征**：提取上映年份、月份、星期几等时间维度信息
3. **上映地点**：从复合信息中提取电影主要上映地区

#### 数值特征计算
1. **演职人员影响力**：基于TOP100演员票房数据，计算电影主演阵容的市场影响力
2. **评论质量指标**：从用户评论中提取量化指标，反映电影口碑
3. **票房走势特征**：提供详细的票房时间序列数据

### 数据集特点

#### 多源数据融合
数据集整合了来自官方统计平台、商业平台和用户生成内容的多维度信息，形成了全面覆盖电影商业表现、内容特征和市场反响的特征体系。

#### 特征丰富性
共包含45个特征字段，涵盖：
- 电影基本信息（4个特征）
- 内容类型特征（11个二元特征）
- 时间特征（4个特征）
- 评论分析特征（3个特征）
- 演员影响力特征（1个特征）
- 票房时间序列（20个特征）
- 目标变量（1个特征）


## 🔍 使用建议

1. **初次使用**：先运行`Catboost.py`了解基础模型性能
2. **提升性能**：使用`train_catboost_with_resfc.py`集成深度特征
3. **深入分析**：使用`advanced_analysis.py`生成详细报告
4. **模型对比**：比较两种方法的预测结果，选择最佳模型


## 📄 许可证

本项目仅供学习和研究使用。
