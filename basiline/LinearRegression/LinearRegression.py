import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import os

# ===================== 1. 数据加载与预处理 =====================
# 定义文件路径
train_path = "box_office/data/train_ceshi.xlsx"
test_path = "box_office/data/test_ceshi.xlsx"

# 加载数据
train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

# 数据处理：兼容无id字段，保留name作为识别，处理分类变量未知类别
def process_data(df, is_train=True, cat_feature_categories=None):
    df = df.copy()
    
    # 1. 剔除id字段（仅当存在时）
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    
    # 2. 保留name作为识别列（不参与建模）
    name_series = df["name"] if "name" in df.columns else pd.Series([f"影片_{i}" for i in range(len(df))])
    
    # 3. 分离特征和目标变量
    X = df.drop(["name", "票房_万"], axis=1) if "票房_万" in df.columns else df.drop("name", axis=1)
    y = df["票房_万"] if "票房_万" in df.columns else None
    
    # 4. 定义分类变量和数值变量
    cat_features = [
        "holiday", "show_place", "类型_剧情", "类型_动作", "类型_犯罪",
        "类型_悬疑", "类型_爱情", "类型_科幻", "类型_喜剧", "类型_冒险",
        "类型_战争", "类型_历史", "类型_家庭", "year", "month", "weekday"
    ]
    # 过滤数据中实际存在的分类变量（避免列不存在报错）
    cat_features = [col for col in cat_features if col in X.columns]
    num_features = [col for col in X.columns if col not in cat_features]
    
    # 5. 处理分类变量未知类别（训练集记录类别，测试集映射未知类别为"其他"）
    if is_train:
        # 训练集：记录每个分类变量的所有类别
        cat_feature_categories = {}
        for col in cat_features:
            # 填充缺失值为"未知"
            X[col] = X[col].fillna("未知")
            # 记录类别（包含"未知"）
            cat_feature_categories[col] = X[col].unique().tolist()
    else:
        # 测试集：将未知类别替换为"其他"
        for col in cat_features:
            X[col] = X[col].fillna("未知")
            # 只保留训练集见过的类别，其余替换为"其他"
            known_cats = cat_feature_categories[col]
            X[col] = X[col].apply(lambda x: x if x in known_cats else "其他")
    
    return name_series, X, y, cat_features, num_features, cat_feature_categories

# 处理训练集
train_names, X_train, y_train, cat_features_train, num_features_train, cat_categories = process_data(train_df, is_train=True)

# 处理测试集（使用训练集的类别信息）
test_names, X_test, y_test, _, _, _ = process_data(test_df, is_train=False, cat_feature_categories=cat_categories)

# ===================== 2. 构建预处理管道（兼容未知类别） =====================
# 数值变量处理：填充缺失值 + 标准化
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # 中位数填充数值缺失值
    ("scaler", StandardScaler())
])

# 分类变量处理：填充缺失值 + OneHot编码（drop="first"避免共线性）
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="未知")),  # 填充为"未知"
    ("onehot", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"))  # 忽略未知类别
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features_train),
        ("cat", cat_transformer, cat_features_train)
    ],
    remainder="passthrough"  # 防止遗漏字段
)

# ===================== 3. 模型构建与训练 =====================
# 构建线性回归管道
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 训练模型
model.fit(X_train, y_train)

# ===================== 4. 预测与结果关联 =====================
# 预测
y_pred = model.predict(X_test)
# 关联影片名称和预测/真实票房（方便识别）
pred_result = pd.DataFrame({
    "影片名称": test_names.values,
    "真实票房_万": y_test.values if y_test is not None else np.nan,
    "预测票房_万": y_pred.round(2)
})

# ===================== 5. 评估与报告生成 =====================
# 计算评估指标（仅当有真实值时）
if y_test is not None:
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
else:
    mae = mse = rmse = mape = r2 = np.nan

# 生成评估报告（包含影片识别结果）
report_content = f"""
===================== 线性回归模型评估报告 =====================
一、核心评估指标（测试集）：
1. 平均绝对误差（MAE）：{mae:.2f} 万
2. 均方误差（MSE）：{mse:.2f} 万²
3. 均方根误差（RMSE）：{rmse:.2f} 万
4. 平均百分比误差（MAPE）：{mape:.2%}
5. 决定系数（R²）：{r2:.4f}

二、部分影片预测结果（前10条，方便识别）：
{pred_result.head(10).to_string(index=False)}

三、数据处理说明：
- 已兼容无id字段的情况（删除id前先检查存在性）
- 分类变量（如show_place）未知类别已映射为"其他"，避免编码报错
- 数值变量缺失值用中位数填充，分类变量缺失值用"未知"填充
- 处理规则：剔除id字段（若存在），name仅作为识别标识（不参与建模）
- 分类变量采用OneHotEncoder（drop='first'）处理，数值变量采用StandardScaler标准化
"""

# 保存报告
report_path = "box_office/basiline/LinearRegression/LinearRegression_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

# 打印结果
print("线性回归模型训练完成！")
print(f"评估报告已保存至：{os.path.abspath(report_path)}")
if y_test is not None:
    print(f"MAPE：{mape:.2%} | R²：{r2:.4f}")
