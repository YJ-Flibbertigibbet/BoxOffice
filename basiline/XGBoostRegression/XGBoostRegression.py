import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import os

# ===================== 1. 数据加载与预处理 =====================
train_path = "box_office/data/train_ceshi.xlsx"
test_path = "box_office/data/test_ceshi.xlsx"

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

# 数据处理：兼容无id字段，处理分类变量未知类别
def process_data(df, is_train=True, cat_feature_categories=None):
    df = df.copy()
    
    # 1. 剔除id字段（仅当存在时）
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    
    # 2. 保留name作为识别列
    name_series = df["name"] if "name" in df.columns else pd.Series([f"影片_{i}" for i in range(len(df))])
    
    # 3. 分离特征和目标变量
    X = df.drop(["name", "票房_万"], axis=1) if "票房_万" in df.columns else df.drop("name", axis=1)
    y = df["票房_万"] if "票房_万" in df.columns else None
    
    # 4. 定义并过滤分类/数值变量
    cat_features = [
        "holiday", "show_place", "类型_剧情", "类型_动作", "类型_犯罪",
        "类型_悬疑", "类型_爱情", "类型_科幻", "类型_喜剧", "类型_冒险",
        "类型_战争", "类型_历史", "类型_家庭", "year", "month", "weekday"
    ]
    cat_features = [col for col in cat_features if col in X.columns]
    num_features = [col for col in X.columns if col not in cat_features]
    
    # 5. 处理分类变量未知类别
    if is_train:
        cat_feature_categories = {}
        for col in cat_features:
            X[col] = X[col].fillna("未知")
            cat_feature_categories[col] = X[col].unique().tolist()
    else:
        for col in cat_features:
            X[col] = X[col].fillna("未知")
            known_cats = cat_feature_categories[col]
            X[col] = X[col].apply(lambda x: x if x in known_cats else "其他")
    
    return name_series, X, y, cat_features, num_features, cat_feature_categories

# 处理训练集
train_names, X_train, y_train, cat_features_train, num_features_train, cat_categories = process_data(train_df, is_train=True)

# 处理测试集
test_names, X_test, y_test, _, _, _ = process_data(test_df, is_train=False, cat_feature_categories=cat_categories)

# ===================== 2. 构建预处理管道 =====================
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="未知")),
    ("onehot", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features_train),
        ("cat", cat_transformer, cat_features_train)
    ]
)

# ===================== 3. 模型构建与训练 =====================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# ===================== 4. 预测与结果关联 =====================
y_pred = model.predict(X_test)
pred_result = pd.DataFrame({
    "影片名称": test_names.values,
    "真实票房_万": y_test.values if y_test is not None else np.nan,
    "预测票房_万": y_pred.round(2)
})

# ===================== 5. 评估与报告生成 =====================
if y_test is not None:
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
else:
    mae = mse = rmse = mape = r2 = np.nan

# 特征重要性
feature_names = []
if num_features_train:
    feature_names += num_features_train
if cat_features_train:
    feature_names += list(model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_features_train))

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.named_steps["regressor"].feature_importances_
}).sort_values("importance", ascending=False).head(10)

# 生成报告
report_content = f"""
===================== XGBoost回归模型评估报告 =====================
一、模型参数：
- objective=reg:squarederror（回归目标）
- learning_rate=0.05（学习率）
- max_depth=6（树深度）
- n_estimators=200（树数量）

二、核心评估指标（测试集）：
1. 平均绝对误差（MAE）：{mae:.2f} 万
2. 均方误差（MSE）：{mse:.2f} 万²
3. 均方根误差（RMSE）：{rmse:.2f} 万
4. 平均百分比误差（MAPE）：{mape:.2%}
5. 决定系数（R²）：{r2:.4f}

三、部分影片预测结果（前10条，方便识别）：
{pred_result.head(10).to_string(index=False)}

四、TOP10重要特征：
{feature_importance.to_string(index=False)}

五、补充说明：
- XGBoost是票房预测最优模型之一，适合捕捉非线性特征交互（如首日票房×类型）
- 调参方向：learning_rate=0.01/0.1，max_depth=4/8，n_estimators=100/300
- 已处理：1. 无id字段兼容 2. show_place未知类别映射为"其他"
- 分类变量处理：OneHotEncoder（handle_unknown="ignore"），数值变量：StandardScaler
"""

# 保存报告
report_path = "box_office/basiline/XGBoostRegression/XGBoostRegression_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print("XGBoost回归模型训练完成！")
print(f"评估报告已保存至：{os.path.abspath(report_path)}")
if y_test is not None:
    print(f"MAPE：{mape:.2%} | R²：{r2:.4f}")
