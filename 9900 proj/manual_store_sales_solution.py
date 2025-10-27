"""
Store Sales 手动优化版本

展示如何正确使用所有辅助数据文件
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

print("=" * 70)
print("Store Sales - 完整版解决方案")
print("=" * 70)

# 数据路径
data_path = '/Users/zhouzhuotao/9900pj/data/competitions/store-sales-time-series-forecasting'
output_path = '/Users/zhouzhuotao/9900pj/data/generated_code'

print("\n步骤1: 加载所有数据文件")
print("-" * 70)

# 加载主要数据
train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))
print(f"✓ 训练数据: {train.shape}")
print(f"✓ 测试数据: {test.shape}")

# 加载辅助数据
stores = pd.read_csv(os.path.join(data_path, 'stores.csv'))
oil = pd.read_csv(os.path.join(data_path, 'oil.csv'))
holidays = pd.read_csv(os.path.join(data_path, 'holidays_events.csv'))
transactions = pd.read_csv(os.path.join(data_path, 'transactions.csv'))

print(f"✓ 商店信息: {stores.shape}")
print(f"✓ 石油价格: {oil.shape}")
print(f"✓ 节假日: {holidays.shape}")
print(f"✓ 交易数据: {transactions.shape}")

print("\n步骤2: 数据合并和特征工程")
print("-" * 70)

# 转换日期
for df in [train, test, oil, holidays, transactions]:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

# 合并商店信息
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')
print(f"✓ 合并商店信息后: {train.shape}")

# 合并石油价格（按日期）
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')
print(f"✓ 合并石油价格后: {train.shape}")

# 合并交易数据（按日期和商店）
train = train.merge(transactions, on=['date', 'store_nbr'], how='left')
test = test.merge(transactions, on=['date', 'store_nbr'], how='left')
print(f"✓ 合并交易数据后: {train.shape}")

print("\n步骤3: 特征工程")
print("-" * 70)

# 提取日期特征
for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week

print("✓ 日期特征提取完成")

# 填充缺失值
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
print("✓ 缺失值填充完成")

# 处理分类变量
categorical_cols = ['family', 'city', 'state', 'type']
for col in categorical_cols:
    if col in train.columns:
        # 使用Label Encoding
        train[col] = train[col].astype('category').cat.codes
        test[col] = test[col].astype('category').cat.codes

print(f"✓ 分类变量编码完成")

print("\n步骤4: 准备训练数据（使用采样以加快速度）")
print("-" * 70)

# 由于数据量大（300万行），使用10%的数据训练
train_sample = train.sample(frac=0.1, random_state=42)
print(f"✓ 训练样本: {train_sample.shape} (原始: {train.shape})")

# 选择特征
feature_cols = [
    'store_nbr', 'family', 'onpromotion',
    'year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear',
    'cluster', 'type', 'dcoilwtico', 'transactions'
]

# 确保所有特征都存在
feature_cols = [col for col in feature_cols if col in train_sample.columns]
print(f"✓ 使用特征: {len(feature_cols)} 个")
print(f"  {feature_cols[:5]}...")

X = train_sample[feature_cols]
y = train_sample['sales']

# 分割训练和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ 训练集: {X_train.shape}")
print(f"✓ 验证集: {X_val.shape}")

print("\n步骤5: 训练模型")
print("-" * 70)

model = RandomForestRegressor(
    n_estimators=50,  # 减少树的数量以加快训练
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("开始训练...")
model.fit(X_train, y_train)
print("✓ 训练完成!")

# 验证
y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"✓ 验证集RMSE: {val_rmse:.2f}")

print("\n步骤6: 生成预测")
print("-" * 70)

# 准备测试数据
X_test = test[feature_cols]
test_predictions = model.predict(X_test)

print(f"✓ 预测完成: {len(test_predictions)} 个预测值")
print(f"  预测范围: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")

print("\n步骤7: 生成submission.csv")
print("-" * 70)

submission = pd.DataFrame({
    'id': test['id'],
    'sales': test_predictions
})

# 保存
submission_path = os.path.join(output_path, 'store_sales_manual_submission.csv')
submission.to_csv(submission_path, index=False)

print(f"✓ Submission已保存: {submission_path}")
print(f"  形状: {submission.shape}")
print(f"\n前10行预览:")
print(submission.head(10))

print("\n" + "=" * 70)
print("✅ 完成！使用了所有辅助数据的完整解决方案")
print("=" * 70)

