#!/usr/bin/env python3
"""
简单测试Titanic问题
直接测试代码生成，不运行完整流程
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def test_titanic_simple():
    """简单测试Titanic数据处理"""
    
    print("🧪 简单测试Titanic数据处理")
    print("=" * 50)
    
    # 数据路径
    input_path = '/Users/zhouzhuotao/9900pj/data/competitions/titanic'
    
    # 加载数据
    train = pd.read_csv(os.path.join(input_path, 'train.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test.csv'))
    
    print(f"✅ 训练数据形状: {train.shape}")
    print(f"✅ 测试数据形状: {test.shape}")
    print(f"✅ 训练数据列: {list(train.columns)}")
    print(f"✅ 测试数据列: {list(test.columns)}")
    
    # 特征工程
    def feature_engineering(df):
        df = df.copy()  # 避免SettingWithCopyWarning
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 1
        df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna('S')
        return df
    
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    print(f"✅ 特征工程完成")
    
    # 编码object列
    train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
    test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)
    
    print(f"✅ 编码完成")
    print(f"✅ 训练数据列数: {len(train.columns)}")
    print(f"✅ 测试数据列数: {len(test.columns)}")
    
    # 使用inner join对齐列（关键修复！）
    common_cols = list(set(train.columns) & set(test.columns))
    print(f"✅ 共同列数: {len(common_cols)}")
    print(f"✅ 共同列: {sorted(common_cols)}")
    
    # 只保留共同列
    train_aligned = train[common_cols]
    test_aligned = test[common_cols]
    
    print(f"✅ 对齐后训练数据形状: {train_aligned.shape}")
    print(f"✅ 对齐后测试数据形状: {test_aligned.shape}")
    
    # 准备训练数据
    feature_cols = [col for col in common_cols if col not in ['PassengerId', 'Name', 'Ticket', 'Cabin']]
    print(f"✅ 特征列: {feature_cols}")
    
    X = train_aligned[feature_cols]
    y = train['Survived']  # 直接从原始训练数据获取目标变量
    X_test = test_aligned[feature_cols]
    
    print(f"✅ 训练特征形状: {X.shape}")
    print(f"✅ 测试特征形状: {X_test.shape}")
    print(f"✅ 目标变量形状: {y.shape}")
    
    # 检查是否有缺失值
    print(f"✅ 训练特征缺失值: {X.isnull().sum().sum()}")
    print(f"✅ 测试特征缺失值: {X_test.isnull().sum().sum()}")
    
    # 模型训练
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 预测
    predictions = model.predict(X_test)
    
    print(f"✅ 预测完成，预测值范围: {predictions.min()} - {predictions.max()}")
    print(f"✅ 预测值分布: {np.bincount(predictions)}")
    
    # 生成提交文件
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    
    output_path = '/Users/zhouzhuotao/9900pj/test_submission.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"✅ 提交文件生成: {output_path}")
    print(f"✅ 提交文件形状: {submission.shape}")
    print(f"✅ 提交文件前5行:")
    print(submission.head())
    
    return True

if __name__ == "__main__":
    try:
        success = test_titanic_simple()
        if success:
            print("\n🎉 简单测试成功！")
        else:
            print("\n❌ 简单测试失败！")
    except Exception as e:
        print(f"\n💥 测试异常: {e}")
        import traceback
        traceback.print_exc()
