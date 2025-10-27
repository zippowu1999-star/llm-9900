"""
代码执行引擎示例

展示如何使用CodeExecutor安全执行Python代码
"""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.executor import CodeExecutor


def example_1_simple_execution():
    """示例1: 简单代码执行"""
    print("\n" + "=" * 60)
    print("示例1: 简单代码执行")
    print("=" * 60)
    
    executor = CodeExecutor(mode="subprocess", timeout=10)
    
    code = """
import pandas as pd
import numpy as np

print("开始数据处理...")

# 创建示例数据
data = {
    'id': range(1, 11),
    'value': np.random.rand(10)
}

df = pd.DataFrame(data)
print(f"数据形状: {df.shape}")
print(f"前5行:\\n{df.head()}")

print("数据处理完成！")
"""
    
    result = executor.execute(code)
    
    print(f"\n执行结果:")
    print(f"  成功: {result.success}")
    print(f"  耗时: {result.execution_time:.2f}秒")
    print(f"  返回码: {result.return_code}")
    
    print(f"\n输出:")
    print("-" * 60)
    print(result.output)
    print("-" * 60)
    
    if result.error:
        print(f"\n错误:")
        print(result.error)


def example_2_submission_generation():
    """示例2: 生成submission.csv"""
    print("\n" + "=" * 60)
    print("示例2: 生成submission.csv")
    print("=" * 60)
    
    # 创建临时工作目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        executor = CodeExecutor(mode="subprocess", timeout=30)
        
        code = """
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("步骤1: 生成模拟数据")
# 模拟训练数据
X_train = np.random.rand(100, 3)
y_train = X_train @ np.array([1.5, -2.0, 0.5]) + np.random.randn(100) * 0.1

# 模拟测试数据
X_test = np.random.rand(20, 3)
test_ids = range(1, 21)

print("步骤2: 训练模型")
model = LinearRegression()
model.fit(X_train, y_train)
print(f"  模型训练完成，系数: {model.coef_}")

print("步骤3: 生成预测")
predictions = model.predict(X_test)
print(f"  预测范围: [{predictions.min():.3f}, {predictions.max():.3f}]")

print("步骤4: 创建submission.csv")
submission = pd.DataFrame({
    'id': test_ids,
    'prediction': predictions
})

submission.to_csv('submission.csv', index=False)
print(f"✓ submission.csv 已生成")
print(f"  形状: {submission.shape}")
print(f"  前5行:\\n{submission.head()}")
"""
        
        result = executor.execute(code, working_dir=tmp_path)
        
        print(f"\n执行结果:")
        print(f"  成功: {result.success}")
        print(f"  耗时: {result.execution_time:.2f}秒")
        print(f"  Submission路径: {result.submission_path}")
        
        print(f"\n输出:")
        print("-" * 60)
        print(result.output)
        print("-" * 60)
        
        if result.submission_path:
            print(f"\n✓ submission.csv 已生成: {result.submission_path}")
            
            # 读取并显示
            import pandas as pd
            df = pd.read_csv(result.submission_path)
            print(f"\nSubmission内容 (前10行):")
            print(df.head(10))


def example_3_error_handling():
    """示例3: 错误处理"""
    print("\n" + "=" * 60)
    print("示例3: 错误处理")
    print("=" * 60)
    
    executor = CodeExecutor(mode="subprocess", timeout=10)
    
    code = """
import pandas as pd

print("步骤1: 正常输出")

print("步骤2: 触发错误")
# 故意触发错误
result = 1 / 0

print("步骤3: 这行不会执行")
"""
    
    result = executor.execute(code)
    
    print(f"\n执行结果:")
    print(f"  成功: {result.success}")
    print(f"  返回码: {result.return_code}")
    
    print(f"\n标准输出:")
    print(result.output)
    
    print(f"\n错误信息:")
    print("-" * 60)
    print(result.error)
    print("-" * 60)


def example_4_code_validation():
    """示例4: 代码验证"""
    print("\n" + "=" * 60)
    print("示例4: 代码验证")
    print("=" * 60)
    
    executor = CodeExecutor()
    
    test_cases = [
        ("print('Hello')", "有效代码"),
        ("print('Hello'", "缺少括号"),
        ("for i in range(10)\n    print(i)", "缺少冒号"),
        ("import pandas as pd\ndf = pd.DataFrame()", "正常导入"),
    ]
    
    for code, description in test_cases:
        is_valid, error = executor.validate_code(code)
        
        status = "✓" if is_valid else "✗"
        print(f"\n{status} {description}")
        print(f"  代码: {code[:50]}...")
        if not is_valid:
            print(f"  错误: {error}")


def main():
    """主函数"""
    print("=" * 60)
    print("代码执行引擎示例")
    print("=" * 60)
    
    try:
        example_1_simple_execution()
        example_2_submission_generation()
        example_3_error_handling()
        example_4_code_validation()
        
        print("\n" + "=" * 60)
        print("所有示例完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

