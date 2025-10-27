"""
代码执行引擎测试
"""
import pytest
import time
from pathlib import Path
from backend.executor import CodeExecutor, ExecutionResult


class TestCodeExecutor:
    """测试CodeExecutor"""
    
    def test_simple_execution(self, tmp_path):
        """测试简单代码执行"""
        executor = CodeExecutor(mode="subprocess", timeout=10)
        
        code = """
print("Hello, World!")
print("Test execution")
"""
        
        result = executor.execute(code, working_dir=tmp_path)
        
        assert "Hello, World!" in result.output
        assert "Test execution" in result.output
        assert result.execution_time > 0
    
    def test_submission_generation(self, tmp_path):
        """测试生成submission.csv"""
        executor = CodeExecutor(mode="subprocess", timeout=10)
        
        code = """
import pandas as pd

# 创建submission
df = pd.DataFrame({
    'id': [1, 2, 3],
    'prediction': [0.5, 0.7, 0.3]
})

df.to_csv('submission.csv', index=False)
print("Submission created")
"""
        
        result = executor.execute(code, working_dir=tmp_path)
        
        assert result.success
        assert result.submission_path is not None
        assert Path(result.submission_path).exists()
        
        # 验证文件内容
        import pandas as pd
        df = pd.read_csv(result.submission_path)
        assert len(df) == 3
        assert list(df.columns) == ['id', 'prediction']
    
    def test_error_handling(self, tmp_path):
        """测试错误处理"""
        executor = CodeExecutor(mode="subprocess", timeout=10)
        
        code = """
print("Before error")
raise ValueError("Test error")
print("After error")
"""
        
        result = executor.execute(code, working_dir=tmp_path)
        
        assert not result.success
        assert result.return_code != 0
        assert "ValueError" in result.error
        assert "Before error" in result.output
    
    def test_timeout(self, tmp_path):
        """测试超时"""
        executor = CodeExecutor(mode="subprocess", timeout=2)
        
        code = """
import time
print("Sleeping...")
time.sleep(10)  # 超过timeout
print("Done")
"""
        
        result = executor.execute(code, working_dir=tmp_path)
        
        assert not result.success
        assert "超时" in result.error or "timeout" in result.error.lower()
        assert result.execution_time >= 2
    
    def test_code_validation(self):
        """测试代码验证"""
        executor = CodeExecutor()
        
        # 有效代码
        valid_code = "print('Hello')"
        is_valid, error = executor.validate_code(valid_code)
        assert is_valid
        assert error is None
        
        # 无效代码
        invalid_code = "print('Hello'"  # 缺少括号
        is_valid, error = executor.validate_code(invalid_code)
        assert not is_valid
        assert error is not None
    
    def test_execute_file(self, tmp_path):
        """测试执行文件"""
        executor = CodeExecutor(mode="subprocess", timeout=10)
        
        # 创建测试文件
        code_file = tmp_path / "test_script.py"
        code_file.write_text("""
import pandas as pd
df = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
df.to_csv('submission.csv', index=False)
print("File execution test")
""")
        
        result = executor.execute_file(code_file)
        
        assert result.success
        assert "File execution test" in result.output
        assert result.submission_path is not None


class TestSafeCodeExecutor:
    """测试SafeCodeExecutor"""
    
    def test_dangerous_code_detection(self, tmp_path):
        """测试危险代码检测"""
        from backend.executor.code_executor import SafeCodeExecutor
        
        executor = SafeCodeExecutor(mode="subprocess", timeout=10)
        
        # 包含危险代码
        code = """
import os
os.system('echo hello')
"""
        
        # 应该警告但仍执行（当前实现）
        result = executor.execute(code, working_dir=tmp_path)
        # 注意：实际生产环境应该阻止执行


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

