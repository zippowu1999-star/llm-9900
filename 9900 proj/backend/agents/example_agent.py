"""
示例代理实现

展示如何继承BaseAgent创建具体的代理
"""
from typing import Any, Dict
from backend.agents.base_agent import BaseAgent, AgentConfig
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ExampleAgent(BaseAgent):
    """
    示例代理
    
    这是一个简单的示例，展示如何实现BaseAgent的抽象方法
    实际的ReAct、RAG、Multi-Agent会有更复杂的实现
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._log("ExampleAgent 初始化完成")
    
    async def analyze_problem(self, problem_description: str) -> Dict[str, Any]:
        """
        分析问题描述
        
        实际实现中，这里会调用LLM来分析问题
        """
        self._log("开始分析问题...")
        
        # 模拟LLM分析
        analysis = {
            "problem_type": "time_series_forecasting",
            "key_insights": [
                "需要预测未来的销售额",
                "数据包含时间序列特征",
                "可能需要考虑季节性和趋势"
            ],
            "suggested_approach": "使用时间序列模型或梯度提升模型",
            "data_requirements": [
                "训练数据：历史销售记录",
                "测试数据：需要预测的时间点",
                "特征：日期、店铺、产品等"
            ]
        }
        
        self.result.llm_calls += 1
        self._log(f"问题分析完成: {analysis['problem_type']}")
        
        return analysis
    
    async def generate_code(
        self,
        problem_analysis: Dict[str, Any],
        data_info: Dict[str, Any]
    ) -> str:
        """
        生成数据分析代码
        
        实际实现中，这里会使用LLM根据问题分析和数据信息生成代码
        """
        self._log("开始生成代码...")
        
        # 模拟代码生成
        code_template = '''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 加载数据
print("加载数据...")
train_df = pd.read_csv('{data_path}/train.csv')
test_df = pd.read_csv('{data_path}/test.csv')

print(f"训练集形状: {{train_df.shape}}")
print(f"测试集形状: {{test_df.shape}}")

# 数据预处理
print("\\n数据预处理...")

# 提取时间特征（如果有日期列）
if 'date' in train_df.columns:
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df['year'] = train_df['date'].dt.year
    train_df['month'] = train_df['date'].dt.month
    train_df['day'] = train_df['date'].dt.day
    train_df['dayofweek'] = train_df['date'].dt.dayofweek

if 'date' in test_df.columns:
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df['year'] = test_df['date'].dt.year
    test_df['month'] = test_df['date'].dt.month
    test_df['day'] = test_df['date'].dt.day
    test_df['dayofweek'] = test_df['date'].dt.dayofweek

# 选择特征
feature_cols = [col for col in train_df.columns if col not in ['id', 'target', 'date']]
target_col = 'sales' if 'sales' in train_df.columns else train_df.columns[-1]

print(f"特征列: {{feature_cols[:5]}}... (共{{len(feature_cols)}}列)")
print(f"目标列: {{target_col}}")

# 处理缺失值
train_df = train_df.fillna(train_df.median(numeric_only=True))
test_df = test_df.fillna(test_df.median(numeric_only=True))

# 准备训练数据
X = train_df[feature_cols].select_dtypes(include=[np.number])
y = train_df[target_col]

# 训练模型
print("\\n训练模型...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# 预测
print("\\n生成预测...")
X_test = test_df[feature_cols].select_dtypes(include=[np.number])
predictions = model.predict(X_test)

# 创建提交文件
print("\\n创建submission.csv...")
submission = pd.DataFrame({{
    'id': test_df['id'] if 'id' in test_df.columns else range(len(predictions)),
    'prediction': predictions
}})

submission.to_csv('{output_dir}/submission.csv', index=False)
print(f"✓ submission.csv 已保存")
print(f"  形状: {{submission.shape}}")
print(f"  预测范围: [{{predictions.min():.2f}}, {{predictions.max():.2f}}]")
'''
        
        # 填充模板
        generated_code = code_template.format(
            data_path=self.config.data_path,
            output_dir=self.config.output_dir
        )
        
        self.result.llm_calls += 1
        self._log(f"代码生成完成: {len(generated_code)} 字符")
        
        return generated_code
    
    async def execute_code(self, code: str) -> Dict[str, Any]:
        """
        执行生成的代码
        
        实际实现中，这里会在沙箱环境中执行代码
        """
        self._log("开始执行代码...")
        
        try:
            # 这里应该使用代码执行引擎
            # 目前只是模拟
            submission_path = self.config.output_dir / "submission.csv"
            
            # 模拟执行结果
            result = {
                "success": True,
                "output": "代码执行成功（示例）",
                "error": None,
                "submission_path": str(submission_path)
            }
            
            self._log("代码执行完成")
            return result
            
        except Exception as e:
            self._log(f"代码执行失败: {str(e)}", level="error")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "submission_path": None
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取评估指标"""
        return {
            "total_time": self.result.total_time,
            "code_generation_time": self.result.code_generation_time,
            "execution_time": self.result.execution_time,
            "llm_calls": self.result.llm_calls,
            "code_lines": self.result.code_lines,
            "success": self.result.status.value == "completed"
        }

