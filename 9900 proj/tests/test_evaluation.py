"""
评估模块测试
"""
import pytest
from pathlib import Path
from backend.agents.base_agent import AgentResult, AgentType, AgentStatus
from backend.evaluation import MetricsCalculator, AgentComparator, AgentMetrics


@pytest.fixture
def sample_result():
    """创建示例结果"""
    result = AgentResult(
        agent_type=AgentType.REACT,
        competition_name="test-competition",
        status=AgentStatus.COMPLETED
    )
    
    result.generated_code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 训练模型
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

# 保存提交
submission = pd.DataFrame({'id': test_ids, 'prediction': predictions})
submission.to_csv('submission.csv', index=False)
"""
    
    result.total_time = 45.5
    result.code_generation_time = 15.2
    result.execution_time = 30.3
    result.llm_calls = 3
    result.code_lines = len(result.generated_code.split('\n'))
    result.submission_file_path = Path("submission.csv")
    result.thoughts = ["分析问题", "选择模型", "生成代码"]
    result.actions = [{"action": "analyze"}, {"action": "generate"}]
    result.observations = ["数据加载成功", "模型训练完成"]
    
    return result


class TestMetricsCalculator:
    """测试MetricsCalculator"""
    
    def test_calculate_metrics(self, sample_result):
        """测试指标计算"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(sample_result)
        
        assert metrics.agent_type == "react"
        assert metrics.total_time == 45.5
        assert metrics.llm_calls == 3
        assert metrics.success == True
        assert metrics.code_lines > 0
        assert metrics.code_quality_score > 0
        assert metrics.autonomy_score > 0
        assert metrics.explainability_score > 0
    
    def test_code_quality_calculation(self, sample_result):
        """测试代码质量计算"""
        calculator = MetricsCalculator()
        
        code = sample_result.generated_code
        quality_score = calculator._calculate_code_quality(code)
        
        assert 0 <= quality_score <= 100
        assert quality_score > 50  # 示例代码应该有合理的质量分
    
    def test_complexity_calculation(self, sample_result):
        """测试复杂度计算"""
        calculator = MetricsCalculator()
        
        code = sample_result.generated_code
        complexity = calculator._calculate_complexity(code)
        
        assert complexity > 0
    
    def test_model_complexity_inference(self, sample_result):
        """测试模型复杂度推断"""
        calculator = MetricsCalculator()
        
        # RandomForest应该被识别为medium
        complexity = calculator._infer_model_complexity(sample_result.generated_code)
        assert complexity == "medium"
    
    def test_overall_score(self, sample_result):
        """测试综合得分"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(sample_result)
        
        overall_score = metrics.get_overall_score()
        assert 0 <= overall_score <= 100
    
    def test_metrics_to_dict(self, sample_result):
        """测试指标序列化"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(sample_result)
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert "agent_type" in metrics_dict
        assert "total_time" in metrics_dict
        assert "code_quality_score" in metrics_dict


class TestAgentComparator:
    """测试AgentComparator"""
    
    @pytest.fixture
    def multiple_metrics(self, sample_result):
        """创建多个代理的指标"""
        calculator = MetricsCalculator()
        
        # React代理
        metrics_react = calculator.calculate(sample_result)
        
        # RAG代理（模拟）
        result_rag = sample_result
        result_rag.agent_type = AgentType.RAG
        result_rag.total_time = 35.0  # 更快
        result_rag.llm_calls = 5  # 更多调用
        metrics_rag = calculator.calculate(result_rag)
        
        # Multi-Agent（模拟）
        result_multi = sample_result
        result_multi.agent_type = AgentType.MULTI_AGENT
        result_multi.total_time = 55.0  # 更慢
        result_multi.llm_calls = 8  # 最多调用
        metrics_multi = calculator.calculate(result_multi)
        
        return [metrics_react, metrics_rag, metrics_multi]
    
    def test_comparison(self, multiple_metrics):
        """测试代理比较"""
        comparator = AgentComparator()
        report = comparator.compare(multiple_metrics, "test-competition")
        
        assert report.competition_name == "test-competition"
        assert len(report.agents) == 3
        assert len(report.overall_ranking) == 3
        assert len(report.conclusions) > 0
        assert len(report.recommendations) > 0
    
    def test_rankings(self, multiple_metrics):
        """测试排名计算"""
        comparator = AgentComparator()
        report = comparator.compare(multiple_metrics)
        
        # 应该有时间排名
        assert "total_time" in report.rankings
        # 每个代理都有排名
        assert len(report.rankings["total_time"]) == 3
    
    def test_best_performers(self, multiple_metrics):
        """测试最佳表现者识别"""
        comparator = AgentComparator()
        report = comparator.compare(multiple_metrics)
        
        assert len(report.best_performer) > 0
        # 应该识别出最快的代理
        assert "total_time" in report.best_performer
    
    def test_report_to_dict(self, multiple_metrics):
        """测试报告序列化"""
        comparator = AgentComparator()
        report = comparator.compare(multiple_metrics)
        
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "agents" in report_dict
        assert "overall_ranking" in report_dict
    
    def test_markdown_generation(self, multiple_metrics):
        """测试Markdown报告生成"""
        comparator = AgentComparator()
        report = comparator.compare(multiple_metrics)
        
        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# AI代理性能对比报告" in markdown
        assert "综合排名" in markdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

