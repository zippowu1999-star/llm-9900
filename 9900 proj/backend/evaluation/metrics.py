"""
评估指标计算器

计算AI代理的各项性能指标
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from backend.agents.base_agent import AgentResult, AgentType
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentMetrics:
    """
    AI代理评估指标
    
    包含多个维度的评估指标
    """
    # 基本信息
    agent_type: str
    competition_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 性能指标
    total_time: float = 0.0  # 总耗时（秒）
    code_generation_time: float = 0.0  # 代码生成时间
    execution_time: float = 0.0  # 执行时间
    
    # 效率指标
    llm_calls: int = 0  # LLM调用次数
    llm_tokens: int = 0  # 使用的token数（如果可获取）
    code_lines: int = 0  # 生成的代码行数
    iterations: int = 0  # 迭代次数
    
    # 质量指标
    success: bool = False  # 是否成功生成submission
    code_quality_score: float = 0.0  # 代码质量分数（0-100）
    submission_valid: bool = False  # submission是否有效
    
    # 复杂度指标
    code_complexity: float = 0.0  # 代码复杂度
    feature_count: int = 0  # 使用的特征数量
    model_complexity: str = "unknown"  # 模型复杂度（simple/medium/complex）
    
    # 自主性指标
    autonomy_score: float = 0.0  # 自主性得分（0-100）
    human_interventions: int = 0  # 需要人工干预的次数
    
    # 可解释性指标
    explainability_score: float = 0.0  # 可解释性得分（0-100）
    comments_ratio: float = 0.0  # 注释比例
    thoughts_count: int = 0  # 记录的思考步骤数
    
    # 资源消耗
    memory_peak_mb: float = 0.0  # 峰值内存使用（MB）
    cpu_usage_percent: float = 0.0  # CPU使用率
    
    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 额外信息
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_type": self.agent_type,
            "competition_name": self.competition_name,
            "timestamp": self.timestamp,
            "total_time": self.total_time,
            "code_generation_time": self.code_generation_time,
            "execution_time": self.execution_time,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "code_lines": self.code_lines,
            "iterations": self.iterations,
            "success": self.success,
            "code_quality_score": self.code_quality_score,
            "submission_valid": self.submission_valid,
            "code_complexity": self.code_complexity,
            "feature_count": self.feature_count,
            "model_complexity": self.model_complexity,
            "autonomy_score": self.autonomy_score,
            "human_interventions": self.human_interventions,
            "explainability_score": self.explainability_score,
            "comments_ratio": self.comments_ratio,
            "thoughts_count": self.thoughts_count,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "errors": self.errors,
            "warnings": self.warnings,
            "extra_metrics": self.extra_metrics
        }
    
    def save(self, path: Path):
        """保存指标到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_overall_score(self) -> float:
        """
        计算综合得分（0-100）
        
        权重分配：
        - 成功性：30%
        - 效率：25%
        - 质量：25%
        - 自主性：10%
        - 可解释性：10%
        """
        scores = []
        
        # 1. 成功性 (30%)
        success_score = 100.0 if self.success and self.submission_valid else 0.0
        scores.append(success_score * 0.30)
        
        # 2. 效率 (25%)
        # 时间越短越好（假设300秒为基准）
        time_score = max(0, 100 - (self.total_time / 300) * 100)
        # LLM调用次数越少越好（假设10次为基准）
        llm_score = max(0, 100 - (self.llm_calls / 10) * 100)
        efficiency_score = (time_score + llm_score) / 2
        scores.append(efficiency_score * 0.25)
        
        # 3. 质量 (25%)
        scores.append(self.code_quality_score * 0.25)
        
        # 4. 自主性 (10%)
        scores.append(self.autonomy_score * 0.10)
        
        # 5. 可解释性 (10%)
        scores.append(self.explainability_score * 0.10)
        
        return sum(scores)


class MetricsCalculator:
    """
    指标计算器
    
    从AgentResult计算各种评估指标
    """
    
    def __init__(self):
        """初始化计算器"""
        logger.info("初始化MetricsCalculator")
    
    def calculate(self, result: AgentResult) -> AgentMetrics:
        """
        从AgentResult计算完整指标
        
        Args:
            result: Agent执行结果
            
        Returns:
            计算后的指标
        """
        logger.info(f"开始计算指标: {result.agent_type.value}")
        
        metrics = AgentMetrics(
            agent_type=result.agent_type.value,
            competition_name=result.competition_name
        )
        
        # 性能指标
        metrics.total_time = result.total_time
        metrics.code_generation_time = result.code_generation_time
        metrics.execution_time = result.execution_time
        
        # 效率指标
        metrics.llm_calls = result.llm_calls
        metrics.code_lines = result.code_lines
        metrics.iterations = len(result.actions)
        
        # 质量指标
        metrics.success = (result.submission_file_path is not None)
        metrics.submission_valid = (result.execution_error is None)
        
        # 分析代码质量
        if result.generated_code:
            metrics.code_quality_score = self._calculate_code_quality(result.generated_code)
            metrics.code_complexity = self._calculate_complexity(result.generated_code)
            metrics.comments_ratio = self._calculate_comments_ratio(result.generated_code)
            metrics.model_complexity = self._infer_model_complexity(result.generated_code)
        
        # 自主性评分
        metrics.autonomy_score = self._calculate_autonomy(result)
        
        # 可解释性评分
        metrics.explainability_score = self._calculate_explainability(result)
        metrics.thoughts_count = len(result.thoughts)
        
        # 错误和警告
        if result.error_message:
            metrics.errors.append(result.error_message)
        if result.execution_error:
            metrics.errors.append(result.execution_error)
        
        logger.info(f"✓ 指标计算完成，综合得分: {metrics.get_overall_score():.2f}")
        
        return metrics
    
    def _calculate_code_quality(self, code: str) -> float:
        """
        计算代码质量分数（0-100）
        
        考虑因素：
        - 代码长度适中
        - 有注释
        - 有错误处理
        - 有日志输出
        - 模块化
        """
        score = 50.0  # 基础分
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 1. 长度合理性 (+10分)
        if 50 <= len(non_empty_lines) <= 500:
            score += 10
        
        # 2. 有注释 (+15分)
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        if len(comment_lines) >= 5:
            score += 15
        elif len(comment_lines) >= 2:
            score += 8
        
        # 3. 有错误处理 (+10分)
        if 'try:' in code or 'except' in code:
            score += 10
        
        # 4. 有日志/打印输出 (+5分)
        if 'print(' in code or 'logger' in code:
            score += 5
        
        # 5. 有函数定义（模块化）(+10分)
        if 'def ' in code:
            score += 10
        
        # 6. 导入了常用库 (+10分)
        common_imports = ['pandas', 'numpy', 'sklearn']
        import_count = sum(1 for lib in common_imports if lib in code)
        score += min(10, import_count * 3)
        
        return min(100.0, score)
    
    def _calculate_complexity(self, code: str) -> float:
        """
        计算代码复杂度（圈复杂度的简化版本）
        
        Returns:
            复杂度分数（越低越好）
        """
        complexity = 1.0  # 基础复杂度
        
        # 控制流语句增加复杂度
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except']
        for keyword in control_keywords:
            complexity += code.count(f' {keyword} ') + code.count(f'\n{keyword} ')
        
        return complexity
    
    def _calculate_comments_ratio(self, code: str) -> float:
        """计算注释比例"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        
        if len(non_empty_lines) == 0:
            return 0.0
        
        return len(comment_lines) / len(non_empty_lines)
    
    def _infer_model_complexity(self, code: str) -> str:
        """
        推断模型复杂度
        
        Returns:
            "simple", "medium", "complex"
        """
        # 简单模型
        simple_models = ['LinearRegression', 'LogisticRegression', 'DecisionTree']
        # 中等模型
        medium_models = ['RandomForest', 'GradientBoosting', 'SVM', 'KNN']
        # 复杂模型
        complex_models = ['XGBoost', 'LightGBM', 'CatBoost', 'Neural', 'LSTM', 'Transformer']
        
        for model in complex_models:
            if model in code:
                return "complex"
        
        for model in medium_models:
            if model in code:
                return "medium"
        
        for model in simple_models:
            if model in code:
                return "simple"
        
        return "unknown"
    
    def _calculate_autonomy(self, result: AgentResult) -> float:
        """
        计算自主性得分（0-100）
        
        考虑因素：
        - 是否需要人工干预
        - 是否自动完成所有步骤
        - 错误恢复能力
        """
        score = 100.0
        
        # 如果有错误，降低自主性得分
        if result.error_message:
            score -= 30
        
        # 如果没有生成submission，说明自主性不足
        if not result.submission_file_path:
            score -= 40
        
        # 如果迭代次数过多，说明需要多次尝试
        if result.llm_calls > 10:
            score -= 10
        
        return max(0.0, score)
    
    def _calculate_explainability(self, result: AgentResult) -> float:
        """
        计算可解释性得分（0-100）
        
        考虑因素：
        - 是否有思考过程记录
        - 代码注释质量
        - 是否有中间输出
        """
        score = 0.0
        
        # 1. 有思考记录 (+40分)
        if len(result.thoughts) > 0:
            score += min(40, len(result.thoughts) * 10)
        
        # 2. 有动作记录 (+20分)
        if len(result.actions) > 0:
            score += min(20, len(result.actions) * 5)
        
        # 3. 有观察记录 (+20分)
        if len(result.observations) > 0:
            score += min(20, len(result.observations) * 5)
        
        # 4. 代码有注释 (+20分)
        if result.generated_code:
            comment_ratio = self._calculate_comments_ratio(result.generated_code)
            score += comment_ratio * 100 * 0.2
        
        return min(100.0, score)
    
    def calculate_batch(self, results: List[AgentResult]) -> List[AgentMetrics]:
        """
        批量计算指标
        
        Args:
            results: Agent结果列表
            
        Returns:
            指标列表
        """
        logger.info(f"批量计算指标: {len(results)} 个结果")
        metrics_list = []
        
        for i, result in enumerate(results, 1):
            logger.info(f"处理 {i}/{len(results)}: {result.agent_type.value}")
            metrics = self.calculate(result)
            metrics_list.append(metrics)
        
        logger.info("✓ 批量计算完成")
        return metrics_list

