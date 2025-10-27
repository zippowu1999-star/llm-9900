"""
AI代理比较器

对比不同架构代理的性能
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from backend.evaluation.metrics import AgentMetrics
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonReport:
    """
    比较报告
    
    包含多个代理的性能对比结果
    """
    competition_name: str
    agents: List[str] = field(default_factory=list)  # 参与比较的代理类型
    metrics_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 排名
    rankings: Dict[str, Dict[str, int]] = field(default_factory=dict)  # 各项指标的排名
    overall_ranking: List[tuple[str, float]] = field(default_factory=list)  # 综合排名
    
    # 最佳表现
    best_performer: Dict[str, str] = field(default_factory=dict)  # 各项指标的最佳代理
    
    # 统计分析
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # 结论和建议
    conclusions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "competition_name": self.competition_name,
            "agents": self.agents,
            "metrics_comparison": self.metrics_comparison,
            "rankings": self.rankings,
            "overall_ranking": self.overall_ranking,
            "best_performer": self.best_performer,
            "statistics": self.statistics,
            "conclusions": self.conclusions,
            "recommendations": self.recommendations
        }
    
    def save(self, path: Path):
        """保存报告"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """生成Markdown格式报告"""
        lines = []
        lines.append(f"# AI代理性能对比报告")
        lines.append(f"\n## 竞赛: {self.competition_name}")
        lines.append(f"\n参与对比的代理: {', '.join(self.agents)}")
        
        # 综合排名
        lines.append(f"\n## 综合排名")
        lines.append("\n| 排名 | 代理类型 | 综合得分 |")
        lines.append("|------|----------|----------|")
        for i, (agent, score) in enumerate(self.overall_ranking, 1):
            lines.append(f"| {i} | {agent} | {score:.2f} |")
        
        # 各项指标对比
        lines.append(f"\n## 详细指标对比")
        
        if self.metrics_comparison:
            # 选择关键指标展示
            key_metrics = [
                "total_time", "code_generation_time", "execution_time",
                "llm_calls", "code_lines", "code_quality_score",
                "autonomy_score", "explainability_score"
            ]
            
            for metric in key_metrics:
                if metric in self.metrics_comparison:
                    lines.append(f"\n### {metric}")
                    lines.append("\n| 代理 | 值 |")
                    lines.append("|------|-----|")
                    for agent, value in self.metrics_comparison[metric].items():
                        lines.append(f"| {agent} | {value:.2f} |")
        
        # 最佳表现
        lines.append(f"\n## 各项指标最佳代理")
        lines.append("\n| 指标 | 最佳代理 |")
        lines.append("|------|----------|")
        for metric, agent in self.best_performer.items():
            lines.append(f"| {metric} | {agent} |")
        
        # 结论
        if self.conclusions:
            lines.append(f"\n## 结论")
            for conclusion in self.conclusions:
                lines.append(f"\n- {conclusion}")
        
        # 建议
        if self.recommendations:
            lines.append(f"\n## 建议")
            for recommendation in self.recommendations:
                lines.append(f"\n- {recommendation}")
        
        return "\n".join(lines)


class AgentComparator:
    """
    代理比较器
    
    对比多个代理的性能指标
    """
    
    def __init__(self):
        """初始化比较器"""
        logger.info("初始化AgentComparator")
    
    def compare(
        self,
        metrics_list: List[AgentMetrics],
        competition_name: Optional[str] = None
    ) -> ComparisonReport:
        """
        比较多个代理的指标
        
        Args:
            metrics_list: 指标列表
            competition_name: 竞赛名称
            
        Returns:
            比较报告
        """
        if len(metrics_list) < 2:
            logger.warning("至少需要2个代理进行比较")
        
        logger.info(f"开始比较 {len(metrics_list)} 个代理")
        
        # 创建报告
        report = ComparisonReport(
            competition_name=competition_name or metrics_list[0].competition_name,
            agents=[m.agent_type for m in metrics_list]
        )
        
        # 1. 提取和对比指标
        report.metrics_comparison = self._extract_metrics(metrics_list)
        
        # 2. 计算排名
        report.rankings = self._calculate_rankings(report.metrics_comparison)
        
        # 3. 计算综合排名
        report.overall_ranking = self._calculate_overall_ranking(metrics_list)
        
        # 4. 找出最佳表现者
        report.best_performer = self._find_best_performers(report.metrics_comparison)
        
        # 5. 生成统计分析
        report.statistics = self._generate_statistics(report.metrics_comparison)
        
        # 6. 生成结论
        report.conclusions = self._generate_conclusions(report)
        
        # 7. 生成建议
        report.recommendations = self._generate_recommendations(report)
        
        logger.info("✓ 比较完成")
        return report
    
    def _extract_metrics(
        self,
        metrics_list: List[AgentMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """提取所有指标值"""
        comparison = {}
        
        # 获取所有指标名称
        metric_names = [
            "total_time", "code_generation_time", "execution_time",
            "llm_calls", "code_lines", "code_quality_score",
            "code_complexity", "autonomy_score", "explainability_score",
            "comments_ratio", "thoughts_count"
        ]
        
        for metric_name in metric_names:
            comparison[metric_name] = {}
            for metrics in metrics_list:
                value = getattr(metrics, metric_name, 0)
                comparison[metric_name][metrics.agent_type] = float(value)
        
        return comparison
    
    def _calculate_rankings(
        self,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, int]]:
        """
        计算各项指标的排名
        
        Returns:
            {metric_name: {agent_type: rank}}
        """
        rankings = {}
        
        # 定义哪些指标是越小越好
        lower_is_better = {
            "total_time", "code_generation_time", "execution_time",
            "llm_calls", "code_complexity"
        }
        
        for metric_name, values in metrics_comparison.items():
            # 排序
            sorted_agents = sorted(
                values.items(),
                key=lambda x: x[1],
                reverse=(metric_name not in lower_is_better)
            )
            
            # 分配排名
            rankings[metric_name] = {}
            for rank, (agent, _) in enumerate(sorted_agents, 1):
                rankings[metric_name][agent] = rank
        
        return rankings
    
    def _calculate_overall_ranking(
        self,
        metrics_list: List[AgentMetrics]
    ) -> List[tuple[str, float]]:
        """
        计算综合排名
        
        Returns:
            [(agent_type, overall_score)]，按分数降序
        """
        scores = []
        for metrics in metrics_list:
            overall_score = metrics.get_overall_score()
            scores.append((metrics.agent_type, overall_score))
        
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _find_best_performers(
        self,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        """找出各项指标的最佳代理"""
        best_performers = {}
        
        lower_is_better = {
            "total_time", "code_generation_time", "execution_time",
            "llm_calls", "code_complexity"
        }
        
        for metric_name, values in metrics_comparison.items():
            if not values:
                continue
            
            if metric_name in lower_is_better:
                best_agent = min(values.items(), key=lambda x: x[1])[0]
            else:
                best_agent = max(values.items(), key=lambda x: x[1])[0]
            
            best_performers[metric_name] = best_agent
        
        return best_performers
    
    def _generate_statistics(
        self,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """生成统计信息"""
        statistics = {}
        
        for metric_name, values in metrics_comparison.items():
            if not values:
                continue
            
            values_list = list(values.values())
            statistics[metric_name] = {
                "min": min(values_list),
                "max": max(values_list),
                "mean": sum(values_list) / len(values_list),
                "range": max(values_list) - min(values_list)
            }
        
        return statistics
    
    def _generate_conclusions(self, report: ComparisonReport) -> List[str]:
        """生成结论"""
        conclusions = []
        
        if report.overall_ranking:
            best_agent = report.overall_ranking[0][0]
            best_score = report.overall_ranking[0][1]
            conclusions.append(
                f"{best_agent} 表现最佳，综合得分 {best_score:.2f}"
            )
        
        # 分析时间效率
        if "total_time" in report.metrics_comparison:
            times = report.metrics_comparison["total_time"]
            fastest = min(times.items(), key=lambda x: x[1])
            slowest = max(times.items(), key=lambda x: x[1])
            conclusions.append(
                f"{fastest[0]} 最快 ({fastest[1]:.2f}秒)，"
                f"{slowest[0]} 最慢 ({slowest[1]:.2f}秒)"
            )
        
        # 分析代码质量
        if "code_quality_score" in report.metrics_comparison:
            quality = report.metrics_comparison["code_quality_score"]
            best_quality = max(quality.items(), key=lambda x: x[1])
            conclusions.append(
                f"{best_quality[0]} 代码质量最高 ({best_quality[1]:.2f}分)"
            )
        
        # 分析自主性
        if "autonomy_score" in report.metrics_comparison:
            autonomy = report.metrics_comparison["autonomy_score"]
            most_autonomous = max(autonomy.items(), key=lambda x: x[1])
            conclusions.append(
                f"{most_autonomous[0]} 自主性最强 ({most_autonomous[1]:.2f}分)"
            )
        
        return conclusions
    
    def _generate_recommendations(self, report: ComparisonReport) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于综合排名给出建议
        if report.overall_ranking:
            best_agent = report.overall_ranking[0][0]
            recommendations.append(
                f"对于{report.competition_name}类型的问题，推荐使用 {best_agent} 架构"
            )
        
        # 基于时间效率
        if "total_time" in report.best_performer:
            fastest = report.best_performer["total_time"]
            recommendations.append(
                f"如果注重速度，选择 {fastest}"
            )
        
        # 基于代码质量
        if "code_quality_score" in report.best_performer:
            best_quality = report.best_performer["code_quality_score"]
            recommendations.append(
                f"如果注重代码质量，选择 {best_quality}"
            )
        
        # 基于可解释性
        if "explainability_score" in report.best_performer:
            most_explainable = report.best_performer["explainability_score"]
            recommendations.append(
                f"如果需要高可解释性，选择 {most_explainable}"
            )
        
        return recommendations
    
    def generate_visualization_data(
        self,
        report: ComparisonReport
    ) -> Dict[str, Any]:
        """
        生成可视化数据（用于前端绘图）
        
        Returns:
            适合Plotly或其他可视化库使用的数据结构
        """
        viz_data = {
            "radar_chart": self._prepare_radar_chart_data(report),
            "bar_chart": self._prepare_bar_chart_data(report),
            "time_comparison": self._prepare_time_chart_data(report),
        }
        
        return viz_data
    
    def _prepare_radar_chart_data(self, report: ComparisonReport) -> Dict:
        """准备雷达图数据（综合能力对比）"""
        # 选择关键维度
        dimensions = [
            "code_quality_score",
            "autonomy_score",
            "explainability_score"
        ]
        
        data = {"agents": report.agents, "dimensions": [], "values": {}}
        
        for dim in dimensions:
            if dim in report.metrics_comparison:
                data["dimensions"].append(dim)
                for agent in report.agents:
                    if agent not in data["values"]:
                        data["values"][agent] = []
                    value = report.metrics_comparison[dim].get(agent, 0)
                    data["values"][agent].append(value)
        
        return data
    
    def _prepare_bar_chart_data(self, report: ComparisonReport) -> Dict:
        """准备柱状图数据（综合得分对比）"""
        data = {
            "agents": [agent for agent, _ in report.overall_ranking],
            "scores": [score for _, score in report.overall_ranking]
        }
        return data
    
    def _prepare_time_chart_data(self, report: ComparisonReport) -> Dict:
        """准备时间对比图数据"""
        time_metrics = ["code_generation_time", "execution_time"]
        
        data = {"agents": report.agents, "metrics": [], "values": {}}
        
        for metric in time_metrics:
            if metric in report.metrics_comparison:
                data["metrics"].append(metric)
                for agent in report.agents:
                    if agent not in data["values"]:
                        data["values"][agent] = {}
                    data["values"][agent][metric] = report.metrics_comparison[metric].get(agent, 0)
        
        return data

