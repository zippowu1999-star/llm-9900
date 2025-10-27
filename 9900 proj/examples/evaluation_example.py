"""
评估模块示例

展示如何使用MetricsCalculator和AgentComparator
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents.base_agent import AgentResult, AgentType, AgentStatus
from backend.evaluation import MetricsCalculator, AgentComparator


def create_sample_result(agent_type: AgentType, time: float, llm_calls: int) -> AgentResult:
    """创建示例结果"""
    result = AgentResult(
        agent_type=agent_type,
        competition_name="store-sales-forecasting",
        status=AgentStatus.COMPLETED
    )
    
    result.generated_code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 加载数据
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 特征工程
print("Feature engineering...")
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day

# 准备特征和标签
feature_cols = ['year', 'month', 'day', 'store_id', 'item_id']
X = train_df[feature_cols]
y = train_df['sales']

# 训练模型
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 预测
print("Making predictions...")
X_test = test_df[feature_cols]
predictions = model.predict(X_test)

# 生成提交文件
print("Creating submission...")
submission = pd.DataFrame({
    'id': test_df['id'],
    'sales': predictions
})
submission.to_csv('submission.csv', index=False)
print("Done!")
"""
    
    result.total_time = time
    result.code_generation_time = time * 0.3
    result.execution_time = time * 0.7
    result.llm_calls = llm_calls
    result.code_lines = len(result.generated_code.split('\n'))
    result.submission_file_path = Path("submission.csv")
    result.thoughts = [f"Thought {i}" for i in range(llm_calls)]
    result.actions = [{"action": f"action_{i}"} for i in range(llm_calls)]
    result.observations = [f"Observation {i}" for i in range(llm_calls)]
    
    return result


def example_1_single_metrics():
    """示例1: 计算单个代理的指标"""
    print("\n" + "=" * 60)
    print("示例1: 计算单个代理的指标")
    print("=" * 60)
    
    # 创建示例结果
    result = create_sample_result(AgentType.REACT, time=45.5, llm_calls=3)
    
    # 计算指标
    calculator = MetricsCalculator()
    metrics = calculator.calculate(result)
    
    # 显示结果
    print(f"\n代理类型: {metrics.agent_type}")
    print(f"竞赛名称: {metrics.competition_name}")
    print(f"\n性能指标:")
    print(f"  总耗时: {metrics.total_time:.2f}秒")
    print(f"  代码生成: {metrics.code_generation_time:.2f}秒")
    print(f"  执行时间: {metrics.execution_time:.2f}秒")
    
    print(f"\n效率指标:")
    print(f"  LLM调用: {metrics.llm_calls}次")
    print(f"  代码行数: {metrics.code_lines}行")
    print(f"  迭代次数: {metrics.iterations}次")
    
    print(f"\n质量指标:")
    print(f"  成功: {metrics.success}")
    print(f"  代码质量: {metrics.code_quality_score:.2f}/100")
    print(f"  代码复杂度: {metrics.code_complexity:.2f}")
    print(f"  模型复杂度: {metrics.model_complexity}")
    
    print(f"\n自主性和可解释性:")
    print(f"  自主性得分: {metrics.autonomy_score:.2f}/100")
    print(f"  可解释性得分: {metrics.explainability_score:.2f}/100")
    print(f"  注释比例: {metrics.comments_ratio:.2%}")
    print(f"  思考步骤: {metrics.thoughts_count}步")
    
    print(f"\n综合得分: {metrics.get_overall_score():.2f}/100")


def example_2_compare_agents():
    """示例2: 比较多个代理"""
    print("\n" + "=" * 60)
    print("示例2: 比较多个代理")
    print("=" * 60)
    
    # 创建三个不同代理的结果
    results = [
        create_sample_result(AgentType.REACT, time=45.5, llm_calls=3),
        create_sample_result(AgentType.RAG, time=35.0, llm_calls=5),
        create_sample_result(AgentType.MULTI_AGENT, time=55.0, llm_calls=8),
    ]
    
    # 计算指标
    calculator = MetricsCalculator()
    metrics_list = calculator.calculate_batch(results)
    
    # 比较代理
    comparator = AgentComparator()
    report = comparator.compare(metrics_list, "store-sales-forecasting")
    
    # 显示报告
    print(f"\n竞赛: {report.competition_name}")
    print(f"参与代理: {', '.join(report.agents)}")
    
    print(f"\n综合排名:")
    for i, (agent, score) in enumerate(report.overall_ranking, 1):
        print(f"  {i}. {agent}: {score:.2f}分")
    
    print(f"\n各项指标最佳代理:")
    for metric, agent in report.best_performer.items():
        if metric in ["total_time", "code_generation_time", "execution_time"]:
            value = report.metrics_comparison[metric][agent]
            print(f"  {metric}: {agent} ({value:.2f}秒)")
        elif metric in ["code_quality_score", "autonomy_score", "explainability_score"]:
            value = report.metrics_comparison[metric][agent]
            print(f"  {metric}: {agent} ({value:.2f}分)")
    
    print(f"\n结论:")
    for conclusion in report.conclusions:
        print(f"  - {conclusion}")
    
    print(f"\n建议:")
    for recommendation in report.recommendations:
        print(f"  - {recommendation}")


def example_3_markdown_report():
    """示例3: 生成Markdown报告"""
    print("\n" + "=" * 60)
    print("示例3: 生成Markdown报告")
    print("=" * 60)
    
    # 创建结果
    results = [
        create_sample_result(AgentType.REACT, time=45.5, llm_calls=3),
        create_sample_result(AgentType.RAG, time=35.0, llm_calls=5),
        create_sample_result(AgentType.MULTI_AGENT, time=55.0, llm_calls=8),
    ]
    
    # 计算和比较
    calculator = MetricsCalculator()
    metrics_list = calculator.calculate_batch(results)
    
    comparator = AgentComparator()
    report = comparator.compare(metrics_list)
    
    # 生成Markdown
    markdown = report.to_markdown()
    
    print("\nMarkdown报告:")
    print("-" * 60)
    print(markdown)
    print("-" * 60)
    
    # 保存到文件
    output_path = Path("comparison_report.md")
    output_path.write_text(markdown, encoding='utf-8')
    print(f"\n✓ Markdown报告已保存: {output_path}")


def example_4_detailed_metrics():
    """示例4: 详细指标分析"""
    print("\n" + "=" * 60)
    print("示例4: 详细指标分析")
    print("=" * 60)
    
    result = create_sample_result(AgentType.REACT, time=45.5, llm_calls=3)
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate(result)
    
    # 显示所有指标
    print("\n完整指标列表:")
    metrics_dict = metrics.to_dict()
    
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        elif isinstance(value, bool):
            print(f"  {key}: {'是' if value else '否'}")
        elif isinstance(value, (list, dict)) and value:
            print(f"  {key}: {value}")


def main():
    """主函数"""
    print("=" * 60)
    print("评估模块示例")
    print("=" * 60)
    
    try:
        example_1_single_metrics()
        example_2_compare_agents()
        example_3_markdown_report()
        example_4_detailed_metrics()
        
        print("\n" + "=" * 60)
        print("所有示例完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

