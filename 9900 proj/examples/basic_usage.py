"""
基础使用示例

展示如何使用BaseAgent和ExampleAgent
"""
import asyncio
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents import AgentConfig, AgentType
from backend.agents.example_agent import ExampleAgent


async def main():
    """主函数"""
    print("=" * 60)
    print("AI Agent 基础使用示例")
    print("=" * 60)
    
    # 1. 创建配置
    print("\n步骤1: 创建代理配置")
    config = AgentConfig(
        agent_type=AgentType.REACT,
        competition_name="store-sales-forecasting",
        competition_url="https://www.kaggle.com/competitions/store-sales-time-series-forecasting",
        data_path=Path("data/competitions/store-sales"),
        llm_model="llama3",
        temperature=0.7,
        verbose=True
    )
    print(f"✓ 配置创建完成: {config.competition_name}")
    
    # 2. 初始化代理
    print("\n步骤2: 初始化代理")
    agent = ExampleAgent(config)
    print(f"✓ 代理初始化完成: {agent}")
    
    # 3. 设置回调函数（用于实时更新UI）
    print("\n步骤3: 设置回调函数")
    
    def on_status_change(status):
        print(f"  [状态] {status.value}")
    
    def on_log(message):
        print(f"  [日志] {message}")
    
    agent.set_callbacks(
        status_callback=on_status_change,
        log_callback=on_log
    )
    print("✓ 回调函数设置完成")
    
    # 4. 准备输入数据
    print("\n步骤4: 准备输入数据")
    problem_description = """
    这是一个时间序列预测问题。需要预测商店在未来时间的销售额。
    数据包含：
    - 历史销售记录
    - 商店信息
    - 产品信息
    - 时间特征（日期、星期等）
    
    评估指标：均方根误差（RMSE）
    """
    
    data_info = {
        "train_shape": (3000000, 6),
        "test_shape": (28512, 5),
        "columns": ["id", "date", "store_nbr", "family", "sales", "onpromotion"],
        "target": "sales",
        "key_features": ["date", "store_nbr", "family", "onpromotion"]
    }
    print("✓ 输入数据准备完成")
    
    # 5. 运行代理
    print("\n步骤5: 运行代理")
    print("-" * 60)
    result = await agent.run(problem_description, data_info)
    print("-" * 60)
    
    # 6. 查看结果
    print("\n步骤6: 查看结果")
    print(f"\n执行结果:")
    print(f"  状态: {result.status.value}")
    print(f"  代码文件: {result.code_file_path}")
    print(f"  提交文件: {result.submission_file_path}")
    print(f"  总耗时: {result.total_time:.2f}秒")
    print(f"  代码生成耗时: {result.code_generation_time:.2f}秒")
    print(f"  执行耗时: {result.execution_time:.2f}秒")
    print(f"  LLM调用次数: {result.llm_calls}")
    print(f"  代码行数: {result.code_lines}")
    
    # 7. 获取评估指标
    print("\n步骤7: 获取评估指标")
    metrics = agent.get_metrics()
    print("指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # 8. 查看生成的代码（前20行）
    if result.code_file_path and result.code_file_path.exists():
        print("\n步骤8: 查看生成的代码（前20行）")
        print("-" * 60)
        code_lines = result.generated_code.split('\n')[:20]
        for i, line in enumerate(code_lines, 1):
            print(f"{i:3d} | {line}")
        if len(result.generated_code.split('\n')) > 20:
            print("...")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    # 运行示例
    result = asyncio.run(main())

