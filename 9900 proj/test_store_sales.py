"""
测试项目要求的时间序列预测任务

Store Sales - Time Series Forecasting
https://www.kaggle.com/competitions/store-sales-time-series-forecasting
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.kaggle import KaggleDataFetcher
from backend.agents import AgentConfig, AgentType
from backend.agents.react_agent import ReactAgent


async def main():
    """主函数"""
    print("=" * 70)
    print("🏪 Store Sales Time Series Forecasting 测试")
    print("=" * 70)
    
    # 步骤1: 获取数据
    print("\n📥 步骤1: 获取竞赛数据")
    print("-" * 70)
    
    fetcher = KaggleDataFetcher()
    
    # 项目要求的时间序列预测竞赛
    competition_url = "https://www.kaggle.com/competitions/store-sales-time-series-forecasting"
    print(f"竞赛: {competition_url}")
    print("⚠️  注意：此数据集较大，下载可能需要一些时间...")
    
    try:
        info = fetcher.fetch_complete_info(
            competition_url,
            download_data=True,
            force_download=False
        )
        
        print(f"\n✓ 数据已下载: {info.data_path}")
        print(f"  问题类型: {info.problem_type}")
        print(f"  评估指标: {info.evaluation_metric}")
        print(f"  训练文件: {info.train_files}")
        print(f"  测试文件: {info.test_files}")
        if info.train_shape:
            print(f"  训练集形状: {info.train_shape}")
        if info.test_shape:
            print(f"  测试集形状: {info.test_shape}")
        
    except Exception as e:
        print(f"\n✗ 数据获取失败: {e}")
        print("\n可能的原因：")
        print("  1. 需要先在Kaggle网站上接受竞赛规则")
        print("  2. 网络问题")
        print("  3. Kaggle API配置问题")
        return None
    
    # 步骤2: 配置Agent
    print("\n⚙️  步骤2: 配置ReactAgent")
    print("-" * 70)
    
    config = AgentConfig(
        agent_type=AgentType.REACT,
        competition_name=info.competition_name,
        competition_url=info.competition_url,
        data_path=info.data_path,
        llm_model="gpt-4o-mini",
        temperature=0.3,  # 降低温度提高确定性
        max_tokens=4000,  # 时间序列可能需要更多代码
        max_execution_time=600  # 增加超时到10分钟
    )
    
    print(f"✓ 配置完成")
    print(f"  竞赛: {config.competition_name}")
    print(f"  数据路径: {config.data_path}")
    print(f"  输出目录: {config.output_dir}")
    
    # 步骤3: 运行Agent
    print("\n🚀 步骤3: 运行ReactAgent")
    print("-" * 70)
    print("开始生成时间序列预测代码...")
    print("这可能需要较长时间（数据量大 + 复杂模型）")
    print()
    
    agent = ReactAgent(config)
    
    # 准备输入
    problem_description = fetcher.get_data_summary(info)
    
    data_info = {
        "train_files": info.train_files,
        "test_files": info.test_files,
        "columns": info.columns,
        "train_shape": info.train_shape,
        "test_shape": info.test_shape,
        "problem_type": info.problem_type,
        "evaluation_metric": info.evaluation_metric,
        "all_files_info": info.extra_info.get('all_files', {})  # ✅ 传递所有文件信息
    }
    
    # 运行
    result = await agent.run(problem_description, data_info)
    
    # 步骤4: 查看结果
    print("\n" + "=" * 70)
    print("📊 执行结果")
    print("=" * 70)
    
    print(f"\n状态: {result.status.value}")
    print(f"总耗时: {result.total_time:.2f}秒")
    print(f"  - 代码生成: {result.code_generation_time:.2f}秒")
    print(f"  - 代码执行: {result.execution_time:.2f}秒")
    
    print(f"\n指标:")
    print(f"  - LLM调用: {result.llm_calls}次")
    print(f"  - 代码行数: {result.code_lines}行")
    print(f"  - 思考步骤: {len(result.thoughts)}")
    
    print(f"\n文件:")
    print(f"  - 代码: {result.code_file_path}")
    print(f"  - 提交: {result.submission_file_path}")
    
    if result.submission_file_path:
        print(f"\n✅ 成功生成时间序列预测submission.csv!")
        
        # 查看submission
        import pandas as pd
        try:
            sub_df = pd.read_csv(result.submission_file_path)
            print(f"\nSubmission预览:")
            print(sub_df.head(10))
            print(f"\n形状: {sub_df.shape}")
            print(f"列名: {list(sub_df.columns)}")
        except Exception as e:
            print(f"读取submission失败: {e}")
    else:
        print(f"\n❌ 未能生成submission.csv")
        if result.execution_error:
            print(f"\n错误信息:")
            print(result.execution_error[:500])
    
    # 显示部分代码
    if result.generated_code:
        print(f"\n" + "=" * 70)
        print("💻 生成的代码（前40行）")
        print("=" * 70)
        lines = result.generated_code.split('\n')[:40]
        for i, line in enumerate(lines, 1):
            print(f"{i:3d} | {line}")
        if len(result.generated_code.split('\n')) > 40:
            print("...")
            print(f"（共 {len(result.generated_code.split('\n'))} 行）")
    
    print("\n" + "=" * 70)
    print("🎉 Store Sales 测试完成!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())

