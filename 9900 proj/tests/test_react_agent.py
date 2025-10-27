"""
测试ReactAgent - 端到端工作流程

这个脚本展示了完整的工作流程：
1. 获取Kaggle数据
2. 使用ReactAgent（OpenAI）生成代码
3. 执行代码
4. 生成submission.csv
"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.kaggle import KaggleDataFetcher
from backend.agents import AgentConfig, AgentType
from backend.agents.react_agent import ReactAgent


async def main():
    """主函数"""
    print("=" * 70)
    print("🤖 ReactAgent 端到端测试")
    print("=" * 70)
    
    # 步骤1: 获取Kaggle数据
    print("\n📥 步骤1: 获取Kaggle数据")
    print("-" * 70)
    
    fetcher = KaggleDataFetcher()
    
    # 使用titanic数据集（小而快）
    competition_url = "https://www.kaggle.com/competitions/titanic"
    print(f"竞赛: {competition_url}")
    
    info = fetcher.fetch_complete_info(competition_url, download_data=True)
    
    print(f"✓ 数据已下载: {info.data_path}")
    print(f"  训练集: {info.train_shape}")
    print(f"  测试集: {info.test_shape}")
    print(f"  列数: {len(info.columns)}")
    
    # 步骤2: 创建ReactAgent配置
    print("\n⚙️  步骤2: 配置ReactAgent")
    print("-" * 70)
    
    config = AgentConfig(
        agent_type=AgentType.REACT,
        competition_name=info.competition_name,
        competition_url=info.competition_url,
        data_path=info.data_path,
        llm_model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=3000
    )
    
    print(f"✓ 配置完成")
    print(f"  Agent类型: {config.agent_type.value}")
    print(f"  LLM模型: {config.llm_model}")
    print(f"  输出目录: {config.output_dir}")
    
    # 步骤3: 初始化Agent并运行
    print("\n🚀 步骤3: 运行ReactAgent")
    print("-" * 70)
    
    agent = ReactAgent(config)
    
    # 准备数据信息
    problem_description = fetcher.get_data_summary(info)
    
    data_info = {
        "train_files": info.train_files,
        "test_files": info.test_files,
        "columns": info.columns,
        "train_shape": info.train_shape,
        "test_shape": info.test_shape
    }
    
    print("开始执行...")
    print()
    
    # 运行Agent
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
    print(f"  - LLM调用次数: {result.llm_calls}")
    print(f"  - 代码行数: {result.code_lines}")
    print(f"  - 思考步骤: {len(result.thoughts)}")
    print(f"  - 执行动作: {len(result.actions)}")
    
    print(f"\n文件:")
    print(f"  - 代码文件: {result.code_file_path}")
    print(f"  - 提交文件: {result.submission_file_path}")
    
    if result.submission_file_path:
        print(f"\n✅ 成功生成submission.csv!")
        
        # 查看submission文件
        import pandas as pd
        sub_df = pd.read_csv(result.submission_file_path)
        print(f"\nSubmission预览:")
        print(sub_df.head(10))
        print(f"\n形状: {sub_df.shape}")
    else:
        print(f"\n❌ 未能生成submission.csv")
        if result.execution_error:
            print(f"\n错误: {result.execution_error}")
    
    # 显示生成的代码（前30行）
    if result.generated_code:
        print(f"\n" + "=" * 70)
        print("💻 生成的代码（前30行）")
        print("=" * 70)
        code_lines = result.generated_code.split('\n')[:30]
        for i, line in enumerate(code_lines, 1):
            print(f"{i:3d} | {line}")
        if len(result.generated_code.split('\n')) > 30:
            print("...")
    
    print("\n" + "=" * 70)
    print("🎉 测试完成!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())

