#!/usr/bin/env python3
"""
测试Titanic后端功能
绕过前端，直接测试Agent核心功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from backend.agents import ReactAgent, AgentConfig, AgentType
from backend.kaggle import KaggleDataFetcher


async def test_titanic_backend():
    """测试Titanic竞赛的完整后端流程"""
    
    print("=" * 60)
    print("🧪 测试Titanic后端功能")
    print("=" * 60)
    
    try:
        # 步骤1: 获取竞赛数据
        print("\n📥 步骤1: 获取Titanic竞赛数据")
        print("-" * 40)
        
        fetcher = KaggleDataFetcher()
        competition_info = fetcher.fetch_complete_info("titanic")
        
        print(f"✅ 竞赛名称: {competition_info.competition_name}")
        print(f"✅ 数据路径: {competition_info.data_path}")
        print(f"✅ 训练文件: {competition_info.train_files}")
        print(f"✅ 测试文件: {competition_info.test_files}")
        print(f"✅ 训练数据形状: {competition_info.train_shape}")
        print(f"✅ 测试数据形状: {competition_info.test_shape}")
        
        # 步骤2: 创建Agent配置
        print("\n⚙️ 步骤2: 创建Agent配置")
        print("-" * 40)
        
        config = AgentConfig(
            agent_type=AgentType.REACT,
            competition_name=competition_info.competition_name,
            competition_url=f"https://www.kaggle.com/competitions/{competition_info.competition_name}",
            data_path=Path(competition_info.data_path),
            llm_model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=4000,
            max_retries=3,
            max_execution_time=600
        )
        
        print(f"✅ Agent配置创建成功")
        print(f"   - 模型: {config.llm_model}")
        print(f"   - 温度: {config.temperature}")
        print(f"   - 最大重试: {config.max_retries}")
        
        # 步骤3: 创建ReactAgent
        print("\n🤖 步骤3: 创建ReactAgent")
        print("-" * 40)
        
        agent = ReactAgent(config)
        print(f"✅ ReactAgent初始化成功")
        
        # 步骤4: 准备数据信息
        print("\n📊 步骤4: 准备数据信息")
        print("-" * 40)
        
        data_info = {
            'train_files': competition_info.train_files,
            'test_files': competition_info.test_files,
            'columns': competition_info.columns,
            'all_files_info': competition_info.extra_info.get('all_files', {}) if competition_info.extra_info else {}
        }
        
        print(f"✅ 数据信息准备完成")
        print(f"   - 训练文件数: {len(data_info['train_files'])}")
        print(f"   - 测试文件数: {len(data_info['test_files'])}")
        print(f"   - 所有文件数: {len(data_info['all_files_info'])}")
        
        # 步骤5: 运行Agent
        print("\n🚀 步骤5: 运行ReactAgent")
        print("-" * 40)
        
        problem_description = f"Kaggle Competition: {competition_info.competition_name}"
        
        print(f"📝 问题描述: {problem_description}")
        print(f"⏳ 开始执行...")
        
        result = await agent.run(
            problem_description=problem_description,
            data_info=data_info
        )
        
        # 步骤6: 分析结果
        print("\n📈 步骤6: 分析执行结果")
        print("-" * 40)
        
        print(f"✅ 执行状态: {result.status}")
        print(f"✅ 总耗时: {result.total_time:.2f}秒")
        print(f"✅ LLM调用次数: {result.llm_calls}")
        print(f"✅ 代码行数: {result.code_lines}")
        print(f"✅ 思考步骤: {len(result.thoughts)}")
        print(f"✅ 执行动作: {len(result.actions)}")
        
        if result.submission_file_path:
            print(f"✅ Submission文件: {result.submission_file_path}")
            if Path(result.submission_file_path).exists():
                print(f"   📁 文件存在: 是")
                with open(result.submission_file_path, 'r') as f:
                    lines = f.readlines()
                    print(f"   📊 行数: {len(lines)}")
                    print(f"   📋 前3行: {lines[:3]}")
            else:
                print(f"   📁 文件存在: 否")
        
        if result.generated_code:
            print(f"✅ 生成代码长度: {len(result.generated_code)} 字符")
            print(f"✅ 代码预览:")
            print("   " + "=" * 50)
            code_lines = result.generated_code.split('\n')[:10]
            for i, line in enumerate(code_lines, 1):
                print(f"   {i:2d}| {line}")
            if len(result.generated_code.split('\n')) > 10:
                print(f"   ... (还有 {len(result.generated_code.split('\n')) - 10} 行)")
            print("   " + "=" * 50)
        
        if result.error_message:
            print(f"❌ 错误信息: {result.error_message}")
        
        # 步骤7: 总结
        print("\n🎯 步骤7: 测试总结")
        print("-" * 40)
        
        if result.status.value == "completed":
            print("🎉 测试成功！后端功能正常")
            if result.submission_path and Path(result.submission_path).exists():
                print("✅ 成功生成submission.csv")
            else:
                print("⚠️ 未生成submission.csv，但代码生成成功")
        else:
            print("❌ 测试失败，需要检查后端逻辑")
            
        return result
        
    except Exception as e:
        print(f"\n💥 测试过程中出现异常:")
        print(f"   类型: {type(e).__name__}")
        print(f"   消息: {str(e)}")
        import traceback
        print(f"   详细错误:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("开始测试Titanic后端...")
    result = asyncio.run(test_titanic_backend())
    
    if result:
        print("\n" + "=" * 60)
        print("✅ 后端测试完成")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 后端测试失败")
        print("=" * 60)
        sys.exit(1)
