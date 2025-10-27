#!/usr/bin/env python3
"""
测试前端+后端集成功能
模拟Streamlit前端的操作流程
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from backend.agents import ReactAgent, AgentConfig, AgentType
from backend.kaggle import KaggleDataFetcher


async def test_frontend_backend_integration():
    """测试前端+后端集成功能"""
    
    print("=" * 70)
    print("🧪 测试前端+后端集成功能")
    print("=" * 70)
    
    try:
        # 步骤1: 模拟前端数据获取
        print("\n📥 步骤1: 模拟前端数据获取")
        print("-" * 50)
        
        competition_url = "https://www.kaggle.com/competitions/titanic"
        competition_name = "titanic"
        
        print(f"✅ 竞赛URL: {competition_url}")
        print(f"✅ 竞赛名称: {competition_name}")
        
        # 获取竞赛数据
        fetcher = KaggleDataFetcher()
        competition_info = fetcher.fetch_complete_info(competition_name)
        
        print(f"✅ 数据获取成功")
        print(f"   - 训练文件: {competition_info.train_files}")
        print(f"   - 测试文件: {competition_info.test_files}")
        print(f"   - 训练数据形状: {competition_info.train_shape}")
        print(f"   - 测试数据形状: {competition_info.test_shape}")
        
        # 步骤2: 模拟前端Agent配置
        print("\n⚙️ 步骤2: 模拟前端Agent配置")
        print("-" * 50)
        
        config = AgentConfig(
            agent_type=AgentType.REACT,
            competition_name=competition_info.competition_name,
            competition_url=competition_url,
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
        print(f"   - 执行超时: {config.max_execution_time}秒")
        
        # 步骤3: 创建Agent并运行
        print("\n🤖 步骤3: 创建并运行ReactAgent")
        print("-" * 50)
        
        agent = ReactAgent(config)
        print(f"✅ ReactAgent初始化成功")
        
        # 准备数据信息（模拟前端传递的数据）
        data_info = {
            'train_files': competition_info.train_files,
            'test_files': competition_info.test_files,
            'columns': competition_info.columns,
            'all_files_info': competition_info.extra_info.get('all_files', {}) if competition_info.extra_info else {}
        }
        
        problem_description = f"Kaggle Competition: {competition_info.competition_name}"
        
        print(f"📝 问题描述: {problem_description}")
        print(f"⏳ 开始执行Agent...")
        
        # 运行Agent（这是前端会调用的核心方法）
        result = await agent.run(
            problem_description=problem_description,
            data_info=data_info
        )
        
        # 步骤4: 分析结果
        print("\n📈 步骤4: 分析执行结果")
        print("-" * 50)
        
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
                    print(f"   📋 前3行:")
                    for i, line in enumerate(lines[:3], 1):
                        print(f"      {i}: {line.strip()}")
            else:
                print(f"   📁 文件存在: 否")
        
        if result.generated_code:
            print(f"✅ 生成代码长度: {len(result.generated_code)} 字符")
            print(f"✅ 代码预览:")
            print("   " + "=" * 60)
            code_lines = result.generated_code.split('\n')[:15]
            for i, line in enumerate(code_lines, 1):
                print(f"   {i:2d}| {line}")
            if len(result.generated_code.split('\n')) > 15:
                print(f"   ... (还有 {len(result.generated_code.split('\n')) - 15} 行)")
            print("   " + "=" * 60)
        
        if result.error_message:
            print(f"❌ 错误信息: {result.error_message}")
        
        # 步骤5: 验证前端显示需求
        print("\n🖥️ 步骤5: 验证前端显示数据")
        print("-" * 50)
        
        # 模拟前端需要的数据结构
        frontend_data = {
            "competition_info": {
                "name": competition_info.competition_name,
                "url": competition_url,
                "train_shape": competition_info.train_shape,
                "test_shape": competition_info.test_shape,
                "files": competition_info.train_files + competition_info.test_files
            },
            "agent_result": {
                "status": result.status.value,
                "total_time": result.total_time,
                "llm_calls": result.llm_calls,
                "code_lines": result.code_lines,
                "has_submission": bool(result.submission_file_path and Path(result.submission_file_path).exists()),
                "error_message": result.error_message
            }
        }
        
        print(f"✅ 前端数据结构验证:")
        print(f"   - 竞赛信息完整: {bool(frontend_data['competition_info']['name'])}")
        print(f"   - 执行状态有效: {bool(frontend_data['agent_result']['status'])}")
        print(f"   - 有提交文件: {frontend_data['agent_result']['has_submission']}")
        print(f"   - 执行时间合理: {frontend_data['agent_result']['total_time'] < 300}")
        
        # 步骤6: 总结测试结果
        print("\n🎯 步骤6: 测试总结")
        print("-" * 50)
        
        success_criteria = [
            result.status.value == "completed",
            result.submission_file_path is not None,
            Path(result.submission_file_path).exists() if result.submission_file_path else False,
            result.total_time < 300,  # 5分钟内完成
            result.error_message is None or "FutureWarning" in result.error_message  # 允许pandas警告
        ]
        
        passed = sum(success_criteria)
        total = len(success_criteria)
        
        print(f"✅ 测试通过: {passed}/{total}")
        print(f"   ✓ 状态完成: {success_criteria[0]}")
        print(f"   ✓ 有提交文件: {success_criteria[1]}")
        print(f"   ✓ 文件存在: {success_criteria[2]}")
        print(f"   ✓ 时间合理: {success_criteria[3]}")
        print(f"   ✓ 无严重错误: {success_criteria[4]}")
        
        if passed >= 4:  # 至少4/5通过
            print(f"\n🎉 前端+后端集成测试成功！")
            print(f"   可以在Streamlit界面中使用以下配置:")
            print(f"   - 竞赛URL: {competition_url}")
            print(f"   - Agent类型: ReactAgent")
            print(f"   - 模型: gpt-4o-mini")
            print(f"   - 温度: 0.3")
            return True
        else:
            print(f"\n❌ 前端+后端集成测试失败")
            return False
            
    except Exception as e:
        print(f"\n💥 测试过程中出现异常:")
        print(f"   类型: {type(e).__name__}")
        print(f"   消息: {str(e)}")
        import traceback
        print(f"   详细错误:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始测试前端+后端集成...")
    result = asyncio.run(test_frontend_backend_integration())
    
    if result:
        print("\n" + "=" * 70)
        print("✅ 前端+后端集成测试完成 - 可以正常使用！")
        print("🌐 请在浏览器中访问: http://localhost:8501")
        print("📝 输入竞赛URL: https://www.kaggle.com/competitions/titanic")
        print("🚀 点击'开始生成解决方案'按钮")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ 前端+后端集成测试失败")
        print("=" * 70)
        sys.exit(1)
