"""
BaseAgent测试
"""
import pytest
import asyncio
from pathlib import Path
from backend.agents.base_agent import (
    BaseAgent, AgentConfig, AgentResult, 
    AgentType, AgentStatus
)
from backend.agents.example_agent import ExampleAgent


@pytest.fixture
def sample_config():
    """创建示例配置"""
    return AgentConfig(
        agent_type=AgentType.REACT,
        competition_name="test-competition",
        competition_url="https://www.kaggle.com/c/test",
        data_path=Path("data/competitions/test")
    )


@pytest.fixture
def sample_agent(sample_config):
    """创建示例代理"""
    return ExampleAgent(sample_config)


class TestAgentConfig:
    """测试AgentConfig"""
    
    def test_config_creation(self, sample_config):
        """测试配置创建"""
        assert sample_config.agent_type == AgentType.REACT
        assert sample_config.competition_name == "test-competition"
        assert sample_config.temperature == 0.7
        assert sample_config.max_tokens == 4096
    
    def test_config_to_dict(self, sample_config):
        """测试配置转换为字典"""
        config_dict = sample_config.to_dict()
        assert config_dict["agent_type"] == "react"
        assert config_dict["competition_name"] == "test-competition"
        assert "llm_model" in config_dict
    
    def test_config_save(self, sample_config, tmp_path):
        """测试配置保存"""
        config_file = tmp_path / "config.json"
        sample_config.save(config_file)
        assert config_file.exists()


class TestAgentResult:
    """测试AgentResult"""
    
    def test_result_creation(self):
        """测试结果创建"""
        result = AgentResult(
            agent_type=AgentType.REACT,
            competition_name="test",
            status=AgentStatus.IDLE
        )
        assert result.agent_type == AgentType.REACT
        assert result.status == AgentStatus.IDLE
        assert result.start_time is not None
    
    def test_result_mark_completed(self):
        """测试标记完成"""
        result = AgentResult(
            agent_type=AgentType.REACT,
            competition_name="test",
            status=AgentStatus.EXECUTING
        )
        result.mark_completed()
        assert result.status == AgentStatus.COMPLETED
        assert result.end_time is not None
        assert result.total_time > 0
    
    def test_result_mark_failed(self):
        """测试标记失败"""
        result = AgentResult(
            agent_type=AgentType.REACT,
            competition_name="test",
            status=AgentStatus.EXECUTING
        )
        result.mark_failed("测试错误")
        assert result.status == AgentStatus.FAILED
        assert result.error_message == "测试错误"
    
    def test_result_to_dict(self):
        """测试结果转换为字典"""
        result = AgentResult(
            agent_type=AgentType.REACT,
            competition_name="test",
            status=AgentStatus.COMPLETED
        )
        result_dict = result.to_dict()
        assert result_dict["agent_type"] == "react"
        assert result_dict["status"] == "completed"


class TestBaseAgent:
    """测试BaseAgent"""
    
    def test_agent_initialization(self, sample_agent):
        """测试代理初始化"""
        assert sample_agent.status == AgentStatus.IDLE
        assert sample_agent.result.agent_type == AgentType.REACT
    
    def test_set_callbacks(self, sample_agent):
        """测试设置回调"""
        status_updates = []
        log_messages = []
        
        def on_status(status):
            status_updates.append(status)
        
        def on_log(message):
            log_messages.append(message)
        
        sample_agent.set_callbacks(
            status_callback=on_status,
            log_callback=on_log
        )
        
        sample_agent._update_status(AgentStatus.ANALYZING)
        sample_agent._log("测试消息")
        
        assert len(status_updates) == 1
        assert status_updates[0] == AgentStatus.ANALYZING
        assert len(log_messages) == 1
    
    @pytest.mark.asyncio
    async def test_analyze_problem(self, sample_agent):
        """测试问题分析"""
        result = await sample_agent.analyze_problem("测试问题描述")
        assert "problem_type" in result
        assert "key_insights" in result
    
    @pytest.mark.asyncio
    async def test_generate_code(self, sample_agent):
        """测试代码生成"""
        problem_analysis = {"problem_type": "regression"}
        data_info = {"columns": ["col1", "col2"]}
        
        code = await sample_agent.generate_code(problem_analysis, data_info)
        assert isinstance(code, str)
        assert len(code) > 0
    
    def test_save_code(self, sample_agent):
        """测试保存代码"""
        test_code = "print('Hello, World!')"
        code_file = sample_agent._save_code(test_code)
        
        assert code_file.exists()
        assert code_file.read_text() == test_code
    
    def test_get_metrics(self, sample_agent):
        """测试获取指标"""
        metrics = sample_agent.get_metrics()
        assert isinstance(metrics, dict)
        assert "total_time" in metrics


# 如果直接运行此文件，执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

