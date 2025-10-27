"""
基础AI代理抽象类

定义了所有AI代理的通用接口和行为模式
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json
import time

from backend.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AgentType(Enum):
    """代理类型枚举"""
    REACT = "react"
    RAG = "rag"
    MULTI_AGENT = "multi_agent"


class AgentStatus(Enum):
    """代理状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    GENERATING_CODE = "generating_code"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentConfig:
    """
    代理配置类
    
    包含运行代理所需的所有配置参数
    """
    # 基础配置
    agent_type: AgentType
    competition_name: str
    competition_url: str
    data_path: Path
    
    # LLM配置
    llm_model: str = "llama3"
    llm_base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # 执行配置
    max_execution_time: int = 300  # 秒
    max_memory_mb: int = 2048
    max_retries: int = 3
    
    # 输出配置
    output_dir: Optional[Path] = None
    save_intermediate_results: bool = True
    verbose: bool = True
    
    # 扩展配置（供子类使用）
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.output_dir is None:
            # 默认输出目录：data/generated_code/{competition_name}_{timestamp}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = config.generated_code_dir / f"{self.competition_name}_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_type": self.agent_type.value,
            "competition_name": self.competition_name,
            "competition_url": self.competition_url,
            "data_path": str(self.data_path),
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_execution_time": self.max_execution_time,
            "max_memory_mb": self.max_memory_mb,
            "max_retries": self.max_retries,
            "output_dir": str(self.output_dir),
            "save_intermediate_results": self.save_intermediate_results,
            "verbose": self.verbose,
            "extra_config": self.extra_config
        }
    
    def save(self, path: Path):
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class AgentResult:
    """
    代理执行结果类
    
    包含代理运行的所有输出和元数据
    """
    # 基本信息
    agent_type: AgentType
    competition_name: str
    status: AgentStatus
    
    # 生成的代码
    generated_code: str = ""
    code_file_path: Optional[Path] = None
    
    # 执行结果
    submission_file_path: Optional[Path] = None
    execution_output: str = ""
    execution_error: Optional[str] = None
    
    # 性能指标
    total_time: float = 0.0  # 总耗时（秒）
    code_generation_time: float = 0.0  # 代码生成耗时
    execution_time: float = 0.0  # 执行耗时
    llm_calls: int = 0  # LLM调用次数
    code_lines: int = 0  # 生成的代码行数
    
    # 中间结果
    thoughts: List[str] = field(default_factory=list)  # 思考过程
    actions: List[Dict[str, Any]] = field(default_factory=list)  # 执行的动作
    observations: List[str] = field(default_factory=list)  # 观察结果
    
    # 评估指标
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def mark_completed(self):
        """标记为完成"""
        self.status = AgentStatus.COMPLETED
        self.end_time = datetime.now()
        if self.start_time:
            self.total_time = (self.end_time - self.start_time).total_seconds()
    
    def mark_failed(self, error_message: str):
        """标记为失败"""
        self.status = AgentStatus.FAILED
        self.error_message = error_message
        self.end_time = datetime.now()
        if self.start_time:
            self.total_time = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_type": self.agent_type.value,
            "competition_name": self.competition_name,
            "status": self.status.value,
            "generated_code": self.generated_code,
            "code_file_path": str(self.code_file_path) if self.code_file_path else None,
            "submission_file_path": str(self.submission_file_path) if self.submission_file_path else None,
            "execution_output": self.execution_output,
            "execution_error": self.execution_error,
            "total_time": self.total_time,
            "code_generation_time": self.code_generation_time,
            "execution_time": self.execution_time,
            "llm_calls": self.llm_calls,
            "code_lines": self.code_lines,
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations,
            "metrics": self.metrics,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message
        }
    
    def save(self, path: Path):
        """保存结果到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class BaseAgent(ABC):
    """
    基础AI代理抽象类
    
    定义了所有AI代理必须实现的接口和通用行为
    """
    
    def __init__(self, config: AgentConfig):
        """
        初始化代理
        
        Args:
            config: 代理配置
        """
        self.config = config
        self.status = AgentStatus.IDLE
        self.result = AgentResult(
            agent_type=config.agent_type,
            competition_name=config.competition_name,
            status=self.status
        )
        
        # 回调函数（用于实时更新UI）
        self.status_callback: Optional[Callable[[AgentStatus], None]] = None
        self.log_callback: Optional[Callable[[str], None]] = None
        
        logger.info(f"初始化 {config.agent_type.value} 代理: {config.competition_name}")
    
    def set_callbacks(
        self,
        status_callback: Optional[Callable[[AgentStatus], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """
        设置回调函数
        
        Args:
            status_callback: 状态更新回调
            log_callback: 日志回调
        """
        self.status_callback = status_callback
        self.log_callback = log_callback
    
    def _update_status(self, status: AgentStatus):
        """更新状态"""
        self.status = status
        self.result.status = status
        
        if self.status_callback:
            self.status_callback(status)
        
        logger.info(f"状态更新: {status.value}")
    
    def _log(self, message: str, level: str = "info"):
        """记录日志"""
        log_func = getattr(logger, level, logger.info)
        log_func(message)
        
        if self.log_callback:
            self.log_callback(message)
    
    def _save_code(self, code: str) -> Path:
        """
        保存生成的代码
        
        Args:
            code: 生成的代码
            
        Returns:
            代码文件路径
        """
        code_file = self.config.output_dir / "generated_solution.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        self._log(f"代码已保存: {code_file}")
        return code_file
    
    def _save_result(self):
        """保存执行结果"""
        result_file = self.config.output_dir / "result.json"
        self.result.save(result_file)
        
        # 也保存配置
        config_file = self.config.output_dir / "config.json"
        self.config.save(config_file)
        
        self._log(f"结果已保存: {result_file}")
    
    @abstractmethod
    async def analyze_problem(self, problem_description: str) -> Dict[str, Any]:
        """
        分析问题描述
        
        Args:
            problem_description: Kaggle竞赛问题描述
            
        Returns:
            分析结果字典，包含：
            - problem_type: 问题类型（分类、回归、时序预测等）
            - key_insights: 关键洞察
            - suggested_approach: 建议方法
            - data_requirements: 数据要求
        """
        pass
    
    @abstractmethod
    async def generate_code(
        self,
        problem_analysis: Dict[str, Any],
        data_info: Dict[str, Any]
    ) -> str:
        """
        生成数据分析代码
        
        Args:
            problem_analysis: 问题分析结果
            data_info: 数据信息（列名、类型、统计信息等）
            
        Returns:
            生成的Python代码
        """
        pass
    
    @abstractmethod
    async def execute_code(self, code: str) -> Dict[str, Any]:
        """
        执行生成的代码
        
        Args:
            code: 要执行的代码
            
        Returns:
            执行结果字典，包含：
            - success: 是否成功
            - output: 标准输出
            - error: 错误信息（如果有）
            - submission_path: submission.csv路径
        """
        pass
    
    async def run(
        self,
        problem_description: str,
        data_info: Dict[str, Any]
    ) -> AgentResult:
        """
        运行完整的数据分析流程
        
        这是主入口方法，协调所有步骤
        
        Args:
            problem_description: 问题描述
            data_info: 数据信息
            
        Returns:
            代理执行结果
        """
        try:
            self._update_status(AgentStatus.INITIALIZING)
            self._log("开始运行AI代理...")
            
            # 1. 分析问题
            self._update_status(AgentStatus.ANALYZING)
            self._log("正在分析问题...")
            start_time = time.time()
            
            problem_analysis = await self.analyze_problem(problem_description)
            self.result.thoughts.append(f"问题分析: {problem_analysis}")
            
            # 2. 生成代码
            self._update_status(AgentStatus.GENERATING_CODE)
            self._log("正在生成代码...")
            code_start = time.time()
            
            generated_code = await self.generate_code(problem_analysis, data_info)
            self.result.generated_code = generated_code
            self.result.code_lines = len(generated_code.split('\n'))
            self.result.code_generation_time = time.time() - code_start
            
            # 保存代码
            code_file = self._save_code(generated_code)
            self.result.code_file_path = code_file
            
            # 3. 执行代码（带重试机制）
            self._update_status(AgentStatus.EXECUTING)
            self._log("正在执行代码...")
            exec_start = time.time()
            
            execution_result = await self.execute_code(generated_code)
            self.result.execution_time = time.time() - exec_start
            
            # 如果执行失败，尝试修复（如果Agent支持fix_code方法）
            retry_count = 0
            max_retries = self.config.max_retries
            
            while not execution_result.get("success") and retry_count < max_retries:
                error_msg = execution_result.get("error", "未知错误")
                self._log(f"代码执行失败，尝试修复（第{retry_count + 1}/{max_retries}次）...", level="warning")
                
                # 检查是否有fix_code方法
                if hasattr(self, 'fix_code'):
                    try:
                        # 调用LLM修复代码（传递数据信息）
                        retry_start = time.time()
                        fixed_code = await self.fix_code(generated_code, error_msg, data_info)
                        
                        # 保存修复后的代码
                        fixed_code_file = self.config.output_dir / f"generated_solution_fixed_v{retry_count + 1}.py"
                        with open(fixed_code_file, 'w', encoding='utf-8') as f:
                            f.write(fixed_code)
                        self._log(f"修复后的代码已保存: {fixed_code_file}")
                        
                        # 重新执行
                        execution_result = await self.execute_code(fixed_code)
                        self.result.execution_time += time.time() - retry_start
                        
                        if execution_result.get("success"):
                            # 修复成功，更新生成的代码
                            generated_code = fixed_code
                            self.result.generated_code = fixed_code
                            self.result.code_file_path = fixed_code_file
                            self._log(f"✓ 代码修复成功！")
                            break
                        else:
                            retry_count += 1
                            generated_code = fixed_code  # 使用修复后的代码继续下一轮
                    except Exception as e:
                        self._log(f"代码修复异常: {e}", level="error")
                        retry_count += 1
                else:
                    # 没有fix_code方法，退出重试
                    break
            
            if execution_result.get("success"):
                self.result.submission_file_path = Path(execution_result["submission_path"])
                self.result.execution_output = execution_result.get("output", "")
                self._log("代码执行成功！")
            else:
                self.result.execution_error = execution_result.get("error", "未知错误")
                self._log(f"代码执行失败: {self.result.execution_error}", level="error")
            
            # 4. 标记完成
            self.result.mark_completed()
            self._update_status(AgentStatus.COMPLETED)
            self._log(f"代理运行完成！总耗时: {self.result.total_time:.2f}秒")
            
        except Exception as e:
            error_msg = f"代理运行失败: {str(e)}"
            self._log(error_msg, level="error")
            self.result.mark_failed(error_msg)
            self._update_status(AgentStatus.FAILED)
            logger.exception("代理执行异常")
        
        finally:
            # 保存结果
            self._save_result()
        
        return self.result
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取代理特定的评估指标
        
        Returns:
            指标字典
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self.config.agent_type.value}, status={self.status.value})>"

