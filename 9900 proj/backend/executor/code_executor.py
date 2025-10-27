"""
代码执行引擎

安全地执行生成的Python代码
"""
import sys
import io
import subprocess
import time
import signal
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import tempfile
import os

from backend.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """
    代码执行结果
    """
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    return_code: int = 0
    submission_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "return_code": self.return_code,
            "submission_path": self.submission_path
        }


class TimeoutException(Exception):
    """超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutException("代码执行超时")


class CodeExecutor:
    """
    代码执行引擎
    
    支持三种执行模式：
    1. subprocess模式（推荐，隔离性好）
    2. exec模式（快速，但隔离性差）
    3. docker模式（最安全，需要Docker）
    """
    
    def __init__(
        self,
        mode: str = "subprocess",
        timeout: int = 300,
        max_memory_mb: int = 2048,
        enable_network: bool = True
    ):
        """
        初始化执行器
        
        Args:
            mode: 执行模式 (subprocess, exec, docker)
            timeout: 超时时间（秒）
            max_memory_mb: 最大内存（MB）
            enable_network: 是否允许网络访问
        """
        self.mode = mode
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.enable_network = enable_network
        
        logger.info(f"初始化CodeExecutor: mode={mode}, timeout={timeout}s")
    
    def execute(
        self,
        code: str,
        working_dir: Optional[Path] = None,
        env_vars: Optional[Dict[str, str]] = None,
        output_dir: Optional[Path] = None
    ) -> ExecutionResult:
        """
        执行Python代码
        
        Args:
            code: 要执行的代码
            working_dir: 工作目录
            env_vars: 环境变量
            output_dir: 输出目录（用于查找submission.csv）
            
        Returns:
            执行结果
        """
        logger.info(f"开始执行代码 (mode={self.mode})")
        
        if self.mode == "subprocess":
            return self._execute_subprocess(code, working_dir, env_vars, output_dir)
        elif self.mode == "exec":
            return self._execute_exec(code, working_dir)
        elif self.mode == "docker":
            return self._execute_docker(code, working_dir)
        else:
            raise ValueError(f"不支持的执行模式: {self.mode}")
    
    def execute_file(
        self,
        code_file: Path,
        working_dir: Optional[Path] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        执行Python文件
        
        Args:
            code_file: 代码文件路径
            working_dir: 工作目录
            env_vars: 环境变量
            
        Returns:
            执行结果
        """
        if not code_file.exists():
            return ExecutionResult(
                success=False,
                output="",
                error=f"代码文件不存在: {code_file}",
                return_code=-1
            )
        
        code = code_file.read_text(encoding='utf-8')
        return self.execute(code, working_dir or code_file.parent, env_vars)
    
    def _execute_subprocess(
        self,
        code: str,
        working_dir: Optional[Path],
        env_vars: Optional[Dict[str, str]],
        output_dir: Optional[Path] = None
    ) -> ExecutionResult:
        """
        使用subprocess执行代码（推荐模式）
        
        优点：
        - 进程隔离
        - 可以设置超时
        - 可以捕获输出
        - 相对安全
        """
        start_time = time.time()
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                temp_file = Path(f.name)
                f.write(code)
            
            # 设置工作目录
            if working_dir is None:
                working_dir = temp_file.parent
            
            # 准备环境变量
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)
            
            # 执行代码
            logger.info(f"执行临时文件: {temp_file}")
            logger.info(f"工作目录: {working_dir}")
            
            process = subprocess.Popen(
                [sys.executable, str(temp_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(working_dir),
                env=env,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                return_code = process.returncode
                
                execution_time = time.time() - start_time
                
                # 检查是否生成了submission.csv
                submission_path = None
                possible_paths = [
                    working_dir / "submission.csv",
                    Path.cwd() / "submission.csv"
                ]
                
                # 如果指定了输出目录，优先检查输出目录
                if output_dir:
                    possible_paths.insert(0, output_dir / "submission.csv")
                
                for path in possible_paths:
                    if path.exists():
                        submission_path = str(path)
                        logger.info(f"✓ 找到submission.csv: {submission_path}")
                        break
                
                success = (return_code == 0) and (submission_path is not None)
                
                result = ExecutionResult(
                    success=success,
                    output=stdout,
                    error=stderr if stderr else None,
                    execution_time=execution_time,
                    return_code=return_code,
                    submission_path=submission_path
                )
                
                if success:
                    logger.info(f"✓ 代码执行成功 (耗时: {execution_time:.2f}s)")
                else:
                    if submission_path is None:
                        logger.warning("代码执行完成，但未找到submission.csv")
                    if return_code != 0:
                        logger.warning(f"代码返回非零状态码: {return_code}")
                
                return result
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                logger.error(f"代码执行超时 (>{self.timeout}s)")
                
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"执行超时 (>{self.timeout}秒)",
                    execution_time=time.time() - start_time,
                    return_code=-1
                )
            
            finally:
                # 清理临时文件
                try:
                    temp_file.unlink()
                except:
                    pass
        
        except Exception as e:
            logger.error(f"代码执行异常: {e}")
            logger.exception("详细错误:")
            
            return ExecutionResult(
                success=False,
                output="",
                error=f"执行异常: {str(e)}",
                execution_time=time.time() - start_time,
                return_code=-1
            )
    
    def _execute_exec(
        self,
        code: str,
        working_dir: Optional[Path]
    ) -> ExecutionResult:
        """
        使用exec执行代码（快速但不安全）
        
        警告：此模式隔离性差，仅用于测试
        """
        start_time = time.time()
        
        # 切换工作目录
        original_dir = Path.cwd()
        if working_dir:
            os.chdir(working_dir)
        
        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # 设置超时（Unix only）
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            
            # 执行代码
            exec_globals = {
                '__name__': '__main__',
                '__file__': '<string>',
            }
            exec(code, exec_globals)
            
            # 取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            execution_time = time.time() - start_time
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # 查找submission.csv
            submission_path = None
            if working_dir:
                submission_file = working_dir / "submission.csv"
                if submission_file.exists():
                    submission_path = str(submission_file)
            
            return ExecutionResult(
                success=submission_path is not None,
                output=stdout,
                error=stderr if stderr else None,
                execution_time=execution_time,
                return_code=0,
                submission_path=submission_path
            )
        
        except TimeoutException:
            logger.error("代码执行超时")
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"执行超时 (>{self.timeout}秒)",
                execution_time=time.time() - start_time,
                return_code=-1
            )
        
        except Exception as e:
            logger.error(f"代码执行失败: {e}")
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
                return_code=-1
            )
        
        finally:
            # 恢复
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            os.chdir(original_dir)
    
    def _execute_docker(
        self,
        code: str,
        working_dir: Optional[Path]
    ) -> ExecutionResult:
        """
        使用Docker执行代码（最安全但需要Docker）
        
        TODO: 实现Docker执行模式
        """
        logger.warning("Docker执行模式尚未实现，回退到subprocess模式")
        return self._execute_subprocess(code, working_dir, None)
    
    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        验证代码语法
        
        Args:
            code: 要验证的代码
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            error_msg = f"语法错误 (行 {e.lineno}): {e.msg}"
            logger.warning(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"编译错误: {str(e)}"
            logger.warning(error_msg)
            return False, error_msg


class SafeCodeExecutor(CodeExecutor):
    """
    安全代码执行器
    
    添加额外的安全检查
    """
    
    # 危险的导入和函数
    DANGEROUS_IMPORTS = [
        'os.system',
        'subprocess.call',
        'subprocess.Popen',
        'eval',
        'exec',
        '__import__',
        'open(',  # 可能读写敏感文件
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("使用SafeCodeExecutor（增强安全检查）")
    
    def execute(
        self,
        code: str,
        working_dir: Optional[Path] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """执行代码前进行安全检查"""
        
        # 1. 语法验证
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"代码验证失败: {error}",
                return_code=-1
            )
        
        # 2. 危险代码检测（简单版）
        for dangerous in self.DANGEROUS_IMPORTS:
            if dangerous in code:
                logger.warning(f"检测到潜在危险代码: {dangerous}")
                # 注意：这只是警告，不阻止执行
                # 实际生产环境应该更严格
        
        # 3. 执行代码
        return super().execute(code, working_dir, env_vars)

