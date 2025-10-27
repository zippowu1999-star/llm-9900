# 快速开始指南

本指南将帮助你快速上手AI Agent数据分析系统。

## 📦 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## ⚙️ 配置环境

1. **配置Kaggle API**

从 [Kaggle账户设置](https://www.kaggle.com/settings/account) 下载 `kaggle.json`

```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 或者使用环境变量
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

2. **安装和启动Ollama**

```bash
# 下载并安装: https://ollama.ai/

# 拉取模型
ollama pull llama3

# 启动服务（默认在11434端口）
ollama serve
```

## 🚀 基础使用

### 方式1: 使用示例代码

```bash
python examples/basic_usage.py
```

### 方式2: 编写自己的代码

```python
import asyncio
from pathlib import Path
from backend.agents import AgentConfig, AgentType
from backend.agents.example_agent import ExampleAgent

async def main():
    # 1. 创建配置
    config = AgentConfig(
        agent_type=AgentType.REACT,
        competition_name="my-competition",
        competition_url="https://www.kaggle.com/c/my-competition",
        data_path=Path("data/competitions/my-competition"),
        llm_model="llama3",
        temperature=0.7
    )
    
    # 2. 初始化代理
    agent = ExampleAgent(config)
    
    # 3. 设置回调（可选）
    agent.set_callbacks(
        status_callback=lambda s: print(f"状态: {s.value}"),
        log_callback=lambda m: print(f"日志: {m}")
    )
    
    # 4. 运行
    problem_description = "你的问题描述..."
    data_info = {"columns": [...], "shape": (...)}
    
    result = await agent.run(problem_description, data_info)
    
    # 5. 查看结果
    print(f"状态: {result.status.value}")
    print(f"代码: {result.code_file_path}")
    print(f"提交: {result.submission_file_path}")
    print(f"耗时: {result.total_time:.2f}秒")
    
    return result

# 运行
asyncio.run(main())
```

## 📚 核心概念

### 1. AgentConfig（代理配置）

配置代理的所有参数：

```python
config = AgentConfig(
    # 必需参数
    agent_type=AgentType.REACT,          # 代理类型
    competition_name="competition-name",  # 竞赛名称
    competition_url="...",                # Kaggle链接
    data_path=Path("..."),                # 数据路径
    
    # LLM配置
    llm_model="llama3",                   # 模型名称
    temperature=0.7,                      # 温度参数
    max_tokens=4096,                      # 最大token
    
    # 执行配置
    max_execution_time=300,               # 超时（秒）
    max_memory_mb=2048,                   # 内存限制
    max_retries=3,                        # 重试次数
    
    # 输出配置
    output_dir=None,                      # 输出目录（默认自动生成）
    save_intermediate_results=True,       # 保存中间结果
    verbose=True                          # 详细日志
)
```

### 2. BaseAgent（基础代理）

所有代理的抽象基类，定义了标准接口：

**必须实现的方法：**

- `analyze_problem(problem_description)` - 分析问题
- `generate_code(problem_analysis, data_info)` - 生成代码
- `execute_code(code)` - 执行代码
- `get_metrics()` - 获取指标

**通用方法：**

- `run(problem_description, data_info)` - 运行完整流程
- `set_callbacks(status_callback, log_callback)` - 设置回调

### 3. AgentResult（执行结果）

包含代理运行的所有输出：

```python
result = await agent.run(...)

# 基本信息
result.status                    # 状态（COMPLETED/FAILED）
result.generated_code            # 生成的代码
result.code_file_path           # 代码文件路径
result.submission_file_path     # submission.csv路径

# 性能指标
result.total_time               # 总耗时
result.code_generation_time     # 代码生成耗时
result.execution_time           # 执行耗时
result.llm_calls               # LLM调用次数
result.code_lines              # 代码行数

# 中间过程
result.thoughts                # 思考过程
result.actions                 # 执行的动作
result.observations           # 观察结果

# 保存结果
result.save(Path("result.json"))
```

## 🏗️ 实现自定义代理

要创建自己的代理架构，继承 `BaseAgent`：

```python
from backend.agents.base_agent import BaseAgent, AgentConfig

class MyAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # 自定义初始化
        self.llm = self._init_llm()
    
    async def analyze_problem(self, problem_description: str):
        """分析问题"""
        # 调用LLM分析
        analysis = await self.llm.analyze(problem_description)
        self.result.llm_calls += 1
        return analysis
    
    async def generate_code(self, problem_analysis, data_info):
        """生成代码"""
        # 调用LLM生成代码
        code = await self.llm.generate(problem_analysis, data_info)
        self.result.llm_calls += 1
        return code
    
    async def execute_code(self, code: str):
        """执行代码"""
        # 在沙箱中执行
        result = self.executor.run(code)
        return result
    
    def get_metrics(self):
        """获取指标"""
        return {
            "total_time": self.result.total_time,
            "llm_calls": self.result.llm_calls,
            # ... 其他指标
        }
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_base_agent.py -v

# 运行带覆盖率的测试
pytest tests/ --cov=backend --cov-report=html
```

## 📂 项目结构

```
9900pj/
├── backend/
│   ├── agents/
│   │   ├── base_agent.py         # ✓ 基础代理（已完成）
│   │   ├── example_agent.py      # ✓ 示例代理（已完成）
│   │   ├── react_agent.py        # TODO: ReAct代理
│   │   ├── rag_agent.py          # TODO: RAG代理
│   │   └── multi_agent.py        # TODO: Multi-Agent
│   ├── kaggle/                   # TODO: Kaggle集成
│   ├── executor/                 # TODO: 代码执行引擎
│   ├── evaluation/               # TODO: 评估模块
│   ├── utils/
│   │   └── logger.py             # ✓ 日志工具（已完成）
│   └── config.py                 # ✓ 配置管理（已完成）
├── frontend/
│   └── streamlit_app.py          # TODO: Streamlit界面
├── data/                         # 数据目录
├── examples/
│   └── basic_usage.py            # ✓ 基础示例（已完成）
├── tests/
│   └── test_base_agent.py        # ✓ 测试（已完成）
└── requirements.txt              # ✓ 依赖（已完成）
```

## ✅ 当前进度

- ✅ 项目结构创建
- ✅ BaseAgent实现
- ✅ 配置管理
- ✅ 日志系统
- ✅ 测试框架
- ✅ 示例代码

**下一步：**
1. 实现Kaggle数据获取模块
2. 实现代码执行引擎
3. 实现具体的Agent（ReAct、RAG、Multi-Agent）
4. 创建Streamlit前端界面

## 🆘 常见问题

### Q: Ollama连接失败？
```bash
# 检查Ollama是否运行
curl http://localhost:11434/api/tags

# 重启Ollama
ollama serve
```

### Q: Kaggle API认证失败？
```bash
# 检查kaggle.json位置
ls -la ~/.kaggle/kaggle.json

# 检查权限
chmod 600 ~/.kaggle/kaggle.json
```

### Q: 如何查看详细日志？
日志文件位于 `logs/app_YYYY-MM-DD.log`

### Q: 如何修改LLM模型？
在配置中修改 `llm_model` 参数：
```python
config.llm_model = "llama3:70b"  # 使用更大的模型
config.llm_model = "mistral"     # 使用其他模型
```

## 📖 更多文档

- [架构设计](architecture.md)
- [Agent详细说明](../backend/agents/README.md)
- [API文档](api.md)

## 💬 获取帮助

如有问题，请：
1. 查看日志文件
2. 运行测试验证环境
3. 查看示例代码
4. 提交Issue到GitHub

---

**准备好了吗？开始你的AI Agent之旅！** 🚀

