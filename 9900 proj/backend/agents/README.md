# AI Agents 模块

这个模块包含了不同架构的AI代理实现。

## 架构说明

### BaseAgent（基础代理）

`base_agent.py` 定义了所有AI代理的抽象基类和核心数据结构：

#### 核心类

1. **AgentConfig**：代理配置类
   - 基础配置：竞赛名称、URL、数据路径
   - LLM配置：模型、温度、token限制
   - 执行配置：超时、内存限制、重试次数
   - 输出配置：保存路径、日志设置

2. **AgentResult**：执行结果类
   - 生成的代码和文件路径
   - 执行输出和错误信息
   - 性能指标（时间、LLM调用次数等）
   - 中间过程（思考、动作、观察）
   - 评估指标

3. **BaseAgent**：抽象基类
   - 必须实现的方法：
     - `analyze_problem()`: 分析问题描述
     - `generate_code()`: 生成数据分析代码
     - `execute_code()`: 执行生成的代码
     - `get_metrics()`: 获取评估指标
   - 通用方法：
     - `run()`: 运行完整流程
     - 状态管理和回调
     - 结果保存

## 使用示例

```python
from pathlib import Path
from backend.agents import BaseAgent, AgentConfig, AgentType
from backend.agents.react_agent import ReactAgent

# 1. 创建配置
config = AgentConfig(
    agent_type=AgentType.REACT,
    competition_name="store-sales-forecasting",
    competition_url="https://www.kaggle.com/competitions/store-sales-time-series-forecasting",
    data_path=Path("data/competitions/store-sales"),
    llm_model="llama3",
    temperature=0.7
)

# 2. 初始化代理
agent = ReactAgent(config)

# 3. 设置回调（可选）
def on_status_change(status):
    print(f"状态更新: {status.value}")

def on_log(message):
    print(f"日志: {message}")

agent.set_callbacks(
    status_callback=on_status_change,
    log_callback=on_log
)

# 4. 运行代理
problem_description = "预测商店未来的销售额..."
data_info = {
    "train_shape": (1000, 10),
    "columns": ["date", "store_id", "sales", ...]
}

result = await agent.run(problem_description, data_info)

# 5. 获取结果
print(f"状态: {result.status.value}")
print(f"代码文件: {result.code_file_path}")
print(f"提交文件: {result.submission_file_path}")
print(f"总耗时: {result.total_time}秒")
print(f"指标: {agent.get_metrics()}")
```

## 实现新的代理

要实现一个新的代理架构，继承 `BaseAgent` 并实现抽象方法：

```python
from backend.agents.base_agent import BaseAgent, AgentConfig

class MyCustomAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # 自定义初始化
    
    async def analyze_problem(self, problem_description: str):
        # 实现问题分析逻辑
        pass
    
    async def generate_code(self, problem_analysis, data_info):
        # 实现代码生成逻辑
        pass
    
    async def execute_code(self, code: str):
        # 实现代码执行逻辑
        pass
    
    def get_metrics(self):
        # 返回评估指标
        pass
```

## 待实现的代理

- [ ] `react_agent.py` - ReAct架构（推理-行动循环）
- [ ] `rag_agent.py` - RAG架构（检索增强生成）
- [ ] `multi_agent.py` - Multi-Agent架构（多代理协作）

## 状态流转

```
IDLE (空闲)
  ↓
INITIALIZING (初始化)
  ↓
ANALYZING (分析问题)
  ↓
GENERATING_CODE (生成代码)
  ↓
EXECUTING (执行代码)
  ↓
COMPLETED (完成) / FAILED (失败)
```

## 性能指标

每个代理都会记录以下指标：

- **总耗时**：从开始到结束的时间
- **代码生成时间**：生成代码所需时间
- **执行时间**：代码执行所需时间
- **LLM调用次数**：调用大模型的次数
- **代码行数**：生成的代码行数
- **成功率**：是否成功生成submission.csv

## 扩展性

- 通过 `AgentConfig.extra_config` 可以传递自定义配置
- 通过回调函数可以实时更新UI状态
- 支持保存中间结果用于调试和分析

