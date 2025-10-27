# AI Agent System For Data Analytics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

一个基于多种AI代理架构的数据分析系统，能够自动解决Kaggle竞赛问题。

## 🎯 项目概述

本项目实现了三种不同的AI代理架构（ReAct、RAG、Multi-Agent），用于自动化数据分析和Kaggle竞赛求解。用户只需输入Kaggle竞赛链接，选择AI架构，系统即可自动：
- 获取和分析数据
- 生成分析代码
- 执行代码
- 生成submission.csv

## 🏗️ 系统架构

### 三种AI代理架构

1. **ReAct Agent（推理-行动循环）**
   - 基于思考-行动-观察的循环
   - 适合需要多步骤推理的任务

2. **RAG Agent（检索增强生成）**
   - 从知识库检索相关解决方案
   - 结合历史案例生成代码

3. **Multi-Agent System（多代理协作）**
   - 多个专门代理分工协作
   - 包含规划、EDA、特征工程、建模等角色

## 📁 项目结构

```
ai-agent-analytics/
├── backend/
│   ├── agents/              # AI代理实现
│   ├── kaggle/              # Kaggle集成
│   ├── executor/            # 代码执行引擎
│   ├── evaluation/          # 评估模块
│   └── utils/               # 工具函数
├── frontend/
│   └── streamlit_app.py     # Streamlit应用
├── data/                    # 数据目录
├── knowledge_base/          # RAG知识库
├── notebooks/               # 演示Notebook
├── tests/                   # 测试
└── requirements.txt
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-repo/ai-agent-analytics.git
cd ai-agent-analytics

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置Kaggle API

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，填入你的Kaggle凭证
# 从 https://www.kaggle.com/settings/account 获取API token
```

### 3. 启动Ollama（本地LLM）

```bash
# 安装Ollama: https://ollama.ai/
ollama pull llama3
ollama serve
```

### 4. 运行应用

```bash
# 启动Streamlit前端
streamlit run frontend/streamlit_app.py
```

## 📊 评估指标

系统会对不同架构进行多维度评估：
- ⏱️ 执行时间
- 📈 预测准确度
- 💻 代码复杂度
- 🔍 可解释性
- 🤖 自主性水平
- 💾 LLM模型大小要求

## 🛠️ 技术栈

- **AI框架**: LangChain, LangGraph
- **LLM**: Ollama (Llama3)
- **前端**: Streamlit
- **后端**: FastAPI
- **数据处理**: Pandas, NumPy, Scikit-learn
- **可视化**: Plotly, Matplotlib, Seaborn

## 📝 使用示例

1. 访问 http://localhost:8501
2. 输入Kaggle竞赛链接（例如：https://www.kaggle.com/competitions/store-sales-time-series-forecasting）
3. 选择AI代理架构（ReAct / RAG / Multi-Agent）
4. 点击"开始生成和运行"
5. 查看实时日志和生成的代码
6. 下载生成的submission.csv

## 📖 文档

- [架构设计文档](docs/architecture.md)
- [API文档](docs/api.md)
- [开发指南](docs/development.md)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 [MIT License](LICENSE)

## 👥 团队

UNSW CSE FAIC - 9900 Project Group

## 📧 联系方式

如有问题，请联系：[your-email@example.com]

