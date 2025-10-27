# 🤖 Kaggle AI Agent 系统 - 使用指南

## 📖 简介

Kaggle AI Agent 系统是一个智能数据分析平台，能够自动解决Kaggle竞赛问题。支持多种AI架构（ReAct, RAG, Multi-Agent），自动生成、执行和优化数据分析代码。

## ✨ 主要功能

- 🎯 **自动数据获取** - 输入Kaggle竞赛URL，自动下载数据
- 🤖 **多架构支持** - ReAct, RAG, Multi-Agent三种AI架构
- 💻 **代码自动生成** - LLM自动生成完整数据分析代码
- 🔧 **智能错误修复** - 代码执行失败时自动修复（最多3次）
- 📊 **实时进度展示** - 可视化显示执行过程和日志
- 📥 **结果下载** - 一键下载生成的代码和submission.csv

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置OpenAI API Key（创建.env文件）
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 配置Kaggle凭证（将kaggle.json放到~/.kaggle/）
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. 启动应用

**方式1: 使用启动脚本（推荐）**
```bash
./run_app.sh
```

**方式2: 直接启动**
```bash
streamlit run app.py
```

应用将在 http://localhost:8501 启动

### 3. 使用流程

#### 步骤1: 输入任务
1. 在"输入任务"标签页输入Kaggle竞赛URL
   - 例如: `https://www.kaggle.com/competitions/titanic`
   - 或直接输入竞赛名称: `titanic`
2. 点击"📥 获取数据"按钮
3. 查看竞赛信息和数据文件详情
4. 点击"🚀 开始生成解决方案"

#### 步骤2: 执行过程
- 系统自动切换到"执行过程"标签页
- 实时显示Agent执行状态
- 查看详细执行日志

#### 步骤3: 结果分析
- 执行完成后查看结果
- 查看生成的代码
- 下载submission.csv文件
- 查看执行指标和错误信息

## ⚙️ 高级配置

### 侧边栏配置选项

**1. Agent类型**
- **ReAct** ✅ (推理-行动循环，已实现)
- **RAG** 🔨 (检索增强生成，开发中)
- **Multi-Agent** 🔨 (多代理协作，开发中)

**2. 高级设置**
- **LLM模型**: gpt-4o-mini / gpt-4o / gpt-4-turbo
- **Temperature**: 0.0 - 1.0（控制生成随机性）
- **最大重试次数**: 1 - 5次
- **最大执行时间**: 60 - 1200秒

## 📊 界面说明

### 主界面三个标签页

#### 📥 输入任务
- 输入Kaggle竞赛URL或名称
- 获取并查看竞赛数据信息
- 查看所有数据文件详情
- 开始执行任务

#### 🚀 执行过程
- 实时进度条
- 状态更新
- 详细执行日志
- 错误提示

#### 📊 结果分析
- **状态总览**: 成功/失败、总耗时、LLM调用次数、代码行数
- **生成的代码**: 完整Python代码，支持下载
- **执行指标**: 各阶段耗时、思考步骤、执行动作
- **Submission文件**: 预览前10行，支持下载
- **错误信息**: 如果执行失败，显示详细错误
- **详细日志**: 思考过程、执行动作、观察结果

## 🎨 界面特性

- ✨ **现代化设计** - 美观的渐变卡片和动画效果
- 📱 **响应式布局** - 支持不同屏幕尺寸
- 🎯 **直观操作** - 简洁的三步流程
- 📈 **数据可视化** - 清晰的指标展示
- ⬇️ **便捷下载** - 一键下载代码和结果

## 💡 使用技巧

1. **首次运行较慢** - Kaggle数据下载需要时间，请耐心等待
2. **选择合适的模型** - gpt-4o-mini速度快成本低，gpt-4o质量更高
3. **调整Temperature** - 数据分析任务建议使用0.2-0.4的低温度
4. **查看详细日志** - 如果执行失败，展开详细日志查看具体原因
5. **多次尝试** - 如果首次失败，可以调整参数后重新运行

## 🔍 示例任务

### 示例1: Titanic生存预测
```
URL: https://www.kaggle.com/competitions/titanic
或直接输入: titanic
```

### 示例2: House Prices预测
```
URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
或直接输入: house-prices-advanced-regression-techniques
```

### 示例3: Store Sales时间序列预测
```
URL: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
或直接输入: store-sales-time-series-forecasting
```

## ⚠️ 常见问题

### Q: 提示"未找到Kaggle凭证"
A: 需要将kaggle.json放到 `~/.kaggle/` 目录并设置权限：
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Q: 提示"OpenAI API Key未设置"
A: 创建.env文件并设置API Key：
```bash
echo "OPENAI_API_KEY=your-key" > .env
```

### Q: 代码生成后执行失败
A: 系统会自动重试3次。如果仍失败：
1. 查看详细错误日志
2. 尝试降低数据采样率
3. 选择更强大的模型（如gpt-4o）

### Q: 下载数据很慢
A: Kaggle某些竞赛数据较大（如Store Sales有几百MB），请耐心等待

## 📦 项目结构

```
9900pj/
├── app.py                 # Streamlit前端主文件
├── run_app.sh            # 启动脚本
├── backend/              # 后端模块
│   ├── agents/          # AI Agent实现
│   ├── kaggle/          # Kaggle数据获取
│   ├── executor/        # 代码执行引擎
│   ├── llm/             # LLM客户端
│   └── evaluation/      # 评估指标
├── data/                # 数据目录
│   ├── competitions/    # Kaggle竞赛数据
│   └── generated_code/  # 生成的代码
└── requirements.txt     # Python依赖
```

## 🛠 技术栈

- **前端**: Streamlit 1.32+
- **后端**: Python 3.10+
- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **数据处理**: Pandas, NumPy, Scikit-learn
- **代码执行**: subprocess (沙盒模式)
- **日志**: Loguru

## 📝 开发路线图

- [x] ✅ ReAct Agent架构
- [x] ✅ Streamlit前端界面
- [x] ✅ Kaggle数据自动获取
- [x] ✅ 代码自动生成和执行
- [x] ✅ 错误自动修复机制
- [ ] 🔨 RAG架构（检索增强）
- [ ] 🔨 Multi-Agent架构（多代理协作）
- [ ] 🔨 Agent性能对比分析
- [ ] 🔨 历史任务管理
- [ ] 🔨 代码优化建议

## 📄 License

MIT License

## 👥 作者

COMP9900 项目组

---

**祝您使用愉快！🎉** 如有问题请查看详细日志或联系开发团队。




