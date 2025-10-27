#!/bin/bash

# Kaggle AI Agent 系统启动脚本

echo "🚀 启动 Kaggle AI Agent 系统..."
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import streamlit" 2>/dev/null || {
    echo "⚠️  Streamlit未安装，正在安装依赖..."
    pip install -r requirements.txt
}

# 检查环境变量
if [ ! -f ".env" ]; then
    echo "⚠️  警告: 未找到.env文件"
    echo "💡 请创建.env文件并设置OPENAI_API_KEY"
fi

# 检查Kaggle凭证
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "⚠️  警告: 未找到Kaggle凭证"
    echo "💡 请将kaggle.json放到 ~/.kaggle/ 目录"
fi

echo ""
echo "✅ 启动Streamlit应用..."
echo "🌐 浏览器将自动打开 http://localhost:8501"
echo ""

# 启动Streamlit
streamlit run app.py --server.port 8501 --server.address localhost

