"""
配置文件 - AI代理系统设置
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# OpenAI API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# 数据路径配置
DATA_DIR = "datasets/store-sales-time-series-forecasting"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")
STORES_DATA_PATH = os.path.join(DATA_DIR, "stores.csv")
OIL_DATA_PATH = os.path.join(DATA_DIR, "oil.csv")
HOLIDAYS_DATA_PATH = os.path.join(DATA_DIR, "holidays_events.csv")
TRANSACTIONS_DATA_PATH = os.path.join(DATA_DIR, "transactions.csv")

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
