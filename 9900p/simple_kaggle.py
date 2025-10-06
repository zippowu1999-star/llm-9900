#!/usr/bin/env python3
"""
超级简化版Kaggle预测脚本
功能：加载数据 -> 预测 -> 生成提交文件
"""

import pandas as pd
import time
import logging
from simple_agent import SimpleAgent
from data_loader import DataLoader
from config import OPENAI_API_KEY

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("🏆 Kaggle预测 - 简化版")
    print("=" * 40)
    
    try:
        # 1. 加载数据
        logger.info("📊 加载数据...")
        data_loader = DataLoader()
        data_loader.load_all_data()
        data_loader.preprocess_data()
        
        # 2. 初始化AI代理
        logger.info("🤖 初始化AI代理...")
        agent = SimpleAgent(api_key=OPENAI_API_KEY, data_loader=data_loader)
        
        # 3. 开始预测（只测试前5个样本）
        logger.info("🔮 开始预测（测试前5个样本）...")
        test_data = data_loader.test_data.head(5)  # 只取前5个样本
        start_time = time.time()
        
        predictions = agent.batch_predict(test_data, batch_size=5)
        
        prediction_time = time.time() - start_time
        
        # 4. 生成提交文件
        logger.info("📁 生成提交文件...")
        submission_data = []
        for pred in predictions:
            submission_data.append({
                'id': pred['id'],
                'sales': pred['predicted_sales']
            })
        
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv('test_submission.csv', index=False)
        
        # 5. 显示结果
        print("\n" + "=" * 40)
        print("🎉 预测完成!")
        print(f"⏱️  耗时: {prediction_time:.2f} 秒")
        print(f"🚀 速度: {len(predictions)/prediction_time:.2f} 预测/秒")
        print(f"📁 文件: test_submission.csv")
        print(f"📊 样本数: {len(predictions):,}")
        print(f"💰 平均销售额: ${submission_df['sales'].mean():.2f}")
        print("=" * 40)
        
    except Exception as e:
        logger.error(f"❌ 预测失败: {e}")
        raise

if __name__ == "__main__":
    main()
