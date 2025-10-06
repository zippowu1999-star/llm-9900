#!/usr/bin/env python3
"""
简化版AI代理 - 只做预测，返回销售数据
"""

import pandas as pd
import numpy as np
import time
import logging
import json
import re
import asyncio
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from data_loader import DataLoader
from config import OPENAI_API_KEY, OPENAI_MODEL

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAgent:
    """简化版AI代理 - 只做销售预测"""
    
    def __init__(self, api_key: str = None, model: str = None, data_loader: DataLoader = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置")
        
        self.client = OpenAI(api_key=self.api_key)
        self.data_loader = data_loader if data_loader is not None else DataLoader()
        
        # 预计算销售统计摘要，避免重复计算
        self.sales_summary = None
        self._prepare_sales_summary()
        
        logger.info(f"简化AI代理初始化完成，使用模型: {self.model}")
    
    def _prepare_sales_summary(self):
        """预计算所有商店和产品类别的销售统计，避免重复计算"""
        logger.info("🚀 预计算销售统计摘要...")
        
        try:
            train_data = self.data_loader.train_data
            
            # 计算基础统计指标
            summary = (
                train_data.groupby(['store_nbr', 'family'])['sales']
                .agg(
                    avg_sales='mean',
                    min_sales='min', 
                    max_sales='max',
                    zero_ratio=lambda x: (x == 0).mean()
                )
                .reset_index()
            )
            
            # 计算最近7天平均值
            recent_avg_data = []
            for (store_nbr, family), group in train_data.groupby(['store_nbr', 'family']):
                recent_avg = group.tail(7)['sales'].mean() if len(group) > 0 else 0.0
                recent_avg_data.append({
                    'store_nbr': store_nbr,
                    'family': family,
                    'recent_avg': recent_avg
                })
            
            recent_avg_df = pd.DataFrame(recent_avg_data)
            
            # 合并数据
            self.sales_summary = summary.merge(
                recent_avg_df, 
                on=['store_nbr', 'family'], 
                how='left'
            )
            
            logger.info(f"✅ 销售统计摘要计算完成，共 {len(self.sales_summary):,} 个商店-产品组合")
            
        except Exception as e:
            logger.error(f"预计算销售统计失败: {e}")
            self.sales_summary = pd.DataFrame()
    
    def _get_historical_sales(self, store_nbr: int, family: str) -> Dict:
        """获取历史销售数据 - 使用预计算的结果，O(1)查询"""
        try:
            # 从预计算的摘要中查找
            if self.sales_summary is not None and len(self.sales_summary) > 0:
                row = self.sales_summary[
                    (self.sales_summary['store_nbr'] == store_nbr) &
                    (self.sales_summary['family'] == family)
                ]
                
                if len(row) > 0:
                    return {
                        'avg_sales': float(row.iloc[0]['avg_sales']),
                        'recent_avg': float(row.iloc[0]['recent_avg']),
                        'min_sales': float(row.iloc[0]['min_sales']),
                        'max_sales': float(row.iloc[0]['max_sales']),
                        'zero_ratio': float(row.iloc[0]['zero_ratio'])
                    }
            
            # 如果预计算数据不可用，返回默认值
            return {
                'avg_sales': 0.0,
                'recent_avg': 0.0,
                'min_sales': 0.0,
                'max_sales': 0.0,
                'zero_ratio': 1.0
            }
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return {
                'avg_sales': 0.0,
                'recent_avg': 0.0,
                'min_sales': 0.0,
                'max_sales': 0.0,
                'zero_ratio': 1.0
            }
    
    def _get_data_summary(self) -> Dict:
        """获取数据摘要信息"""
        try:
            train_data = self.data_loader.train_data
            
            return {
                'total_records': len(train_data),
                'date_range': {
                    'start': str(train_data['date'].min()),
                    'end': str(train_data['date'].max())
                },
                'stores_count': train_data['store_nbr'].nunique(),
                'families_count': train_data['family'].nunique(),
                'sales_stats': {
                    'mean': train_data['sales'].mean(),
                    'median': train_data['sales'].median(),
                    'std': train_data['sales'].std(),
                    'min': train_data['sales'].min(),
                    'max': train_data['sales'].max()
                }
            }
        except Exception as e:
            logger.error(f"获取数据摘要失败: {e}")
            return {
                'total_records': 0,
                'date_range': {'start': '', 'end': ''},
                'stores_count': 0,
                'families_count': 0,
                'sales_stats': {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
            }
    
    def _get_context_info(self, sample_data: Dict) -> Dict:
        """获取日期相关的上下文信息（油价、节假日等）"""
        try:
            date_str = sample_data.get('date', '')
            if not date_str:
                return {
                    'oil_price': 'N/A',
                    'is_holiday': 'N/A', 
                    'holiday_type': 'N/A',
                    'recent_oil_avg': 'N/A'
                }
            
            # 获取油价信息
            oil_data = self.data_loader.oil_data
            target_date = pd.to_datetime(date_str)
            
            # 当前日期油价
            oil_row = oil_data[oil_data['date'] == target_date]
            oil_price = oil_row['dcoilwtico'].iloc[0] if len(oil_row) > 0 else None
            
            # 最近7天平均油价
            recent_dates = oil_data[oil_data['date'] <= target_date].tail(7)
            recent_oil_avg = recent_dates['dcoilwtico'].mean() if len(recent_dates) > 0 else None
            
            # 获取节假日信息
            holidays_data = self.data_loader.holidays_data
            holiday_row = holidays_data[holidays_data['date'] == target_date]
            is_holiday = len(holiday_row) > 0
            holiday_type = holiday_row['type'].iloc[0] if len(holiday_row) > 0 else None
            
            return {
                'oil_price': f"{oil_price:.2f}" if oil_price is not None else 'N/A',
                'is_holiday': '是' if is_holiday else '否',
                'holiday_type': holiday_type if holiday_type else '无',
                'recent_oil_avg': f"{recent_oil_avg:.2f}" if recent_oil_avg is not None else 'N/A'
            }
            
        except Exception as e:
            logger.error(f"获取上下文信息失败: {e}")
            return {
                'oil_price': 'N/A',
                'is_holiday': 'N/A',
                'holiday_type': 'N/A', 
                'recent_oil_avg': 'N/A'
            }
    
    def predict_sales(self, sample_data: Dict, retry_count: int = 0) -> Dict:
        """预测单个样本的销售额"""
        try:
            # 获取历史数据上下文
            store_nbr = sample_data.get('store_nbr', 1)
            family = sample_data.get('family', '')
            
            # 从训练数据中获取该商店和产品类别的历史销售数据
            historical_data = self._get_historical_sales(store_nbr, family)
            
            # 获取更详细的数据摘要
            data_summary = self._get_data_summary()
            
            # 获取节假日和油价上下文
            context_info = self._get_context_info(sample_data)
            
            prompt = f"""
基于以下详细数据预测销售额：

预测样本:
- 商店编号: {sample_data.get('store_nbr', 'N/A')}
- 产品类别: {sample_data.get('family', 'N/A')}
- 日期: {sample_data.get('date', 'N/A')}
- 促销状态: {sample_data.get('onpromotion', 'N/A')}

全局数据摘要:
- 总记录数: {data_summary['total_records']:,}
- 日期范围: {data_summary['date_range']['start']} 到 {data_summary['date_range']['end']}
- 商店数量: {data_summary['stores_count']}
- 产品类别数量: {data_summary['families_count']}
- 全局平均销售额: ${data_summary['sales_stats']['mean']:.2f}
- 全局销售额范围: ${data_summary['sales_stats']['min']:.2f} - ${data_summary['sales_stats']['max']:.2f}

该商店该产品类别历史数据:
- 平均销售额: ${historical_data.get('avg_sales', 0):.2f}
- 最近7天平均销售额: ${historical_data.get('recent_avg', 0):.2f}
- 历史销售额范围: ${historical_data.get('min_sales', 0):.2f} - ${historical_data.get('max_sales', 0):.2f}
- 零销售天数占比: {historical_data.get('zero_ratio', 0):.1%}

日期上下文信息:
- 油价: ${context_info.get('oil_price', 'N/A')}
- 是否节假日: {context_info.get('is_holiday', 'N/A')}
- 节假日类型: {context_info.get('holiday_type', 'N/A')}
- 最近7天平均油价: ${context_info.get('recent_oil_avg', 'N/A')}

请综合考虑油价波动、节假日效应、促销状态和历史销售模式进行预测。

请严格按照以下JSON格式回答：
```json
{{
    "predicted_sales": 数值
}}
```
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个销售预测专家，请根据给定信息预测销售额。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,  # 减少token使用
                temperature=0.1
            )
            
            # 解析JSON响应
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{[^}]*"predicted_sales"[^}]*\}', response_text)
            
            if json_match:
                result = json.loads(json_match.group())
                predicted_sales = result.get('predicted_sales', 0.0)
            else:
                logger.warning("JSON解析失败，使用默认值")
                predicted_sales = 0.0
            
            return {
                'id': sample_data.get('id', ''),
                'predicted_sales': float(predicted_sales)
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # 处理速率限制
            if "429" in error_msg and retry_count < 3:
                logger.warning(f"速率限制，等待重试... (第{retry_count + 1}次)")
                time.sleep(2 ** retry_count)  # 指数退避
                return self.predict_sales(sample_data, retry_count + 1)
            
            logger.error(f"预测失败: {e}")
            return {
                'id': sample_data.get('id', ''),
                'predicted_sales': 0.0
            }
    
    def batch_predict(self, test_data: pd.DataFrame, batch_size: int = 1000) -> List[Dict]:
        """批量预测"""
        logger.info(f"开始批量预测，总样本数: {len(test_data):,}")
        
        all_predictions = []
        total_samples = len(test_data)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_data = test_data.iloc[i:batch_end]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"处理批次 {batch_num}/{num_batches} (样本 {i+1:,}-{batch_end:,})")
            
            # 串行预测，避免并发请求导致速率限制
            batch_predictions = []
            for _, row in batch_data.iterrows():
                try:
                    result = self.predict_sales(row.to_dict())
                    batch_predictions.append(result)
                    logger.info(f"   ✅ 预测完成: ID {result['id']}, 销售额 ${result['predicted_sales']:.2f}")
                except Exception as e:
                    logger.error(f"预测失败: {e}")
                    batch_predictions.append({
                        'id': row.get('id', ''),
                        'predicted_sales': 0.0
                    })
            
            # 批次间添加延迟，避免速率限制
            if batch_num < num_batches:
                time.sleep(1)
            
            all_predictions.extend(batch_predictions)
            
            # 显示进度
            progress = (batch_end / total_samples) * 100
            logger.info(f"   ✅ 批次完成，进度: {progress:.1f}%")
        
        return all_predictions
