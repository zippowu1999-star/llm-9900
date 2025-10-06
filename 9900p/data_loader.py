"""
数据加载和预处理模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from config import *

# 设置日志
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.stores_data = None
        self.oil_data = None
        self.holidays_data = None
        self.transactions_data = None
        self.merged_data = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据文件"""
        logger.info("开始加载数据文件...")
        
        try:
            # 加载训练数据
            self.train_data = pd.read_csv(TRAIN_DATA_PATH)
            logger.info(f"训练数据加载完成: {self.train_data.shape}")
            
            # 加载测试数据
            self.test_data = pd.read_csv(TEST_DATA_PATH)
            logger.info(f"测试数据加载完成: {self.test_data.shape}")
            
            # 加载商店数据
            self.stores_data = pd.read_csv(STORES_DATA_PATH)
            logger.info(f"商店数据加载完成: {self.stores_data.shape}")
            
            # 加载油价数据
            self.oil_data = pd.read_csv(OIL_DATA_PATH)
            logger.info(f"油价数据加载完成: {self.oil_data.shape}")
            
            # 加载节假日数据
            self.holidays_data = pd.read_csv(HOLIDAYS_DATA_PATH)
            logger.info(f"节假日数据加载完成: {self.holidays_data.shape}")
            
            # 加载交易数据
            self.transactions_data = pd.read_csv(TRANSACTIONS_DATA_PATH)
            logger.info(f"交易数据加载完成: {self.transactions_data.shape}")
            
            return {
                'train': self.train_data,
                'test': self.test_data,
                'stores': self.stores_data,
                'oil': self.oil_data,
                'holidays': self.holidays_data,
                'transactions': self.transactions_data
            }
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """数据预处理和特征工程"""
        logger.info("开始数据预处理...")
        
        if self.train_data is None:
            self.load_all_data()
        
        # 复制训练数据
        df = self.train_data.copy()
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # 合并商店信息
        df = df.merge(self.stores_data, on='store_nbr', how='left')
        
        # 处理油价数据
        self.oil_data['date'] = pd.to_datetime(self.oil_data['date'])
        self.oil_data['dcoilwtico'] = pd.to_numeric(self.oil_data['dcoilwtico'], errors='coerce')
        
        # 填充缺失的油价数据
        self.oil_data['dcoilwtico'] = self.oil_data['dcoilwtico'].ffill().bfill()
        
        df = df.merge(self.oil_data, on='date', how='left')
        
        # 处理节假日数据
        self.holidays_data['date'] = pd.to_datetime(self.holidays_data['date'])
        
        # 创建节假日标志
        holiday_flags = self.holidays_data.groupby('date')['type'].apply(lambda x: ','.join(x.unique())).reset_index()
        holiday_flags.columns = ['date', 'holiday_type']
        
        df = df.merge(holiday_flags, on='date', how='left')
        df['is_holiday'] = df['holiday_type'].notna().astype(int)
        
        # 处理交易数据
        self.transactions_data['date'] = pd.to_datetime(self.transactions_data['date'])
        df = df.merge(self.transactions_data, on=['date', 'store_nbr'], how='left')
        
        # 填充缺失值
        df['transactions'] = df['transactions'].fillna(0)
        
        # 创建滞后特征
        df = df.sort_values(['store_nbr', 'family', 'date'])
        
        # 按商店和产品类别分组创建滞后特征
        for lag in [1, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
            df[f'transactions_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['transactions'].shift(lag)
        
        # 创建移动平均特征
        for window in [7, 14, 30]:
            df[f'sales_ma_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].rolling(window=window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
            df[f'transactions_ma_{window}'] = df.groupby(['store_nbr', 'family'])['transactions'].rolling(window=window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        
        # 创建趋势特征
        df['sales_trend_7'] = df.groupby(['store_nbr', 'family'])['sales'].rolling(window=7, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0).reset_index(level=[0,1], drop=True)
        
        self.merged_data = df
        logger.info(f"数据预处理完成: {df.shape}")
        
        return df
    
    def get_feature_columns(self) -> list:
        """获取特征列名"""
        if self.merged_data is None:
            self.preprocess_data()
        
        # 排除目标变量和ID列
        exclude_cols = ['id', 'sales', 'date', 'store_nbr', 'family', 'holiday_type']
        feature_cols = [col for col in self.merged_data.columns if col not in exclude_cols]
        
        return feature_cols
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        if self.merged_data is None:
            self.preprocess_data()
        
        # 删除包含NaN的行
        df_clean = self.merged_data.dropna()
        
        feature_cols = self.get_feature_columns()
        X = df_clean[feature_cols]
        y = df_clean['sales']
        
        logger.info(f"训练数据准备完成: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要信息"""
        if self.merged_data is None:
            self.preprocess_data()
        
        summary = {
            'total_records': len(self.merged_data),
            'date_range': {
                'start': self.merged_data['date'].min(),
                'end': self.merged_data['date'].max()
            },
            'stores_count': self.merged_data['store_nbr'].nunique(),
            'families_count': self.merged_data['family'].nunique(),
            'sales_stats': {
                'mean': self.merged_data['sales'].mean(),
                'median': self.merged_data['sales'].median(),
                'std': self.merged_data['sales'].std(),
                'min': self.merged_data['sales'].min(),
                'max': self.merged_data['sales'].max()
            }
        }
        
        return summary
