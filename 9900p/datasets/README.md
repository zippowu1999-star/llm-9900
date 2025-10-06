# 数据集目录说明

## Kaggle Store Sales Time Series Forecasting

### 数据文件结构
```
datasets/kaggle_task1/
├── train.csv                    # 训练数据
├── test.csv                     # 测试数据
├── stores.csv                   # 商店信息
├── oil.csv                      # 油价数据
├── holidays_events.csv          # 节假日数据
└── transactions.csv             # 交易数据
```

### 数据下载步骤

1. **访问Kaggle竞赛页面**
   - 链接: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

2. **下载数据文件**
   - 需要Kaggle账号
   - 接受竞赛规则
   - 下载所有CSV文件

3. **放置数据文件**
   - 将所有CSV文件放在 `datasets/kaggle_task1/` 目录下
   - 确保文件名与上述结构一致

### 数据文件说明

#### train.csv
- **用途**: 训练数据，包含历史销售数据
- **列**: id, date, store_nbr, family, sales, onpromotion
- **时间范围**: 2013-01-01 到 2017-08-15

#### test.csv  
- **用途**: 测试数据，需要预测的销售数据
- **列**: id, date, store_nbr, family, onpromotion
- **时间范围**: 2017-08-16 到 2017-08-31

#### stores.csv
- **用途**: 商店信息
- **列**: store_nbr, city, state, type, cluster

#### oil.csv
- **用途**: 油价数据
- **列**: date, dcoilwtico

#### holidays_events.csv
- **用途**: 节假日和特殊事件
- **列**: date, type, locale, locale_name, description, transferred

#### transactions.csv
- **用途**: 交易数据
- **列**: date, store_nbr, transactions

### 使用说明

1. **确保数据完整性**
   - 检查所有文件是否存在
   - 验证文件大小和格式

2. **运行演示**
   ```bash
   # 运行Kaggle时间序列演示
   python kaggle_timeseries_demo.py
   
   # 或运行Jupyter Notebook
   jupyter notebook demos/ai_agent_comparison_demo.ipynb
   ```

3. **数据预处理**
   - 脚本会自动处理日期格式转换
   - 合并相关数据表
   - 进行数据质量检查

### 注意事项

- 确保有足够的磁盘空间（约100MB）
- 数据文件较大时，处理时间可能较长
- 建议使用SSD存储以提高读取速度
- 如果数据文件缺失，系统会自动使用模拟数据

