"""
Kaggle数据获取示例

展示如何使用KaggleDataFetcher获取和分析竞赛数据
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.kaggle import KaggleDataFetcher


def main():
    """主函数"""
    print("=" * 60)
    print("Kaggle数据获取示例")
    print("=" * 60)
    
    # 初始化数据获取器
    print("\n步骤1: 初始化KaggleDataFetcher")
    try:
        fetcher = KaggleDataFetcher()
        print("✓ Kaggle API认证成功")
    except Exception as e:
        print(f"✗ Kaggle API认证失败: {e}")
        print("\n请配置Kaggle API凭证:")
        print("1. 从 https://www.kaggle.com/settings/account 下载 kaggle.json")
        print("2. 放置到 ~/.kaggle/kaggle.json")
        print("3. 或设置环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY")
        return
    
    # 测试竞赛URL
    # 这是Kaggle入门竞赛Titanic，数据量小，适合测试
    competition_url = "https://www.kaggle.com/competitions/titanic"
    
    print(f"\n步骤2: 提取竞赛ID")
    competition_id = fetcher.extract_competition_id(competition_url)
    print(f"✓ 竞赛ID: {competition_id}")
    
    # 获取完整信息（包括下载数据）
    print(f"\n步骤3: 获取完整竞赛信息")
    print("-" * 60)
    
    try:
        info = fetcher.fetch_complete_info(
            competition_url=competition_url,
            download_data=True,
            force_download=False  # 如果已下载则跳过
        )
        
        print("-" * 60)
        print("\n✓ 竞赛信息获取成功！")
        
        # 显示信息
        print("\n" + "=" * 60)
        print("竞赛详情")
        print("=" * 60)
        
        print(f"\n基本信息:")
        print(f"  ID: {info.competition_id}")
        print(f"  名称: {info.competition_name}")
        print(f"  URL: {info.competition_url}")
        print(f"  问题类型: {info.problem_type}")
        print(f"  评估指标: {info.evaluation_metric}")
        
        print(f"\n数据文件:")
        print(f"  训练文件: {info.train_files}")
        print(f"  测试文件: {info.test_files}")
        print(f"  提交样例: {info.sample_submission_file}")
        
        if info.train_shape:
            print(f"\n数据规模:")
            print(f"  训练集: {info.train_shape}")
            if info.test_shape:
                print(f"  测试集: {info.test_shape}")
        
        if info.columns:
            print(f"\n数据列 (共{len(info.columns)}列):")
            for i, col in enumerate(info.columns[:10], 1):
                col_type = info.column_types.get(col, "unknown")
                print(f"  {i}. {col}: {col_type}")
            if len(info.columns) > 10:
                print(f"  ... 还有 {len(info.columns) - 10} 列")
        
        # 生成数据摘要（用于LLM）
        print("\n" + "=" * 60)
        print("数据摘要（用于LLM）")
        print("=" * 60)
        summary = fetcher.get_data_summary(info)
        print(summary)
        
        print("\n" + "=" * 60)
        print("数据已保存到:")
        print(f"  {info.data_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 获取竞赛信息失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

