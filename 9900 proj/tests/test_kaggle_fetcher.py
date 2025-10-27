"""
Kaggle数据获取器测试
"""
import pytest
from pathlib import Path
from backend.kaggle import KaggleDataFetcher, CompetitionInfo


class TestKaggleDataFetcher:
    """测试KaggleDataFetcher"""
    
    def test_extract_competition_id(self):
        """测试竞赛ID提取"""
        fetcher = KaggleDataFetcher()
        
        # 测试不同格式的URL
        urls = [
            ("https://www.kaggle.com/competitions/titanic", "titanic"),
            ("https://www.kaggle.com/c/titanic", "titanic"),
            ("titanic", "titanic"),
            ("https://www.kaggle.com/competitions/store-sales-time-series-forecasting", 
             "store-sales-time-series-forecasting"),
        ]
        
        for url, expected_id in urls:
            result = fetcher.extract_competition_id(url)
            assert result == expected_id, f"URL: {url}, Expected: {expected_id}, Got: {result}"
    
    def test_infer_problem_type(self, mocker):
        """测试问题类型推断"""
        fetcher = KaggleDataFetcher()
        
        # 时间序列
        info = CompetitionInfo(
            competition_id="test",
            competition_name="Test",
            competition_url="http://test.com",
            title="Store Sales Time Series Forecasting",
            description="Forecast sales using time series",
            evaluation_metric="RMSE"
        )
        assert fetcher._infer_problem_type(info) == "time_series_forecasting"
        
        # 分类
        info.title = "Binary Classification"
        info.description = "Classify into two classes"
        info.evaluation_metric = "AUC"
        assert fetcher._infer_problem_type(info) == "classification"
        
        # 回归
        info.title = "House Price Prediction"
        info.description = "Predict house prices"
        info.evaluation_metric = "RMSE"
        assert fetcher._infer_problem_type(info) == "regression"


class TestCompetitionInfo:
    """测试CompetitionInfo"""
    
    def test_competition_info_creation(self):
        """测试创建竞赛信息"""
        info = CompetitionInfo(
            competition_id="test-comp",
            competition_name="Test Competition",
            competition_url="http://test.com"
        )
        
        assert info.competition_id == "test-comp"
        assert info.competition_name == "Test Competition"
        assert info.train_files == []
        assert info.test_files == []
    
    def test_to_dict(self):
        """测试转换为字典"""
        info = CompetitionInfo(
            competition_id="test",
            competition_name="Test",
            competition_url="http://test.com",
            title="Test Title",
            problem_type="classification"
        )
        
        info_dict = info.to_dict()
        assert info_dict["competition_id"] == "test"
        assert info_dict["problem_type"] == "classification"
        assert "title" in info_dict
    
    def test_save_and_load(self, tmp_path):
        """测试保存和加载"""
        info = CompetitionInfo(
            competition_id="test",
            competition_name="Test",
            competition_url="http://test.com"
        )
        
        save_path = tmp_path / "test_info.json"
        info.save(save_path)
        
        assert save_path.exists()
        
        # 读取并验证
        import json
        with open(save_path) as f:
            loaded = json.load(f)
        assert loaded["competition_id"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

