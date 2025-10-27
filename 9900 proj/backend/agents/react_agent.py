"""
ReAct Agent实现（时间序列优化版）

基于Reasoning + Acting的AI代理架构
适用于Kaggle时间序列预测类任务（生成可提交submission.csv）
"""

from typing import Dict, Any
from pathlib import Path
from backend.agents.base_agent import BaseAgent, AgentConfig
from backend.llm import LLMClient
from backend.executor import CodeExecutor
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ReactAgent(BaseAgent):
    """
    ReAct Agent - 推理 + 行动循环架构
    专为时间序列任务优化
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # 初始化 LLM 客户端
        self.llm = LLMClient(
            provider="openai",
            model=config.llm_model if config.llm_model != "llama3" else "gpt-4o-mini",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        # 初始化代码执行器
        self.executor = CodeExecutor(
            mode="subprocess",
            timeout=config.max_execution_time,
            max_memory_mb=config.max_memory_mb
        )

        self._log("✅ ReactAgent（时间序列优化版）初始化完成")

    # ---------- 🧠 Step 1: 问题分析 ----------
    async def analyze_problem(self, problem_description: str) -> Dict[str, Any]:
        """分析Kaggle任务类型与策略"""
        self._log("开始分析任务描述...")

        system_prompt = """你是Kaggle顶级选手，擅长时间序列预测任务。
请阅读以下竞赛描述，判断其是否为时间序列问题，并总结核心思路。

请以JSON格式返回分析结果：
{
  "problem_type": "time_series",
  "key_insights": ["主要预测销售量", "存在日期列", "可用辅助数据"],
  "suggested_approach": "构建时间特征 + 滞后特征 + LightGBM回归",
  "data_requirements": ["train.csv", "test.csv", "辅助数据文件"]
}"""

        prompt = f"请分析以下Kaggle问题描述：\n\n{problem_description}"

        try:
            response = self.llm.generate(prompt, system_prompt)
            self.result.llm_calls += 1
            text = response.content

            return {
                "problem_type": self._extract_problem_type(text),
                "key_insights": self._extract_insights(text),
                "suggested_approach": "sklearn RandomForest时间序列预测",
                "data_requirements": ["train.csv", "test.csv", "辅助文件（如oil, holidays, transactions）"],
                "problem_description": problem_description
            }

        except Exception as e:
            self._log(f"分析失败: {e}", level="error")
            return {
                "problem_type": "time_series",
                "key_insights": ["自动识别为时间序列任务"],
                "suggested_approach": "sklearn RandomForest baseline",
                "data_requirements": ["train.csv", "test.csv"]
            }

    # ---------- 🧩 Step 2: 代码生成 ----------
    async def generate_code(self, problem_analysis: Dict[str, Any], data_info: Dict[str, Any]) -> str:
        """生成时间序列预测任务完整代码"""
        self._log("开始生成代码...")

        system_prompt = """你是Kaggle时间序列专家。
请编写完整的Python代码，执行以下任务：

✅ 功能要求
1. 加载train.csv、test.csv及辅助数据（如oil.csv、holidays_events.csv等）
2. 自动识别日期列（date/time/timestamp）
3. 构造日期特征（year, month, day, dayofweek, weekofyear, quarter）
4. 自动合并辅助数据（oil, holidays, stores, transactions）
5. 自动编码object列（pd.get_dummies）
6. 确保train/test特征完全对齐
7. 使用sklearn的RandomForestRegressor训练（不要使用lightgbm/xgboost）
8. 生成submission.csv（两列：id, 预测值）

⚠️ 重要规则：
- 只使用train和test都有的列来构造特征
- 不要在test数据上使用目标变量（target/y/sales等）
- 如果要创建基于目标变量的特征（如滞后、统计特征），只能用于train，test需跳过或用其他方法填充
- **列对齐时使用join='inner'而不是join='left'，确保只保留共同列**
- **目标变量必须从原始训练数据中单独获取，不要从对齐后的数据中获取**
- **预测前确保测试数据不包含目标变量列**

🧱 代码结构
- 导入库（pandas, numpy, sklearn）
- 数据加载（带try-except）
- 特征工程（日期特征 + 合并辅助数据）
- 编码object列与列对齐
- 模型训练与验证（输出RMSE）
- 预测与提交文件生成

📦 注意
- **只使用sklearn自带的模型**（RandomForestRegressor, GradientBoostingRegressor等）
- **不要导入lightgbm或xgboost**（环境中可能没有）
- 对缺失值使用 fillna(0)
- 对训练集过大的情况（>200000行）采样20%
- 所有步骤必须加print日志
- 保证即使部分辅助数据不存在，代码仍能运行
- submission.csv保存到 output_dir

只输出完整Python代码，不要任何解释。"""

        # 拼接任务提示
        file_summary = "\n".join(
            [f"- {k}: {v.get('columns', [])}" for k, v in data_info.get("all_files_info", {}).items()]
        )
        prompt = f"""请为以下任务生成完整代码：

任务描述：
{problem_analysis.get('problem_description', '')}

问题类型：
{problem_analysis.get('problem_type', 'time_series')}

数据目录：
- 输入路径: {self.config.data_path}
- 输出路径: {self.config.output_dir}

检测到文件：
{file_summary}
"""

        try:
            response = self.llm.generate(prompt, system_prompt, temperature=0.3, max_tokens=4000)
            self.result.llm_calls += 1

            code = self._clean_code(response.content)
            self._log(f"✓ 代码生成完成（{len(code)}字符）")
            return code

        except Exception as e:
            self._log(f"生成代码失败: {e}", level="error")
            raise

    # ---------- ⚙️ Step 3: 执行代码 ----------
    async def execute_code(self, code: str) -> Dict[str, Any]:
        """执行生成的代码"""
        self._log("开始执行代码...")

        try:
            result = self.executor.execute(code, working_dir=self.config.data_path, output_dir=self.config.output_dir)
            self.result.observations.append(f"执行结果: {'成功' if result.success else '失败'}")

            if result.success:
                self._log("✓ 代码执行成功")
                return {
                    "success": True,
                    "output": result.output,
                    "submission_path": result.submission_path,
                    "error": None
                }
            else:
                self._log(f"❌ 执行失败: {result.error}", level="error")
                return {
                    "success": False,
                    "output": result.output,
                    "submission_path": None,
                    "error": result.error
                }

        except Exception as e:
            self._log(f"执行异常: {e}", level="error")
            return {"success": False, "output": "", "error": str(e), "submission_path": None}

    # ---------- 🛠 Step 4: 修复代码 ----------
    async def fix_code(self, failed_code: str, error_message: str, data_info: Dict[str, Any] = None) -> str:
        """自动修复失败代码"""
        self._log("开始修复代码...")

        system_prompt = """你是Kaggle时间序列修复专家。
请根据错误信息修复以下代码，使其成功运行并生成submission.csv。

修复原则：
1. 修复所有object列编码问题（用pd.get_dummies）
2. 修复merge键不匹配、缺失值或列不对齐问题
3. 保证train/test列一致
4. 保证生成submission.csv
5. 所有print语句保持，结构不变
6. 仅输出修复后的Python代码
"""

        # 构建数据结构信息
        data_structure_info = ""
        if data_info and 'all_files_info' in data_info:
            data_structure_info = "\n## 数据结构参考\n"
            for fn, meta in data_info['all_files_info'].items():
                dtypes_str = ", ".join([f"{k}({v})" for k, v in list(meta.get('dtypes', {}).items())[:8]])
                data_structure_info += f"- {fn}: {dtypes_str}\n"

        prompt = f"""以下代码运行失败，请修复：

## 错误信息
```
{error_message[:1500]}
```

{data_structure_info}

## 失败的代码
```python
{failed_code}
```

常见错误与修复：
- ValueError: could not convert string to float → object列未编码，需pd.get_dummies
- KeyError: 列不存在 → get_dummies后列名已改变
- 列不对齐 → 使用reindex或concat补齐

请生成修复后的完整Python代码："""

        try:
            response = self.llm.generate(prompt, system_prompt, temperature=0.2, max_tokens=4000)
            self.result.llm_calls += 1
            
            fixed_code = self._clean_code(response.content)
            self.result.thoughts.append("代码已修复")
            self._log("✓ 代码修复完成")
            return fixed_code
            
        except Exception as e:
            self._log(f"修复失败: {e}", level="error")
            raise

    # ---------- 📊 Step 5: 评估指标 ----------
    def get_metrics(self) -> Dict[str, Any]:
        """获取评估指标"""
        return {
            "agent_type": "react",
            "total_time": self.result.total_time,
            "code_generation_time": self.result.code_generation_time,
            "execution_time": self.result.execution_time,
            "llm_calls": self.result.llm_calls,
            "code_lines": self.result.code_lines,
            "thoughts_count": len(self.result.thoughts),
            "actions_count": len(self.result.actions),
            "success": self.result.status.value == "completed"
        }

    # ---------- 辅助方法 ----------
    def _extract_problem_type(self, content: str) -> str:
        """从LLM响应提取问题类型"""
        content_lower = content.lower()
        if "time series" in content_lower or "forecasting" in content_lower:
            return "time_series_forecasting"
        elif "classification" in content_lower:
            return "classification"
        elif "regression" in content_lower:
            return "regression"
        return "unknown"

    def _extract_insights(self, content: str) -> list:
        """从LLM响应提取关键洞察"""
        sentences = content.split('.')[:3]
        return [s.strip() for s in sentences if s.strip()]

    def _clean_code(self, code: str) -> str:
        """清理生成的代码（移除markdown标记）"""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        return code.strip()
