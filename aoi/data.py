"""AOI 检测数据的加载、质量校验与清洗衍生模块。

在制造业质量工程（MQE）中，数据清洗不仅是技术处理，更是确立“统计口径”的基石：
1. 原始数据往往包含因重试、复测导致的同一工件多条记录；
2. 质量统计（如良品误报率 FPR、直通率 FPY）必须以每个产品的“首检（First-pass）”为独立样本单位；
3. 将业务上的 AOI 判定与金标准（复检基准 reference_result）对齐，派生真阳性（TP）、假阳性（FP，即误报）等布尔标记。
"""

from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """加载原始 AOI 检测 CSV 数据并规范化时间列。

    使用 errors='coerce' 允许容忍并捕获格式损坏的时间戳，
    便于后续 quality_report 进行数据质量审计。
    """
    df = pd.read_csv(path)
    df["inspection_time"] = pd.to_datetime(df["inspection_time"], errors="coerce")
    return df


def quality_report(raw: pd.DataFrame) -> pd.Series:
    """生成原始数据质量健康度诊断报告。

    统计维度涵盖：
    - 数据脏污情况：无法解析的时间戳、重复的检测流水号、完全重复行；
    - 实体完整性：独立产品数及存在多次复测的产品数量；
    - 关键特征缺失：人工金标准缺失、光学成像特征（亮度/对比度）缺失。
    """
    by_inspection = raw.drop_duplicates("inspection_id")
    inspections_per_product = by_inspection.groupby("product_id").size()
    return pd.Series(
        {
            "rows": len(raw),
            "unparseable inspection_time": int(raw["inspection_time"].isna().sum()),
            "duplicate inspection_id rows (dirty)": int(raw.duplicated("inspection_id").sum()),
            "fully duplicated rows": int(raw.duplicated().sum()),
            "distinct products": int(raw["product_id"].nunique()),
            "products inspected more than once": int((inspections_per_product > 1).sum()),
            "missing reference_result": int(raw["reference_result"].isna().sum()),
            "missing brightness": int(raw["brightness"].isna().sum()),
            "missing contrast": int(raw["contrast"].isna().sum()),
        },
        name="count",
    )


def retest_first_results(raw: pd.DataFrame) -> pd.Series:
    """统计被多次检测产品的“首检结果”分布。

    在标准制造流程中，复测逻辑应当是：“首检报警（NG）才允许送二次复测；首检良品（OK）直接流转”。
    此函数用于验证真实数据流是否符合该生产调度预期，排除无故重复打光的异常数据。
    """
    ordered = (
        raw.drop_duplicates("inspection_id")
        .sort_values(["inspection_time", "inspection_id"])
        .copy()
    )
    ordered["seq"] = ordered.groupby("product_id").cumcount() + 1
    multi = ordered[ordered.groupby("product_id")["inspection_id"].transform("size") > 1]
    return multi[multi["seq"] == 1]["aoi_result"].value_counts()


def build_clean(raw: pd.DataFrame) -> pd.DataFrame:
    """数据清洗与统计口径字段派生。

    处理步骤：
    1. 去重与时序排序：按检测 ID 去重，按时间与 ID 正序排列；
    2. 时间维度离散化：派生自然日（date）与小时（hour）；
    3. 工件生命周期标记：标记检测序号（inspection_seq）与首检标识（is_first_pass）；
    4. 质量统计口径判定（基于首检）：
       - is_alarm: AOI 报警（机检判 NG）；
       - is_actual_defect: 金标准真实不良（人工/显微镜复检判 NG）；
       - is_actual_good: 金标准真实良品（人工复检判 OK）；
       - is_true_positive (TP): 抓出真缺陷（机检 NG 且 真 NG）；
       - is_false_positive (FP): 良品误报（机检 NG 但 真 OK）；
       - is_miss (FN): 缺陷漏检（机检 OK 但 真 NG）。
    """
    clean = (
        raw.drop_duplicates("inspection_id")
        .sort_values(["inspection_time", "inspection_id"])
        .reset_index(drop=True)
    )
    clean["date"] = clean["inspection_time"].dt.normalize()
    clean["hour"] = clean["inspection_time"].dt.hour
    clean["inspection_seq"] = clean.groupby("product_id").cumcount() + 1
    clean["is_first_pass"] = clean["inspection_seq"] == 1
    clean["has_reference"] = clean["reference_result"].isin(["OK", "NG"])
    clean["is_alarm"] = clean["aoi_result"] == "NG"
    clean["is_actual_defect"] = clean["reference_result"] == "NG"
    clean["is_actual_good"] = clean["reference_result"] == "OK"
    clean["is_true_positive"] = clean["is_alarm"] & clean["is_actual_defect"]
    clean["is_false_positive"] = clean["is_alarm"] & clean["is_actual_good"]
    clean["is_miss"] = ~clean["is_alarm"] & clean["is_actual_defect"]
    return clean


def load_clean(path: Path) -> pd.DataFrame:
    """从磁盘加载已清洗规范化的数据文件，并恢复时间序列类型。"""
    return pd.read_csv(path, parse_dates=["inspection_time", "date"])


def first_pass(clean: pd.DataFrame) -> pd.DataFrame:
    """过滤出所有工件的“首检（First Pass）”记录。

    保证每个 product_id 仅贡献 1 条样本，消除复测导致的重复计数与指标稀释。
    """
    return clean[clean["is_first_pass"]]


def with_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """过滤出包含明确基准复检结果（Ground Truth 为 OK 或 NG）的记录。

    计算 FPR、Recall、Precision 等监督质量指标时必须以此子集为基准。
    """
    return frame[frame["has_reference"]]
