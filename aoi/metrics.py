"""质量工程（MQE）核心质量指标与统计学评估模块。

主要解决工业视觉检测中的三大核心认知陷阱：
1. 【混淆矩阵定义】：
   - 正类（Positive） = 真实缺陷件（NG）；
   - 负类（Negative） = 真实良品件（OK）；
   - 报警 = 机器判定为 Positive；放行 = 机器判定为 Negative。
2. 【准确率（Accuracy）陷阱】：
   在制造业极高良率（如 99%+）背景下，“全部判 OK”的无脑策略也能获得 99% 的准确率，
   但缺陷漏检率（Miss Rate）高达 100%，这在工业制造中是致命灾难。
3. 【误报率的分母陷阱】：
   工程口径必须严格区分三类分母：
   - FPR（良品误报率） = FP / (FP + TN) （工艺能力指标：良品被误伤的概率）；
   - 报警误报占比 = FP / (TP + FP) = 1 - Precision （复核人因指标：复检员看到多少假报警）；
   - 误报占全检比例 = FP / Total。
4. 【贝叶斯基底效应】：
   即使 AOI 硬件光学与算法能力（Recall 和 FPR）固定不变，当来料制程良率提高（缺陷率下降）时，
   报警的查准率（Precision）会急剧恶化，复核人员会直观感觉“误报变多了”。
"""

import numpy as np
import pandas as pd

COUNT_COLUMNS = ["TP", "FP", "FN", "TN", "Manual review (all alarms)"]
RATE_COLUMNS = ["Precision", "Recall", "FPR", "Miss rate", "Accuracy"]


def _ratio(a: float, b: float) -> float:
    """带零除保护的安全比例计算，分母为 0 时返回 NaN。"""
    return a / b if b else np.nan


def quality_metrics(tn: int, fp: int, fn: int, tp: int) -> dict:
    """根据二分类混淆矩阵计算标准质量工程评估指标。

    参数说明：
    - tn (True Negative):  真实良品被 AOI 正确放行
    - fp (False Positive): 真实良品被 AOI 误报警（误报）
    - fn (False Negative): 真实缺陷被 AOI 漏检放行（漏检）
    - tp (True Positive):  真实缺陷被 AOI 正确报警检出
    """
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": _ratio(tp, tp + fp),
        "Recall": _ratio(tp, tp + fn),
        "FPR": _ratio(fp, fp + tn),
        "Miss rate": _ratio(fn, tp + fn),
        "Accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "Manual review (all alarms)": tp + fp,
    }


def trivial_baselines(tn: int, fp: int, fn: int, tp: int) -> pd.DataFrame:
    """构建两个极端“无信息基线”，量化解释为什么在不平衡数据下不能依赖 Accuracy。

    对比基线：
    - Predict everything OK: 机器完全瘫痪全放行，Accuracy 极高但 Recall=0、漏检率=100%；
    - Predict everything NG: 机器全量报警送检，Recall=100% 但人工复检负荷爆炸。
    """
    positives, negatives = tp + fn, tn + fp
    scenarios = {
        "AOI (as above)": (tn, fp, fn, tp),
        "Predict everything OK": (negatives, 0, positives, 0),
        "Predict everything NG": (0, negatives, 0, positives),
    }
    table = pd.DataFrame({name: quality_metrics(*cells) for name, cells in scenarios.items()}).T
    table[COUNT_COLUMNS] = table[COUNT_COLUMNS].astype(int)
    return table


def false_call_denominators(tn: int, fp: int, fn: int, tp: int) -> pd.DataFrame:
    """对比误报率的三种不同分母口径，避免跨部门沟通时的概念混淆。

    - FPR (良品误报率): 评估检测算法/机台本身的光学灵敏度；
    - 报警中误报占比: 评估后道复检线的人力开销与复核效率；
    - 误报占全部检测数: 评估整条产线因误报受影响的总吞吐比例。
    """
    total = tn + fp + fn + tp
    return pd.DataFrame(
        {
            "Name": [
                "FPR（良品误报率）",
                "报警中的误报占比（1 - Precision）",
                "误报占全部检测数",
            ],
            "Formula": ["FP / (FP + TN)", "FP / (TP + FP)", "FP / Total"],
            "Value": [_ratio(fp, fp + tn), _ratio(fp, tp + fp), _ratio(fp, total)],
        }
    )


def precision_by_defect_rate(recall: float, fpr: float, defect_rates) -> pd.DataFrame:
    """模拟在 AOI 算法性能（Recall 与 FPR）恒定时，Precision 随来料真实缺陷率的演变趋势。

    依据贝叶斯公式：
    P(Defect|Alarm) = Recall * p / [Recall * p + FPR * (1 - p)]
    其中 p 为真实缺陷率。该分析证明：良率越高，报警中真缺陷的比例反而越低。
    """
    rates = np.asarray(defect_rates, dtype=float)
    true_alarms = recall * rates
    precision = true_alarms / (true_alarms + fpr * (1 - rates))
    return pd.DataFrame({"Defect rate": rates, "Precision": precision})


def rollup(frame: pd.DataFrame, by) -> pd.DataFrame:
    """按指定维度（设备、日期、批次、班次等）汇总聚合误报与检出指标。

    要求输入 frame 必须预先过滤为“有基准复判（has_reference）”的样本。
    """
    out = frame.groupby(by, observed=True).agg(
        inspections=("inspection_id", "size"),
        alarms=("is_alarm", "sum"),
        good=("is_actual_good", "sum"),
        defects=("is_actual_defect", "sum"),
        tp=("is_true_positive", "sum"),
        fp=("is_false_positive", "sum"),
        fn=("is_miss", "sum"),
    )
    out["alarm_rate"] = out["alarms"] / out["inspections"]
    out["fpr"] = out["fp"] / out["good"].replace(0, np.nan)
    out["recall"] = out["tp"] / out["defects"].replace(0, np.nan)
    out["fp_share_of_alarms"] = out["fp"] / out["alarms"].replace(0, np.nan)
    return out.reset_index()


def overall_summary(verified_first: pd.DataFrame) -> dict:
    """计算“首检且有基准判定”口径下的全线全局质量与误报基准指标字典。"""
    row = rollup(verified_first.assign(scope="overall"), "scope").iloc[0]
    return {
        "inspections": int(row["inspections"]),
        "alarms": int(row["alarms"]),
        "good": int(row["good"]),
        "defects": int(row["defects"]),
        "tp": int(row["tp"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "fpr": float(row["fpr"]),
        "recall": float(row["recall"]),
        "precision": float(row["tp"] / row["alarms"]),
        "fp_share_of_alarms": float(row["fp_share_of_alarms"]),
    }
