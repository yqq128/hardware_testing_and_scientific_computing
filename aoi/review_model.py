"""AOI 误报二次复判机器学习模型流水线。

在工业视觉质检中，AOI 硬件机台通常保持极高灵敏度以防漏检，随之产生大量良品误报（False Positive），
造成后道人工显微镜复检线（Review Station）严重的工时与疲劳瓶颈。

本项目构建“二次智能复判模型”，对 AOI 报警件进行二次判定：
- 预测目标 y: 1 = 真实缺陷（送人工复检），0 = 误报（自动放行）；
- 核心业务底线：【严控漏放率】——在验证集上严格约束 Recall >= 98%，在该红线内最大化误报自动放行比例（削减人力）；
- 核心算法防线：【杜绝数据泄漏】——工业品存在同件多次复测且跨午夜，必须以产品首检时间（product_first_date）
  按时间序列划分数据集，严禁按记录行随机切分；
- 统计可信度度量：使用 Exact Clopper-Pearson 二项置信区间评估测试集召回率，并监测协变量漂移（反光批次）。
"""

import json
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
# 时间切分点：早于 10-21 为训练集，10-21~10-25 为验证集，10-26 及以后为最终留出测试集
VALID_START = pd.Timestamp("2026-10-21")
TEST_START = pd.Timestamp("2026-10-26")
# 业务硬约束：复检模型在验证集上对真实缺陷的召回率不得低于 98%
TARGET_RECALL = 0.98
# 已知的异常反光批次线别与日期（用于概念漂移评估）
GLARE_LINE, GLARE_DAY = "L03", pd.Timestamp("2026-10-27")
N_REPEATS = 8

NUMERIC = ["aoi_score", "brightness", "contrast"]
CATEGORICAL = ["defect_type"]
FEATURES = NUMERIC + CATEGORICAL

AOI_ONLY = "AOI score only (no model)"
THRESHOLD_GRID = np.round(np.arange(0.01, 0.99, 0.01), 2)

SWEEP_FORMAT = {"recall": "{:.1%}", "workload_reduction": "{:.1%}"}


def with_product_first_date(clean: pd.DataFrame) -> pd.DataFrame:
    """计算每个工件的首检日期，杜绝时间切分时的数据泄漏。

    若某工件在 23:58 首检报警并在次日 00:05 复测，如果按单条记录的自然日切分，
    同一工件的信息会同时出现在训练集和验证集。因此必须以产品首检日期统一归属。
    """
    first_date = clean.groupby("product_id")["inspection_time"].transform("min").dt.normalize()
    return clean.assign(product_first_date=first_date)


def build_alarm_dataset(clean_with_first_date: pd.DataFrame) -> pd.DataFrame:
    """构建二次复判建模样本集。

    样本空间限制：仅取【AOI 报警且有人工基准结果】的工件。
    未报警的工件已在机台直接放行，不进入复判阶段。
    二分类目标 y：1 代表真实缺陷，0 代表误报。
    """
    frame = clean_with_first_date
    alarms = frame[frame["is_alarm"] & frame["has_reference"]].copy()
    alarms["y"] = alarms["is_actual_defect"].astype(int)
    return alarms


@dataclass(frozen=True)
class Splits:
    """按时间划分的训练集、验证集与测试集（不可变数据容器）。"""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    @property
    def pool(self) -> pd.DataFrame:
        """训练集与验证集的合并池（用于演示数据泄漏实验，保持测试集完全隔离）。"""
        return pd.concat([self.train, self.valid]).reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """生成各数据集的时间跨度、报警件量及真实缺陷比例摘要。"""
        return pd.DataFrame(
            {
                name: {
                    "period": f"{frame['date'].min():%m-%d} ~ {frame['date'].max():%m-%d}",
                    "alarms": len(frame),
                    "true defects": int(frame["y"].sum()),
                    "defect share": f"{frame['y'].mean():.1%}",
                }
                for name, frame in {
                    "train": self.train,
                    "valid": self.valid,
                    "test": self.test,
                }.items()
            }
        ).T


def split_by_time(alarms: pd.DataFrame) -> Splits:
    """严格基于产品首检日期进行时间切分，并强制断言无跨集产品泄漏。"""
    split_date = alarms["product_first_date"]
    train = alarms[split_date < VALID_START]
    valid = alarms[(split_date >= VALID_START) & (split_date < TEST_START)]
    test = alarms[split_date >= TEST_START]

    ids = [set(train["product_id"]), set(valid["product_id"]), set(test["product_id"])]
    overlap = (ids[0] & ids[1]) | (ids[0] & ids[2]) | (ids[1] & ids[2])
    if overlap:
        raise ValueError(f"同一产品出现在多个集合：{sorted(overlap)[:5]}")
    return Splits(train, valid, test)


def build_pipeline(model) -> Pipeline:
    """构建特征工程与分类模型流水线。

    数值特征：中位数插补缺失值（鲁棒抗噪）+ Z-score 标准化；
    类别特征（报出的缺陷类型 defect_type）：独热编码，忽略未知类别。
    """
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def candidate_models() -> dict:
    """定义候选复判算法架构（线性模型 vs 非线性集成树模型）。

    两者均显式配置 class_weight 类别平衡，防止因缺陷样本稀缺导致模型偏向全判误报。
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def sweep(y: np.ndarray, proba: np.ndarray, thresholds) -> pd.DataFrame:
    """在 0.01 到 0.99 全范围扫描分类阈值，量化业务得失。

    针对每个决策阈值 t（概率 >= t 判定为真实缺陷送人工）：
    - 缺陷捕获数（defects_caught）与漏放数（defects_missed）；
    - 真实缺陷召回率（Recall）与人工工作量削减比例（Workload Reduction）；
    - 被成功自动放行的误报良品数（false_positives_released）。
    """
    rows = []
    for t in thresholds:
        review = proba >= t
        caught = int((review & (y == 1)).sum())
        missed = int((~review & (y == 1)).sum())
        rows.append(
            {
                "threshold": t,
                "to_review": int(review.sum()),
                "auto_released": int((~review).sum()),
                "defects_caught": caught,
                "defects_missed": missed,
                "recall": caught / max(caught + missed, 1),
                "workload_reduction": 1 - review.mean(),
                "false_positives_released": int((~review & (y == 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(valid_sweep: pd.DataFrame, target_recall: float = TARGET_RECALL) -> float:
    """在验证集上选择满足业务召回率红线的最大操作阈值。

    阈值越高，判定为误报并自动放行的工件越多（人工复检越轻松）；
    但阈值不能无限抬高导致漏放真实缺陷。
    此步骤严格仅基于验证集搜索最佳阈值，严禁使用测试集调参。
    """
    eligible = valid_sweep[valid_sweep["recall"] >= target_recall]
    if eligible.empty:
        raise ValueError(f"验证集上没有任何阈值能达到 Recall >= {target_recall:.0%}")
    return float(eligible["threshold"].max())


def leakage_demo(pool: pd.DataFrame, n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """定量对比【逐行随机划分】与【按产品分组划分】的模型表现，实证数据泄漏危害。

    复测工件具有极高相似度，若同一工件首检在训练集、复测在测试集，
    模型会死记工件特征，导致 PR-AUC 虚高。此实验证明生产环境中 Group 划分的必要性。
    """

    def evaluate(splits) -> dict:
        rows = []
        for train_idx, test_idx in splits:
            pipe = build_pipeline(candidate_models()["RandomForest"]).fit(
                pool.iloc[train_idx][FEATURES], pool.iloc[train_idx]["y"]
            )
            proba = pipe.predict_proba(pool.iloc[test_idx][FEATURES])[:, 1]
            y = pool.iloc[test_idx]["y"]
            rows.append(
                {"ROC-AUC": roc_auc_score(y, proba), "PR-AUC": average_precision_score(y, proba)}
            )
        frame = pd.DataFrame(rows)
        return {
            "ROC-AUC (mean)": frame["ROC-AUC"].mean(),
            "PR-AUC (mean)": frame["PR-AUC"].mean(),
            "PR-AUC (std)": frame["PR-AUC"].std(),
        }

    row_splits = StratifiedShuffleSplit(
        n_splits=n_repeats, test_size=0.3, random_state=RANDOM_STATE
    ).split(pool, pool["y"])
    group_splits = GroupShuffleSplit(
        n_splits=n_repeats, test_size=0.3, random_state=RANDOM_STATE
    ).split(pool, groups=pool["product_id"])
    return pd.DataFrame(
        {
            "Random split by row (leaky)": evaluate(row_splits),
            "Random split grouped by product_id": evaluate(group_splits),
        }
    ).T


def leakage_summary_markdown(leak_table: pd.DataFrame, pool: pd.DataFrame) -> str:
    """生成数据泄漏实验结论说明文本。"""
    gap = (
        leak_table.loc["Random split by row (leaky)", "PR-AUC (mean)"]
        - leak_table.loc["Random split grouped by product_id", "PR-AUC (mean)"]
    )
    retest_share = 1 - pool["product_id"].nunique() / len(pool)
    text = f"""
    逐行随机切分的 PR-AUC 比按产品分组切分高 **{gap:+.3f}**，这部分是复测样本泄漏带来的虚高。

    **要如实解读幅度**：这里复测样本只占 {retest_share:.0%}，特征只有 3 个表格数值，模型容量也有限，
    所以虚高幅度不大。如果是图像任务（同一工件多张几乎一样的照片、重复比例更高），
    虚高会严重得多。**结论：模型评估必须按产品分组（或按时间）切分。**
    """
    return textwrap.dedent(text)


def drift_row(frame: pd.DataFrame) -> dict:
    """计算指定切片样本的误报放行率、真实缺陷漏放数及光学特征均值。"""
    fp = frame[frame["y"] == 0]
    defects = frame[frame["y"] == 1]
    return {
        "alarms": len(frame),
        "false positives": len(fp),
        "FP auto-released": fp["released"].mean(),
        "real defects": len(defects),
        "defects missed": int(defects["released"].sum()),
        "mean brightness": frame["brightness"].mean(),
        "mean contrast": frame["contrast"].mean(),
    }


def drift_table(test_eval: pd.DataFrame) -> pd.DataFrame:
    """对比反光异常批次（glare）与测试集其他样本（rest）的表现，监控协变量偏移。"""
    glare_mask = (test_eval["line_id"] == GLARE_LINE) & (test_eval["date"] == GLARE_DAY)
    return pd.DataFrame(
        {"glare": drift_row(test_eval[glare_mask]), "rest": drift_row(test_eval[~glare_mask])}
    ).T


DRIFT_LABELS = {
    "glare": f"Glare batch ({GLARE_LINE}, {GLARE_DAY:%m-%d})",
    "rest": "Rest of test period",
}
DRIFT_FORMAT = {
    "alarms": "{:.0f}",
    "false positives": "{:.0f}",
    "FP auto-released": "{:.1%}",
    "real defects": "{:.0f}",
    "defects missed": "{:.0f}",
    "mean brightness": "{:.1f}",
    "mean contrast": "{:.1f}",
}


def drift_summary_markdown(table: pd.DataFrame) -> str:
    """生成工业现场概念/协变量漂移监控的指导建议。"""
    glare, rest = table.loc["glare"], table.loc["rest"]
    text = f"""
    漂移批次的亮度偏高（{glare["mean brightness"]:.0f} vs {rest["mean brightness"]:.0f}）、
    对比度偏低（{glare["mean contrast"]:.1f} vs {rest["mean contrast"]:.1f}），和训练期分布不同。
    模型对误报的自动放行比例从 {rest["FP auto-released"]:.0%} 变成 {glare["FP auto-released"]:.0%}，
    该批次 {int(glare["real defects"])} 个真实缺陷中漏放 {int(glare["defects missed"])} 个。

    样本太少，不能据此估计漂移下的真实漏检率；但**放行比例在某个批次突然变化，本身就是必须监控的信号**。
    上线后要监控输入分布和放行比例，出现漂移时回退到全部人工复判，直到重新验证。
    """
    return textwrap.dedent(text)


@dataclass(frozen=True)
class EndToEnd:
    """端到端（AOI 硬件机台 + 二次复判模型）两级级联缺陷保留率。"""

    defects_all: int
    aoi_caught: int
    model_missed_first: int

    @property
    def aoi_missed(self) -> int:
        """AOI 第一级硬件漏网的真实缺陷数。"""
        return self.defects_all - self.aoi_caught

    @property
    def aoi_recall(self) -> float:
        """AOI 第一级硬件检出率。"""
        return self.aoi_caught / self.defects_all

    @property
    def retention(self) -> float:
        """端到端最终送达人工复检的真实缺陷保留率。

        保留率 = (AOI 第一级抓获缺陷 - 复判模型错误放行缺陷) / 全量真实缺陷。
        """
        return (self.aoi_caught - self.model_missed_first) / self.defects_all


@dataclass
class ExperimentResult:
    """模型实验全流程评估产物封装对象。"""

    splits: Splits
    baseline_table: pd.DataFrame
    best_name: str
    model: Pipeline
    valid_proba: np.ndarray
    test_proba: np.ndarray
    valid_sweep: pd.DataFrame
    test_sweep: pd.DataFrame
    chosen_threshold: float
    test_row: pd.Series
    defects_total: int
    recall_ci: tuple[float, float]
    fp_total: int
    fp_released: int
    test_eval: pd.DataFrame
    by_type: pd.DataFrame
    drift: pd.DataFrame
    end_to_end: EndToEnd
    leakage: pd.DataFrame | None = None

    @property
    def pr_auc_gain(self) -> float:
        """相比仅使用 AOI 机台原生打分（AOI-only 基线），复判模型带来的 PR-AUC 提升增益。"""
        col = "PR-AUC (avg precision)"
        return float(
            self.baseline_table.loc[self.best_name, col] - self.baseline_table.loc[AOI_ONLY, col]
        )

    def confusion_on_test(self) -> pd.DataFrame:
        """在留出测试集上，复判模型在选定阈值下的二分类混淆矩阵。"""
        test, flag = self.splits.test, self.test_proba >= self.chosen_threshold
        return pd.DataFrame(
            [
                [int((flag & (test["y"] == 1)).sum()), int((flag & (test["y"] == 0)).sum())],
                [int((~flag & (test["y"] == 1)).sum()), int((~flag & (test["y"] == 0)).sum())],
            ],
            index=["Model: send to review", "Model: auto-release"],
            columns=["Real defect", "False positive"],
        )

    def metrics_dict(self) -> dict:
        """格式化导出核心质量指标为字典（含置信区间与端到端保留率）。"""
        valid, test, row, e2e = self.splits.valid, self.splits.test, self.test_row, self.end_to_end
        metrics = {
            "model": self.best_name,
            "features": FEATURES,
            "target_recall_on_valid": TARGET_RECALL,
            "chosen_threshold": self.chosen_threshold,
            "validation_pr_auc": float(average_precision_score(valid["y"], self.valid_proba)),
            "validation_pr_auc_aoi_score_only": float(
                average_precision_score(valid["y"], valid["aoi_score"])
            ),
            "test": {
                "alarms": len(test),
                "to_review": int(row["to_review"]),
                "auto_released": int(row["auto_released"]),
                "workload_reduction": float(row["workload_reduction"]),
                "defects": self.defects_total,
                "defects_missed": int(row["defects_missed"]),
                "recall": float(row["recall"]),
                "recall_ci95": [float(self.recall_ci[0]), float(self.recall_ci[1])],
                "false_positives": self.fp_total,
                "false_positives_released": self.fp_released,
                "end_to_end_defect_retention": float(e2e.retention),
                "aoi_recall_first_pass": float(e2e.aoi_recall),
            },
        }
        if self.leakage is not None:
            metrics["leakage_demo_pr_auc_mean"] = {
                k: float(v) for k, v in self.leakage["PR-AUC (mean)"].items()
            }
        return metrics

    def metrics_json(self) -> str:
        """将评估指标导出为规范格式的 JSON 字符串。"""
        return json.dumps(self.metrics_dict(), indent=2, ensure_ascii=False)


def run_experiment(clean: pd.DataFrame, *, include_leakage_demo: bool = True) -> ExperimentResult:
    """执行完整的复判模型训练、选参、评估全流程。

    标准 ML 工业实践流程：
    1. 首检日期对齐与时间序列划分（Train / Valid / Test）；
    2. 多模型基线对比（逻辑回归 vs 随机森林 vs 仅 AOI 打分），挑选最优模型；
    3. 在验证集上根据 Recall >= 98% 严格寻找最优放行阈值；
    4. 将选定阈值一次性作用于留出测试集，计算削减比例与 Exact Clopper-Pearson 95% 置信区间；
    5. 概念漂移与端到端保留率计算；
    6. 可选的数据泄漏对照模拟。
    """
    clean = with_product_first_date(clean)
    splits = split_by_time(build_alarm_dataset(clean))
    train, valid, test = splits.train, splits.valid, splits.test

    fitted, valid_scores = {}, {}
    for name, model in candidate_models().items():
        fitted[name] = build_pipeline(model).fit(train[FEATURES], train["y"])
        valid_scores[name] = fitted[name].predict_proba(valid[FEATURES])[:, 1]
    valid_scores[AOI_ONLY] = valid["aoi_score"].to_numpy()

    baseline_table = pd.DataFrame(
        {
            name: {
                "ROC-AUC": roc_auc_score(valid["y"], score),
                "PR-AUC (avg precision)": average_precision_score(valid["y"], score),
            }
            for name, score in valid_scores.items()
        }
    ).T

    best_name = max(fitted, key=lambda n: average_precision_score(valid["y"], valid_scores[n]))
    best_model = fitted[best_name]
    valid_proba = valid_scores[best_name]
    valid_sweep = sweep(valid["y"].to_numpy(), valid_proba, THRESHOLD_GRID)
    chosen_threshold = choose_threshold(valid_sweep)

    test_proba = best_model.predict_proba(test[FEATURES])[:, 1]
    test_sweep = sweep(test["y"].to_numpy(), test_proba, THRESHOLD_GRID)
    test_row = test_sweep[test_sweep["threshold"] == chosen_threshold].iloc[0]

    defects_total = int(test_row["defects_caught"] + test_row["defects_missed"])
    ci = stats.binomtest(int(test_row["defects_caught"]), defects_total).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    fp_total = int((test["y"] == 0).sum())
    fp_released = int(test_row["false_positives_released"])

    test_eval = test.assign(proba=test_proba, released=test_proba < chosen_threshold)
    by_type = (
        test_eval[test_eval["y"] == 1]
        .groupby("defect_type")
        .agg(defects=("y", "size"), missed=("released", "sum"))
        .assign(recall=lambda t: 1 - t["missed"] / t["defects"])
    )

    test_first = test_eval[test_eval["is_first_pass"]]
    all_first_test = clean[
        clean["is_first_pass"]
        & clean["has_reference"]
        & (clean["product_first_date"] >= TEST_START)
    ]
    end_to_end = EndToEnd(
        defects_all=int(all_first_test["is_actual_defect"].sum()),
        aoi_caught=int(test_first["y"].sum()),
        model_missed_first=int(test_first[test_first["y"] == 1]["released"].sum()),
    )

    return ExperimentResult(
        splits=splits,
        baseline_table=baseline_table,
        best_name=best_name,
        model=best_model,
        valid_proba=valid_proba,
        test_proba=test_proba,
        valid_sweep=valid_sweep,
        test_sweep=test_sweep,
        chosen_threshold=chosen_threshold,
        test_row=test_row,
        defects_total=defects_total,
        recall_ci=(float(ci.low), float(ci.high)),
        fp_total=fp_total,
        fp_released=fp_released,
        test_eval=test_eval,
        by_type=by_type,
        drift=drift_table(test_eval),
        end_to_end=end_to_end,
        leakage=leakage_demo(splits.pool) if include_leakage_demo else None,
    )
