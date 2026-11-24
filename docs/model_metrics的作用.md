在本项目（**AOI 检测数据分析与误报复判练习 / Apple MQE 面试准备项目**）中，`data/processed/model_metrics.json` 的**作用**是：**作为第 3 天（Day 3）机器学习复判实验的核心指标契约与单一事实来源（Single Source of Truth, SSOT）**。

它将复杂的离线 Notebook 实验结果（包含模型选型、阈值策略、业务减负收益、首要质量红线、统计不确定性与数据泄漏检验）解耦并持久化为结构化 JSON，一方面为项目沉淀标准数据资产，另一方面为后续面试答辩、报告撰写与工程决策提供直接的量化证据支持。

---

### 一、 数据流与架构定位

在项目的端到端流水线中，它的上下游关系如下：

1. **产生端（Upstream）**：
   由 `notebooks/aoi/03_ml_experiment.ipynb` 训练并评估复判模型后自动计算并落盘：

````python
# 1513:1520:./notebooks/aoi/03_ml_experiment.ipynb
"leakage_demo_pr_auc_mean": {k: float(v) for k, v in leak_table["PR-AUC (mean)"].items()},
}
(ROOT / "data" / "processed" / "model_metrics.json").write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False)
)
display(Markdown(f"```json\n{json.dumps(metrics, indent=2, ensure_ascii=False)}\n```"))
````

2. **消费端（Downstream）**：
   - **面试答辩与报告（`reports/interview_notes.md`）**：在“3 分钟项目介绍”、“机器学习模拟题（Q4-Q7）”及“速记卡”中引用的所有模型量化指标（如人工减负 33.3%、召回 98.5%、置信区间等），全部以该文件的数据为基准事实。
   - **工程契约解耦**：避免上层报告或应用直接依赖体量庞大且运行缓慢的 `.ipynb` 运行时，使得模型评估产物可独立被版本控制（Git）、自动化测试或 CI 质检流水线读取校验。

---

### 二、 指标内容与 MQE（制造质量工程师）视角拆解

查看该文件的完整内容：

```1:34:/Users/Phil/Workspace/Study/MQE/apple-mqe-interview/data/processed/model_metrics.json
{
  "model": "LogisticRegression",
  "features": [
    "aoi_score",
    "brightness",
    "contrast",
    "defect_type"
  ],
  "target_recall_on_valid": 0.98,
  "chosen_threshold": 0.05,
  "validation_pr_auc": 0.8999790352831272,
  "validation_pr_auc_aoi_score_only": 0.8826381778647818,
  "test": {
    "alarms": 835,
    "to_review": 557,
    "auto_released": 278,
    "workload_reduction": 0.332934131736527,
    "defects": 133,
    "defects_missed": 2,
    "recall": 0.9849624060150376,
    "recall_ci95": [
      0.9467356013821775,
      0.9981736611688586
    ],
    "false_positives": 702,
    "false_positives_released": 276,
    "end_to_end_defect_retention": 0.9528301886792453,
    "aoi_recall_first_pass": 0.9716981132075472
  },
  "leakage_demo_pr_auc_mean": {
    "Random split by row (leaky)": 0.8715180611263172,
    "Random split grouped by product_id": 0.8543525836431837
  }
}
```

这些字段在 Apple MQE 语境下承载着关键的业务与质量逻辑：

#### 1. 业务减负价值证明（ROI）

- **`workload_reduction: 0.3329` 与 `auto_released: 278`**：
  在测试集（835 件报警）中，模型成功自动放行 278 件误报，将人工复判件数压降至 557 件，**直接降低人工复判工作量 33.3%**。这是该模型在产线最直接的经济与人力价值。

#### 2. 首要质量红线控制（Quality Guardrail）

- **`target_recall_on_valid: 0.98` 与 `chosen_threshold: 0.05`**：
  在质量工程中，阈值绝不能按默认的 0.5 盲目切分，而必须以“漏检成本极高”为约束在独立验证集上按 Recall $\ge 98\%$ 保守推导出来（阈值压低到 0.05），再拿到独立测试集进行盲测。
- **`recall: 0.985` 与 `defects_missed: 2`**：
  在测试集的 133 件报警真缺陷中，模型保留了 131 件，仅漏放 2 件，实现了 98.5% 的缺陷召回。

#### 3. 端到端系统级质量视角（Apple MQE 核心考察点）

- **`end_to_end_defect_retention: 0.9528` vs `aoi_recall_first_pass: 0.9717`**：
  **这是整个 JSON 中最体现质量工程专业性的一组数据**。AOI 本身首检就有漏检（召回率只有 97.2%），复判模型虽然对报警件有 98.5% 的召回，但又额外漏放了 2 件，导致从产品上线到最终流出，端到端真实缺陷拦截率降到了 95.3%。它提醒工程师：**复判模型不能脱离 AOI 设备的初检漏检单独谈高准确率，两道防线的风险是乘积叠加的**。

#### 4. 小样本统计严谨性

- **`recall_ci95: [0.9467, 0.9982]`**：
  测试集真实缺陷仅 133 件，虽然点估计召回率为 98.5%，但 95% 置信区间跨度宽达 94.7%～99.8%。该指标直接证明了“在缺陷样本量有限时，不能把单个点估计当成量产放行依据”。

#### 5. 工程诚实性与基线对比（拒绝盲目追求复杂模型）

- **`validation_pr_auc: 0.89998` vs `validation_pr_auc_aoi_score_only: 0.88264`**：
  模型引入亮度和对比度特征后，PR-AUC 相比“仅用 AOI 原生分数”的朴素基线仅提升了 +0.017。这一量化对比支撑了项目中“**排查设备光源衰减与批次反光的根因价值远大于训练分类模型**”的实战定论。

#### 6. 产线复测与数据泄漏实证

- **`leakage_demo_pr_auc_mean`**：
  同一产品多次复测记录如果跨训练/测试集切分（按行随机切分 PR-AUC 0.8715），会比按 `product_id` 分组严格切分（0.8544）高估约 0.017。该实验指标展示了对制造产线“报警复测机制”特有数据结构的严谨控制。

---

### 三、 总结

`data/processed/model_metrics.json` 是整个项目中**把“算法实验”转化为“制造质量工程说服力”的核心桥梁**。它不是一个孤立的测试结果文件，而是将**业务收益（人工减负）、质量红线（召回率与端到端风险）、统计严谨性（置信区间）以及工程诚实性（基线对比与防泄漏）**全部锁死的数据锚点。
