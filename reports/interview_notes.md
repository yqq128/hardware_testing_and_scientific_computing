# 面试笔记：AOI 检测数据分析与误报复判

> **使用说明**
>
> - 下文的数字全部来自本地**模拟实验**（`data/raw/aoi_inspection.csv`，由 `scripts/generate_data.py` 生成），
>   数值可由 `notebooks/aoi/` 复现。面试时必须说明这是模拟数据，**不能说成真实产线结果**。
> - 带 `[ ]` 的位置是需要你填入自己真实经历的地方，我不知道你的专业和项目，所以没有替你编。
> - 建议的练习方式：先看一遍，再**合上笔记，用自己的话说出来**，录音回听。

---

## 目录

1. 1 分钟口头解释：准确率高不代表可靠
2. 误报增加时如何排查
3. 异常分析说明（M03 周三）
4. 模型：为什么不能只看准确率，以及数据划分的限制
5. GRR、一致性、稳定性速记，以及"99.9% 准确率"的 2 分钟回答
6. 自我介绍与 3 分钟项目介绍
7. 10 道模拟题
8. 英文面试 8 题
9. 面试最后要问的问题
10. 面试速记卡
11. 被追问时的诚实回答

---

## 1. 1 分钟口头解释：准确率高不代表可靠

> 在制造里，真实缺陷通常很少。比如 10,000 件产品只有 100 件有缺陷，一台设备如果**什么都判良品**，准确率就是 99%，
> 但它一个缺陷都没检出，召回率是 0。
>
> 所以我不会只看准确率。我会先固定口径：缺陷为正类，然后看**召回率、漏检数量、误报数量**，
> 再按缺陷类型和班次、批次拆开看，因为整体指标可能掩盖个别缺陷类型的问题。
>
> 另外要确认统计口径。同样是 495 个误报，除以良品数是 5% 的良品误报率，除以报警数却是 84% 的报警误报占比。
> 现场感受到的往往是后者。

**对应实验**：`notebooks/aoi/01_quality_metrics.ipynb`，其中"全判良品"的准确率 99%、召回率 0%。

---

## 2. 误报增加时如何排查

不要一上来就说"重新训练模型"。先判断是不是数据、设备或工艺变化造成的。

| 方向 | 要问的问题 |
| --- | --- |
| 数据变化 | 所有产品都升，还是集中在某个批次、缺陷类型？**数量变了，还是只是比例变了？** |
| 设备变化 | 是否集中在某台设备、某条产线、某个班次？**同批次其他设备是否正常？** |
| 生产变化 | 近期是否换料、换供应商、调整工艺？ |
| 系统变化 | 检测规则、模型、光照、相机参数是否改过？ |

排查顺序：**确认口径与基线 → 按维度拆分 → 找到集中点 → 做对照 → 现场验证 → 再决定改什么。**

数据只能给线索。例如"M03 误报高"，不能直接说"M03 坏了"，要核对光源状态、维护记录和参数变更。

---

## 3. 异常分析说明（M03 周三）

这是第 2 天的真实分析结果（模拟数据）：

**整体（每件产品首检）**：召回率 94.9%，良品误报率 5.76%，Precision 14.6%。
也就是人工看到的报警里，约 **85% 是误报**。

**线索 1：设备。**

- M03 在 10-14（周三）良品误报率 **14.5%（49/337）**，同批次、同产线的 M04 只有 **5.7%（20/350）**，
  Fisher 检验 p ≈ 1.2e-4。
- 当天**检测数量没有增加**，是比例升高。
- 拆到 3 小时时段：误报率随时间上升，晚上 21 点后约 31%；同期 M03 平均亮度从约 125 一路降到约 101。
- 这和"光源衰减"的假设一致，**但只是线索**。下一步要核对光源寿命、维护记录和参数变更，并做现场验证。

**线索 2：批次。** 10-27 的批次 B081（L03 产线）上的**两台设备**误报率都约 13.5%，亮度偏高、对比度偏低。
单台设备高 → 倾向设备问题；同批次所有设备都高 → 倾向批次、来料或表面反光问题。

**线索 3：班次。** 夜班误报率 6.86%，高于其他班次 5.21%（统计显著），原因要现场确认（温度、操作、交接班都有可能）。

可以这样讲：

> 我先确认 M03 当天的检测数量有没有增加，然后看比例。接着按班次和时段拆分，发现异常集中在后半天，而且同批次的 M04 正常，
> 说明不太像产品问题。我再看图像特征，亮度在持续下降，和光源衰减的假设一致。但我不会直接下结论，
> 会去核对设备光源状态和维护记录，并在现场验证。

---

## 4. 模型：为什么不能只看准确率，以及数据划分的限制

### 4.1 实验设定（说清楚）

- 场景：AOI 先报警，复判模型对**报警样本**二次判断；低风险的自动放行，高风险或不确定的送人工。
- 模型：LogisticRegression，特征为 AOI 分数、亮度、对比度、AOI 报出的缺陷类型。
- 划分：**按产品首检日期的时间切分**（10-01~10-20 训练，10-21~10-25 验证，10-26~10-30 测试），
  同一产品的首检和复测一定在同一个集合。

### 4.2 结果（测试集，模拟数据）

| 项目 | 数值 |
| --- | --- |
| 测试期报警数 | 835 |
| 阈值（在验证集上按"Recall ≥ 98%"选出） | 0.05 |
| 自动放行 / 送人工 | 278 / 557 |
| **人工复判量下降** | **33.3%** |
| 报警中的真实缺陷 | 133 件，漏放 2 件 |
| Recall（报警中的真实缺陷被保留） | **98.5%，95% 置信区间 94.7%–99.8%** |
| 被自动放行的误报 | 276 / 702（39.3%） |
| **端到端缺陷保留率** | **95.3%**（AOI 本身只有 97.2%，模型额外漏放了 2 件） |

### 4.3 必须诚实说明的点

- **模型相对"只用 AOI 分数"的基线增益不大**：验证集 PR-AUC 0.900 vs 0.883（+0.017）。
  一开始我的数据里亮度和对比度没有独立信息，模型和基线完全打平；后来按物理上合理的方式让**真实缺陷带有独立于分数的对比度特征**，
  模型才有了小幅增益。这说明：**模型的价值来自"有没有分数之外的有效信息"，一定要和朴素基线比。**
- **召回置信区间很宽**：测试集真实缺陷只有 133 件，一个点估计不足以支撑"可以放行"的决定。
- **漏检风险叠加**：复判模型救不回 AOI 自己漏掉的缺陷，端到端保留率比 AOI 单独更低。
- **漂移**：测试期有一个训练期没见过的反光批次，模型对误报的自动放行比例从 36% 变成 60%，
  该批次 3 个真实缺陷里漏放了 1 个。样本太少不能估计真实漏检率，但**放行比例突然变化本身就是必须监控的信号**。
- **数据泄漏**：同一产品的复测样本若跨训练/测试集，PR-AUC 会高估约 0.017。这里幅度不大，
  因为复测只占约 16%、特征只有 3 个数值；**图像任务里同一工件有多张几乎一样的照片，高估会严重得多。**

### 4.4 为什么不能只看模型准确率

- 缺陷占比低，全判良品的准确率也很高；
- 报警样本中真实缺陷也只占约 15%，要看 Recall、漏掉的数量、人工量节省，**三者要一起看**；
- 要按缺陷类型看，严重缺陷的漏检即使很少也可能不可接受；
- 要在**独立、按时间/产品分组**的数据上评估，并看置信区间。

### 4.5 如何解释阈值

> 我会先和质量团队确认可接受的漏检风险，再在验证集上选阈值，固定后用独立测试集评估，不用测试集调阈值。
> 看不同阈值下的召回率、漏掉的缺陷数量和能节省的人工量。高风险缺陷用更保守的阈值，低置信度的保留人工复判。
> 最终阈值还要在影子运行和小范围试点中验证，并持续监控。

---

## 5. GRR、一致性、稳定性速记，以及"99.9% 准确率"的 2 分钟回答

### 5.1 速记

- **重复性**：同条件下，同一对象重复测量是否一致。**再现性**：换操作员、设备、夹具等条件后是否一致。
- **GRR**：把测量结果的变化拆成"产品本身的变化"和"测量系统的变化"。用于**连续数值**。
- **OK/NG 属性判定**：用属性一致性分析（重复判定一致率、与参考的 Kappa、漏判率/误判率），不要套 Gage R&R。
- **相关性 ≠ 一致性**：新设备始终比参考高 0.5，相关系数仍然是 1。要看偏差和差值分布，尤其是规格边界附近。
- **稳定性**：不同时间、班次、环境下结果是否保持；**可靠性**：长期运行是否频繁故障、能否恢复。
- 样本量决定你能证明多少：**零漏检要 299 个缺陷才能证明 Recall ≥ 99%（95% 置信）**；每类 75 件只能证明约 96%。

### 5.2 2 分钟回答：供应商说准确率 99.9%，你会同意量产吗？

> 我不会直接同意，先问几个问题。
>
> 第一，**准确率的分母和样本分布**是什么？如果缺陷只占 0.1%，全判良品也能达到 99.9%，这个数字几乎没有信息量。
> 我会要求按召回率、漏检数、误报数分别提供，并按缺陷类型拆开。
>
> 第二，**样品有没有覆盖真实缺陷和边界样品**？是不是供应商自己挑的样品？参考判定由谁、怎么确认的？
>
> 第三，**是不是独立验证**？训练和测试数据有没有重叠，有没有数据泄漏？我们会用自己盲测的样品池重新验证，并看置信区间：
> 比如要证明召回率不低于 99%，零漏检也需要大约 300 个缺陷样本。
>
> 第四，**各缺陷类型的漏检情况**，尤其是严重缺陷。
>
> 第五，**真实产线环境下是否稳定**：不同班次、批次、光照条件，以及节拍、追溯、维护要求。
>
> 所以我会提出验证方案：样品准备、能力验证、重复性与一致性、影子运行，通过之后再小范围试点，最后签核和持续监控。

对应文档：`reports/validation_plan.md`。

---

## 6. 自我介绍与 3 分钟项目介绍

### 6.1 1～2 分钟自我介绍（填空）

> 我的背景是 **[你的专业/领域]**，有 **[你真实的技能或项目]** 的经验。
> 最近我在练习用 Python 做数据分析，并学习自动光学检测（AOI）。
> 我对这个岗位感兴趣，是因为它把**数据分析、机器视觉和制造质量**结合在一起。

### 6.2 3 分钟项目介绍（中文）

**项目名称**：AOI 检测数据分析与误报复判练习（AOI Inspection Data Analysis and False-Call Review Practice）

> **背景**：AOI 设备可能产生很多误报，增加人工复判工作量。我做了一个本地练习项目来研究这个问题。
>
> **数据**：我生成了约 6 万条模拟检测记录，覆盖 3 条产线、6 台设备、30 天，包含检测结果、独立参考判定和图像特征。
> 我在数据里设置了几类问题，用来练习定位，这是模拟数据，不是真实产线。
>
> **分析**：我先固定口径，以每件产品的首检为统计单位，计算混淆矩阵：召回率 94.9%，良品误报率 5.8%，
> 但报警中约 85% 是误报。然后按时间、设备、班次、批次、缺陷类型拆分，
> 发现某台设备在某一天后半段误报明显升高，并且同批次的另一台设备正常，同时图像亮度在下降，
> 提示可能是光源问题；另一个批次所有设备都高，提示是批次或来料问题。
>
> **建模**：然后对 AOI 报警样本训练了一个基础分类模型做二次判断，按时间切分，并保证同一产品不跨集合。
> 在验证集选阈值、测试集评估：在召回 98.5% 的情况下，人工复判量下降约 33%。
> 但我也做了基线对比，模型相对只用 AOI 分数的提升很小，所以我认为**根因分析的价值大于模型本身**。
>
> **局限**：这是模拟数据，且每条记录都有参考判定，真实产线通常没有。测试集缺陷数量少，置信区间很宽；
> 遇到训练期没见过的批次漂移时，模型的放行比例会突然变化。真实使用前需要独立标签、设备验证和产线影子运行。

### 6.3 Project introduction (English, ~2 minutes)

> I built a local practice project on reducing false calls from an AOI system.
>
> I generated about 60,000 simulated inspection records across 3 lines, 6 machines and 30 days, with a reference label for each record.
> I injected a few known problems so I could practice root-cause analysis. It is simulated data, not a real production line.
>
> First I fixed the metric definitions and used the first inspection of each product as the unit. Recall was about 95%, the false positive rate about 5.8%, but around 85% of alarms were false calls.
> Then I broke it down by machine, shift, batch and defect type. One machine had a much higher false-call rate during the second half of one day while a peer machine on the same batch was normal, and image brightness was dropping. That pointed toward a lighting issue. Another batch was high on every machine, which pointed toward a batch or material issue.
>
> Then I trained a simple classifier to review the alarmed samples, using a time-based split and keeping the same product in one set. At a recall of 98.5% on the test set, it cut manual review workload by about a third. But compared with a simple baseline using only the AOI score, the improvement was small, so root-cause analysis was more valuable than the model.
>
> The limitations: simulated data, a small number of defects in the test set, wide confidence intervals, and model behavior shifts on batches it has not seen. Before real use I would need independent labels, equipment validation and a shadow-mode production trial.

### 6.4 不能夸大的地方

- 不要说"我完成了 Apple 产线 AOI 优化""我让真实产线误报率下降了 50%""模型可以安全地自动放行产品"；
- 要说"在模拟数据上"，并说明局限。

---

## 7. 10 道模拟题

### 第一组：AOI 与质量指标

**Q1. 什么是误报和漏检？**

> 先固定正类为缺陷。误报（False Positive）是良品被判为缺陷；漏检（False Negative）是缺陷被判为良品。
> 误报增加复判成本，漏检可能带来质量风险，所以要同时看。

**Q2. 为什么准确率不适合单独评价 AOI？**

> 缺陷占比很低，全判良品也能得到很高的准确率。要结合召回率、漏检数量、误报数量和缺陷类型，也要看业务影响。

**Q3. 如何降低误报？**

> 先确认口径和基线，再按设备、班次、批次、缺陷类型分析；先检查光学系统、工艺和标签，再考虑阈值和模型；
> 同时保持对漏检的独立验证，最后用试点确认产线效果。
> 比如我的练习项目里，真正的原因是某台设备光源衰减和某个批次反光，调阈值或训练模型都不能从根上解决。

### 第二组：机器学习

**Q4. 缺陷样本很少怎么办？**

> 先确认数据质量和缺陷定义，按缺陷类型统计样本量；考虑类别权重、合理采样和符合实际的数据增强；
> 收集边界样本和真实漏检样本；用合适的数据划分和指标；样本不足时明确不确定性，不夸大结果。

**Q5. 什么是数据泄漏？**

> 测试阶段使用了训练时不该获得的信息，导致评估结果比真实效果好。
> 例如同一工件的相似图片同时在训练和测试集、用全量数据计算预处理参数后再划分、使用生产时拿不到的特征。
> 我的练习里测过，复测样本跨集合会让 PR-AUC 高估约 0.017，图像任务里会更严重。

**Q6. 如何选择模型阈值？**

> 先确认质量风险约束，比较不同阈值下的漏检和复判量，按缺陷类型检查；在验证集选阈值，用独立测试集评估；
> 产线试运行后持续监控。**不能在测试集上调阈值。**

### 第三组：设备验证

**Q7. 什么是 GRR？**

> 用来分析测量系统的重复性和再现性，判断结果的变化来自产品还是测量系统。
> 连续数值用 Gage R&R；OK/NG 属性判定用属性一致性分析，两者方法不同。

**Q8. 新设备如何导入量产？**

> 明确目标和验收标准 → 准备覆盖真实情况的样品 → 验证检测能力、重复性和一致性 → 检查节拍、稳定性、可靠性 →
> 产线影子运行 → 验收、追溯和持续监控。

### 第四组：协作与沟通

**Q9. 算法团队认为模型很好，但现场工程师不愿用，怎么办？**

> 先了解顾虑是漏检、节拍、可解释性还是操作复杂度；共同确认评价标准和业务目标；
> 用现场样本和试运行数据验证；记录误判案例并分析原因；先小范围试点，建立反馈机制。

**Q10. 你的结论与供应商结论不一致怎么办？**

> 先对齐数据来源、样品、测试条件和指标定义，复现双方的测试，聚焦事实和差异原因；
> 与相关人员共同确认下一步验证方案，记录结论和责任人。

---

## 8. 英文面试 8 题

**1. Please introduce yourself.**

> My background is in **[your major or field]**. I have experience in **[your actual skills or projects]**.
> Recently, I have been practicing Python data analysis and learning about automated optical inspection.
> I am interested in this role because it combines data analysis, machine vision, and manufacturing quality.

**2. What is the difference between a false positive and a false negative?**

> A false positive means a good product is incorrectly classified as defective.
> A false negative means a defective product is incorrectly classified as good.
> In manufacturing both matter: false positives increase review work, while false negatives may create quality risks.

**3. Why is accuracy not enough?**

> In manufacturing, real defects may be rare. A model can achieve high accuracy by predicting almost everything as good, but it may miss many defects.
> So I also look at recall, false positives, false negatives, and performance for each defect type.

**4. How would you reduce false calls?**

> First, I would define the metrics and establish a baseline. Then I would analyze false calls by machine, shift, batch, and defect type.
> I would check image quality, equipment settings, and label consistency before changing the model.
> After that, I would compare possible solutions and validate them using independent data and a production trial.

**5. How would you evaluate an inspection machine before production?**

> I would first agree on the requirements and acceptance criteria with the quality and manufacturing teams.
> Then I would use representative samples to evaluate detection performance, repeatability, and agreement with a reference method.
> I would also check stability, cycle time, and reliability in a production trial before rollout.

**6. What is data leakage?**

> Data leakage means the test results use information that would not be available in real use, so the performance looks better than it really is.
> For example, if images of the same part appear in both training and test sets, the model can memorize the part instead of learning the defect.

**7. A supplier says their new machine is 99.9% accurate. Would you approve it for production?**

> Not directly. I would ask how accuracy was calculated and what the defect rate was in the test samples, because if defects are very rare, a machine that calls everything good can also look very accurate.
> I would ask whether the samples included real defects and borderline cases, whether the validation was independent, and how each defect type performed.
> Then I would run my own validation with blind samples, check repeatability and agreement with a reference, and run a shadow-mode trial before approval.

**8. What would you do if false calls suddenly increase on one machine?**

> I would first check whether the volume changed or only the rate. Then I would break it down by shift, batch, and defect type, and compare with other machines running the same batch.
> If only one machine is high and image brightness is shifting, I would suspect the lighting or camera and check maintenance records. I would not conclude anything from data alone; I would confirm on the line first.

---

## 9. 面试最后要问的问题

| 中文 | English |
| --- | --- |
| 这个岗位最常处理的是哪一类检测设备或缺陷问题？ | What kinds of inspection equipment and defect problems would this role handle most often? |
| 新设备导入量产时，质量工程师在验证阶段主要负责哪些部分？ | During new equipment ramp-up, which parts of validation would a quality engineer own? |
| 团队现在评价检测设备用哪些关键指标，漏检是怎么估计的？ | Which key metrics does the team use to evaluate inspection equipment, and how are escapes estimated? |
| 数据分析和机器学习在日常工作中是怎么和现场工程师协作的？ | How do data analysis and machine learning work together with line engineers day to day? |
| 入职前三个月，你希望这个岗位做出的第一个成果是什么？ | What would you like this person to deliver in the first three months? |
| 团队在检测模型上线后如何监控漂移，并决定是否回退？ | After a model goes live, how does the team monitor drift and decide whether to roll back? |

---

## 10. 面试速记卡

```text
AOI:
Camera / Lighting / Image / Model / OK-NG / Review

核心指标:
TP / FP / FN / TN
Precision / Recall / False Positive Rate / Miss Rate
先问分母和统计单位（首检 vs 全部检测）

数据分析:
Machine / Line / Batch / Shift / Defect Type / Time
先看数量，再看比例；同批次对照；数据只是线索

机器学习:
Label Quality / Class Imbalance / Data Leakage
Train-Validation-Test / Threshold / Drift
先和朴素基线比；验证集选阈值，测试集评估；看置信区间

设备验证:
Repeatability / Reproducibility
Agreement / Stability / Reliability / Cycle Time
样本量决定能证明多少（零漏检 299 件 ⇒ Recall ≥ 99%）

量产流程:
Requirements → Samples → Validation → Pilot → Monitoring
影子运行 → 小范围试点 → 回退方案

项目表达:
Problem → My Role → Action → Result → Limitation
```

---

## 11. 被追问时的诚实回答

**"模型相对基线只提升了 0.017，为什么还要用？"**

> 提升确实不大，所以我不会把它说成突破。它能在召回约 98.5% 的情况下降低约三分之一的人工量，但这个结论的置信区间很宽，
> 还没有经过产线验证。而且实验里更大的价值来自根因分析：设备光源和批次反光解决了，误报会从源头减少。
> 模型适合作为第二步，在影子运行中验证后再决定是否使用。

**"你怎么证明模型不会漏掉缺陷？"**

> 我不能证明。我只能给出测试集上的召回率和置信区间，并说明样本量有限。真实上线前还要独立验证、影子运行，
> 并对 AOI 放行的产品做独立抽检来估计整体漏检，因为复判模型救不回 AOI 自己漏掉的缺陷。

**"数据是模拟的，你学到的东西有什么价值？"**

> 模拟数据的好处是我知道真实答案，可以检查自己的分析方法能不能找到我植入的问题，也能测试指标口径、数据划分、阈值这些方法本身。
> 但它不能证明在真实产线上有效，真实数据的标签噪声、漂移和样本稀缺都会更难。

**"如果只有 AOI 报警样本被复判，你怎么估计漏检？"**

> 只用报警件算不出漏检和召回率。我会对 AOI 判 OK 的产品随机抽样，用独立方法复检，
> 用置信区间给出漏检率上界。不能用"没有客诉"当作没有漏检。
