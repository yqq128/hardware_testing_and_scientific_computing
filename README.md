# 硬件测试与通信工程练习

仓库包含两条硬件测试面试练习和一条通信方向作品练习。每个方向有独立 ADR 和 Notebook 目录，编号从该方向的 `01` 开始。

> 项目使用模拟数据和理想化信道，所有结果仅用于练习，不代表 Apple、华为或真实设备的测量数据。

Python 初学者可直接从[四周学习计划](docs/学习计划.md)的八行 CSV 练习开始。按最近的面试或申请节点，每周推进一项实操。带教准备和选材依据见[项目评估与调整](docs/项目评估与调整.md)。

| 方向 | 练习主题 | ADR | Notebook | 运行方式 |
| --- | --- | --- | --- | --- |
| AOI | Apple MQE：AOI 误报、复判模型与设备验证 | [五天练习大纲](docs/adr-aoi.md) | [`notebooks/aoi/`](notebooks/aoi/) | [数据、Python 与看板入口](#aoiapple-mqe) |
| 芯片测试 | 华为芯片测试工程：ATE 日志、良率归因、PAT 与测试效率 | [芯片测试练习](docs/adr-chiptest.md) | [`notebooks/chiptest/`](notebooks/chiptest/) | [数据、Python 与看板入口](#chiptest华为-芯片测试工程) |
| QPSK | 通信工程：I/Q 星座、AWGN、硬判决与 BER | [从比特到曲线的完整链路仿真](docs/adr-qpsk.md) | [QPSK 实验](notebooks/qpsk/01_qpsk_link_ber.ipynb) | [Notebook 与 Python 入口](#qpsk通信工程) |

AOI 和芯片测试可以对照练习同批次对照、复测口径和质量指标。

| aoi（Apple MQE） | chiptest（华为 芯片测试） |
| --- | --- |
| 误报率只算每件产品的首检 | FPY 只算每颗芯片的首测，复测后的良率单独报告 |
| 同批次内对照设备（M03 vs M04） | 同 lot 内对照测试机与 site |
| 批次 B081 反光，所有设备一起上升 | L07 整批工艺偏移，各台测试机都有 |
| 复判模型在 Recall 约束下选阈值 | PAT 阈值 k 在精度与召回之间取舍 |
| 全量参考判定只在模拟数据里有 | 过杀 / 漏测只在模拟数据里有 truth |

## AOI 与芯片测试的两种实现

同一套分析有两种实现方式，**都保留**，产出的结果一致：

| | Notebook 版 | Python 版 |
| --- | --- | --- |
| 适合 | 逐步学习、讲解思路、面试时展示推导过程 | 一键复现、做成可交互看板、被测试和复用 |
| 代码 | 每个 Notebook 自带内联代码，可以独立阅读，仓库里的已执行、带输出 | 同样的逻辑抽成模块函数，没有重复的绘图 / 统计代码 |
| 怎么跑 | `jupyter lab`，或 `jupyter nbconvert --execute` | `uv run python -m <方向>.pipeline` |
| 执行顺序 | 有依赖时后一个读取前一个写出的文件（AOI 02 → 03，芯片测试 01 → 02 → 03） | 一条流水线完成全部步骤 |

**怎么保证两个版本一致**：AOI 与芯片测试各有一个一致性测试，会在临时目录里**真实执行** Notebook，
再与 Python 版流水线的产物比较。改动任何一边，另一边的差异都会被测试发现。

| 方向 | 一致性测试 | 比较什么 |
| --- | --- | --- |
| aoi | `tests/test_notebook_parity.py` | `aoi_clean.csv` 字节一致、`model_metrics.json` 数值一致、9 张图文件名一致 |
| chiptest | `tests/test_chip_notebook_parity.py` | `chip_parts.csv`、`chip_tests.csv` 字节一致，`chip_summary.json` 数值一致，7 张图文件名一致 |

两个版本写的是**同名的图和数据文件**，后运行的会覆盖先运行的。

## 快速开始

先安装依赖（一次即可）：

```bash
uv sync    # 安装依赖（Python 3.13）
```

> 原始数据和清洗后的 CSV（`data/raw/`、`data/processed/*.csv`）被 `.gitignore` 排除，**新克隆的仓库里没有它们**，
> 所以 AOI 和芯片测试要**先运行各自的数据生成器**（固定随机种子，结果可复现）。QPSK 的两个入口都直接生成比特。

### aoi（Apple MQE）

```bash
uv run python scripts/generate_data.py                  # 1. 生成模拟数据

# 2A. Python 版（约 10 秒）
uv run python -m aoi.pipeline                           #     清洗 → 诊断 → 复判模型 → aoi_clean.csv、model_metrics.json、9 张图
uv run python -m aoi.pipeline --no-plots                #     只出数据和指标，跳过图片
uv run python -m aoi.pipeline --root DIR                #     处理另一个目录（DIR/data/raw/aoi_inspection.csv），不动仓库里的文件

# 2B. Notebook 版
uv run jupyter lab notebooks/aoi/                      #     逐个打开 01–03 阅读（仓库里的已执行，带输出）
# 或在命令行里按顺序重跑（02 生成 03 依赖的 aoi_clean.csv）
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/aoi/01_quality_metrics.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/aoi/02_aoi_analysis.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/aoi/03_ml_experiment.ipynb
```

### chiptest（华为 芯片测试工程）

```bash
uv run python scripts/generate_chip_data.py             # 1. 生成仿 STDF 的 ATE 日志（8 个 lot、约 117 万行、21.5 MB）

# 2A. Python 版（约 6 秒）
uv run python -m chiptest.pipeline                      #     解析 → 清洗 → 分析 → chip_parts.csv、chip_tests.csv、chip_summary.json、7 张图
uv run python -m chiptest.pipeline --no-plots           #     跳过图片
uv run python -m chiptest.pipeline --root DIR           #     处理另一个目录（DIR/data/raw/chip/*.log），不动仓库里的文件

# 2B. Notebook 版
uv run jupyter lab notebooks/chiptest/                 #     逐个打开 01–03 阅读（仓库里的已执行，带输出）
# 或在命令行里按顺序重跑（01 → 02 → 03，后一个读取前一个写出的文件）
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/chiptest/01_chip_log_parsing.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/chiptest/02_chip_yield_attribution.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/chiptest/03_chip_pat_efficiency.ipynb
```

芯片测试的 02 会重写 `chip_summary.json` 的前半部分，需要再运行 03 补全。

### QPSK（通信工程）

第一次用 macOS 终端时，从 [QPSK ADR 的第一周讲义](docs/adr-qpsk.md#第-1-周从终端开始)开始。装好环境后在仓库根目录运行。

```bash
uv run jupyter lab notebooks/qpsk/
```

打开 `01_qpsk_link_ber.ipynb`，从第一格向下运行。也可以从终端一键重跑并保存输出。

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/qpsk/01_qpsk_link_ber.ipynb
```

Python 版直接从终端生成 [BER 数据表](reports/qpsk/qpsk_ber.csv)、[BER 曲线](reports/qpsk/qpsk_ber.png)和 [I/Q 星座图](reports/qpsk/qpsk_constellation.png)。默认结果写入 `reports/qpsk/`，重新运行会更新这三个文件。

```bash
uv run python -m qpsk.pipeline
uv run python -m qpsk.pipeline --bits 40000 --output-dir terminal-practice/qpsk-small
```

第二条命令把每点比特数从 40 万改为 4 万，并把结果写到单独的目录。默认扫 0–12 dB，随机种子为 2026，可用 `--seed` 修改。

### 启动看板

AOI 和芯片测试共用一个 Streamlit 看板，**先完成上面对应方向的 2A 或 2B**，看板读取它们产出的文件：

```bash
uv run streamlit run app.py
```

看板不依赖 Notebook：数据缺失时页面会提示生成方式。环境变量 `AOI_ROOT` 可以让命令行和看板读取另一个数据目录（两个方向共用，沿用最初的名字）。

| 方向 | 页面 | 对应 Notebook | 内容 |
| --- | --- | --- | --- |
| aoi | `app.py`（主页） | 02 | 质量总览：筛选、指标卡、趋势、设备与缺陷类型分布、明细 |
| aoi | 指标计算器 | 01 | 输入 TP/FP/FN/TN 得到全部指标；准确率陷阱；缺陷率如何影响 Precision；**不需要数据文件** |
| aoi | 异常诊断 | 02 | 设备×日期热力图，下钻任意设备与日期，批次筛查，班次与缺陷类型 |
| aoi | 复判模型 | 03 | 时间切分、基线对比、验证集选阈值、测试集一次评估、漂移、可选的泄漏演示 |
| chiptest | 芯片测试 | 02、03 | 良率与 Pareto、热点归因与假设、测试机参数对比、晶圆图、PAT、测试效率 |

复判模型页有两条刻意的约束：阈值**只由验证集决定**，页面上的滑块只能观察验证集，不会改变最终阈值，更不会碰测试集；
泄漏演示要训练 16 个随机森林，所以放在按钮后面按需运行。

## 怎么使用这个项目

AOI 和芯片测试可按下面的顺序练习：

1. **先看方向说明**（下文“方向一 / 方向二”）：知道这个方向在回答什么问题、要证明什么能力。
2. **生成数据，打开 Notebook 版**：按编号顺序读，看每一步怎么从数据推到结论。仓库里的 Notebook 已执行带输出，不运行也能读。
3. **对照 Python 版**：看同样的逻辑怎么被抽成函数、怎么被测试覆盖。
4. **启动看板**：换筛选条件、换测试机或晶圆，验证 Notebook 里的结论。
5. **读面试笔记，再合上笔记用自己的话讲出来**：两份笔记都有 3 分钟项目介绍、模拟题，以及“被追问时的诚实回答”。

使用边界：数据是模拟的，项目里没有用过真实的 AOI 或 ATE。面试时要说“我用 Python 复现并分析了一套模拟数据”，
**不要说成真实产线经历**。每个方向里都标明了哪些指标只有模拟数据才算得出（因为有“上帝视角”的真值）。

## 方向一：aoi（Apple MQE）

**在做什么**：用 Python 分析 AOI 检测数据 → 找出误报问题 → 尝试建立复判模型 → 设计设备验证方案 → 用中英文讲清楚结果。
这个方向有一份按天安排的练习大纲：[`docs/adr-aoi.md`](docs/adr-aoi.md)。

### 5 天交付物对照

| 天 | 大纲要求的交付物 | 在哪里 |
| --- | --- | --- |
| 1 | 混淆矩阵与指标 Notebook | `notebooks/aoi/01_quality_metrics.ipynb` |
| 1 | 1 分钟解释：准确率高不代表可靠；误报增加如何排查 | `reports/interview_notes.md` 第 1、2 节 |
| 2 | 清洗后的数据 / Notebook；至少 3 张分析图；异常分析说明 | `notebooks/aoi/02_aoi_analysis.ipynb`、`reports/dashboard/`、`interview_notes.md` 第 3 节 |
| 2 | 可选：Streamlit 看板 | `app.py`、`pages/`（多页看板，Python 版） |
| 3 | 可运行的机器学习 Notebook；阈值对比表；准确率与数据划分的说明 | `notebooks/aoi/03_ml_experiment.ipynb`（Python 版：`aoi/review_model.py`）、`interview_notes.md` 第 4 节 |
| 4 | `validation_plan.md`；GRR/一致性/稳定性笔记；2 分钟回答 | `reports/validation_plan.md`、`interview_notes.md` 第 5 节 |
| 5 | 自我介绍、3 分钟项目介绍、10 道模拟题、英文答案、反问问题、速记卡 | `reports/interview_notes.md` 第 6–10 节 |

补充参考文档在 `docs/`：`做了什么.md`（已完成内容与位置对照）、`模型训练与实验技术全景.md`（第 3 天实验）、
`LogisticRegression监督学习分类模型.md`、`model_metrics的作用.md`、`各个Python库的用途.md`。

### 关键设计与诚实说明

- **统计单位**：误报率统一以**每件产品的首检**为单位，避免复测重复计数。
- **全量参考判定只存在于模拟数据**：真实产线通常只复判报警件，无法直接算 Recall 和良品误报率。
  Notebook 02 第 4 节和看板里的"仅模拟数据可得的指标"都明确标出了这一点。
- **复判模型要和朴素基线比较**：Notebook 03 同时评估了"只用 AOI 分数"。在当前数据上，模型相对基线只有小幅提升（PR-AUC +0.017）。
  这不是 bug，而是本练习想让你看到的事实：模型价值取决于有没有分数之外的有效信息。
- **数据泄漏演示的幅度是实测的**：复测样本跨集合会让 PR-AUC 高估约 0.017，在本数据上不大，结论文字里已如实说明。
- **测试集缺陷数量有限**，Recall 的 95% 置信区间约 94.7%–99.8%，不能把点估计当结论。

<details>
<summary>数据字典：<code>data/raw/aoi_inspection.csv</code> 的 13 个字段</summary>

字段与大纲一致：

| 字段 | 含义 |
| --- | --- |
| `inspection_id` | 检测事件 ID（一次检测一个，重复即脏数据） |
| `product_id` | 工件 ID（**同一件产品被 AOI 报警后可能复测，所以会重复，这是合理的**） |
| `inspection_time` | 检测时间 |
| `line_id` / `machine_id` | 产线 L01–L03 / 设备 M01–M06（每线 2 台） |
| `batch_id` | 批次（每条产线每天一个） |
| `shift` | `night`（00–08）、`day`（08–16）、`swing`（16–24） |
| `defect_type` | 参考判定为缺陷时是真实类型；良品被误报时是 AOI 报出的类型；其余为 `none` |
| `aoi_result` | AOI 判定 OK / NG（分数 ≥ 0.5 判 NG） |
| `reference_result` | 独立参考判定 OK / NG（约 0.5% 缺失） |
| `aoi_score` / `brightness` / `contrast` | AOI 缺陷分数 / 图像亮度 / 图像对比度（后两者约 0.4% 缺失） |

数据里**故意放了脏数据**：约 0.2% 的整行重复、少量缺失值，以及"同一产品多次检测"，用于练习第 2 天的数据体检。

`aoi_clean.csv`（Notebook 02 或 `python -m aoi.pipeline` 生成）在原字段上增加了 `date`、`hour`、`inspection_seq`、`is_first_pass`、`has_reference`、
`is_alarm`、`is_actual_defect`、`is_actual_good`、`is_true_positive`、`is_false_positive`、`is_miss`。

</details>

<details>
<summary>剧透：数据里植入的线索（想自己先找出来的话不要展开）</summary>

| 线索 | 内容 |
| --- | --- |
| 设备 | M03 在 2026-10-14（周三）06:00 之后光源逐渐衰减：亮度下降、良品误报上升，次日恢复；同批次的 M04 正常 |
| 批次 | L03 在 2026-10-27 的批次（B081）表面反光：亮度偏高、对比度偏低，该批次所有设备误报上升；训练期没出现，用来检验漂移 |
| 班次 | 夜班基线误报略高 |
| 复测 | 约 15–16% 的报警样本有复测，复测与首检特征高度相似，用于演示数据泄漏 |
| 缺陷特征 | 真实缺陷带有独立于 AOI 分数的对比度特征（stain 最弱），所以对比度能辅助区分误报 |

</details>

## 方向二：chiptest（华为 芯片测试工程）

**在做什么**：把芯片测试机（ATE）吐出的海量文本日志，流式解析、清洗，再回答“良率为什么低”：
是**测试硬件**（接触不良、电源噪声、时序偏移）的问题，还是**芯片/工艺**本身的问题？
并进一步做晶圆图、PAT 离群筛选、过杀与漏测、测试时间优化。

它在 Apple 内容之上延续，对应岗位职责第 4 条“工程分析和大数据挖掘”：处理海量日志、数据清洗、统计分析（含异常值检测）、编写高效工具。
本方向的练习安排见 [芯片测试 ADR](docs/adr-chiptest.md)。面试讲解材料见 [`reports/huawei_chip_test_notes.md`](reports/huawei_chip_test_notes.md)，其中第 1 节把岗位职责与本项目的证据一一对照。

### 三个 Notebook 与对应的 Python 模块

| Notebook | 内容 | 产出 | 对应的 Python 模块 |
| --- | --- | --- | --- |
| [01](notebooks/chiptest/01_chip_log_parsing.ipynb) | 流式解析日志、脏数据体检、清洗（首测 / 最终 / 复测，按限值重算失败） | `chip_parts.csv`、`chip_tests.csv` | `logparse.py`、`data.py` |
| [02](notebooks/chiptest/02_chip_yield_attribution.ipynb) | 三个良率口径、同 lot 对照找热点、测试机噪声与偏移、指纹与假设、晶圆图与置换检验 | c01–c05；`chip_summary.json` 前半部分 | `yield_analysis.py`、`attribution.py`、`parametric.py`、`wafer.py` |
| [03](notebooks/chiptest/03_chip_pat_efficiency.ipynb) | 残差 PAT 与静态阈值对比、过杀与漏测、测试耗时、冗余候选、测试顺序 | c06、c07；补全 `chip_summary.json` | `parametric.py`、`evaluation.py`、`efficiency.py` |

### 关键设计与诚实说明

- **解析器逐行读取日志**：按晶圆和测试轮次整理成 DataFrame 并保存在内存，解析结束后合并，再由流水线写出 CSV。内存会随保留的解析结果增加。本机解析 21.5 MB 约 0.5 秒。
  这个规模并不“海量”，笔记里明确了真实 GB / TB 级需要换存储和并行方式。
- **体检报告记录常见日志异常**：例如截断的参数记录、1E+38 哨兵值、nan、缺失的芯片结束记录和重复的芯片块。统一单位的转换次数也会记录。
- **归因只是假设**：同 lot 内对照测试机与 site，再用复测恢复率、均值与离散度区分“接触 / 噪声 / 时序 / 工艺”。输出的是“像……”的假设和验证方法，不是结论。
- **测试机比较用共识基线**：每片晶圆固定在一台测试机上，晶圆差异会混进来，所以用多台测试机的中位数做基线（至少 3 台）。真实场景仍要用同一批样片在各台机器上做相关性验证。
- **晶圆图的“成片失败”要经过置换检验**，而不是看到红点就下结论。
- **过杀、漏测、PAT 精度召回只在模拟数据里可算**（有 `truth.csv`）。真实场景没有 truth，只能靠复测、可靠性试验与客退间接估计。
- **“冗余测试”只是候选**：覆盖重叠高不等于可以删，必须看是否有芯片只被它抓到，并与设计团队确认。

<details>
<summary>日志格式：仿 STDF 命名的简化文本（<b>不是真实 STDF</b>，字段含义做过简化）</summary>

| 记录 | 内容 |
| --- | --- |
| `HBR` | `HBR\|bin\|名称`：硬 bin 定义 |
| `TSR` | `TSR\|测试号\|名称\|单位\|下限\|上限\|耗时ms`：测试项声明与限值，每片晶圆重复声明，单位可能变化（如 uA / mA） |
| `WIR` / `WRR` | 晶圆开始 `WIR\|lot\|晶圆\|测试机\|程序\|第几次测试\|时间` / 晶圆结束 `WRR\|lot\|晶圆\|第几次测试\|芯片数`（用来核对解析出的芯片数） |
| `PIR` / `PRR` | 芯片开始 `PIR\|编号\|x\|y\|site\|时间` / 芯片结束 `PRR\|编号\|硬bin\|P或F` |
| `PTR` | `PTR\|测试号\|测量值\|标志`：参数测试结果，占默认模拟日志约 86.5% 的行 |

芯片按 2×2 一次测 4 颗（4 个 site）；首测失败的芯片会在同一片晶圆的第 2 次测试里复测，所以同一颗芯片可能出现多次。

</details>

<details>
<summary>剧透：芯片日志里植入的线索（想自己先找出来的话不要展开）</summary>

| 线索 | 内容 |
| --- | --- |
| 接触不良 | L04 的测试机 T02 上，site 3 约 30% 的芯片接触电阻偏高（`CONT_VDD`），复测大量恢复 |
| 电源噪声 | L06 的测试机 T03 上，`VMIN` 的噪声变大（标准差约 2.3 倍），均值不变 |
| 时序偏移 | 测试机 T01 从 L05 起，`SCAN_FMAX` 整体偏低（约 4%），复测基本不恢复 |
| 整批工艺偏移 | L07 整批偏快角：更快、漏电更大、Vmin 更低 |
| 划伤 | L03 的 07 号晶圆上有一条划痕，形成一串相连的失败芯片 |
| 边缘损失 | 越靠晶圆边缘，致命缺陷率越高（最外圈约 4 倍） |
| 潜在隐患 | 约 0.5% 的“好”芯片漏电偏高，通过规格但有隐患（只有 truth 知道），用来评估 PAT |
| 脏数据 | 截断行、哨兵值、nan、缺失记录、重复芯片块；L02 的 05 号晶圆把 `IDD_STANDBY` 改用 mA 声明 |

</details>

## 方向三：QPSK 链路仿真与 BER 曲线

用一份 Notebook 串起比特生成、Gray 映射、复基带 AWGN、I/Q 硬判决和 BER 统计。实验包含四个星座点的无噪声验证、不同信噪比的接收星座、模拟 BER 与理论曲线，以及高信噪比下零错误点的置信上限。

[QPSK ADR](docs/adr-qpsk.md) 给出六周带教安排，第一周从终端、uv 和 Git 起步，后续安排通信推导、独立复现和 CV/PS 表达。Notebook 按步骤解释模型，[Python 版](qpsk/pipeline.py)从终端批量复现。同一组参数的误码数、理论 BER 和零错误上限由一致性测试对照。

## 新增一个岗位方向

后续要为别的硬件测试类岗位加方向时，沿用 chiptest 的做法，一个方向就是下面这一组文件（`<方向>` 换成新方向的名字）：

| 要有什么 | 放在哪里 | 约定 |
| --- | --- | --- |
| 模拟数据生成器 | `scripts/generate_<方向>_data.py` | 固定随机种子；植入可检验的线索；真值（truth）单独存放，只用于模拟数据上的评估 |
| Python 包 | `<方向>/` | `paths.py`（支持 `AOI_ROOT`）、清洗、分析、`plots.py`、`pipeline.py`（`python -m <方向>.pipeline`，带 `--root`、`--no-plots`） |
| ADR | `docs/adr-<方向>.md` | 写清目标、实验顺序和验收方式 |
| Notebook 版 | `notebooks/<方向>/NN_*.ipynb` | 每个方向从 01 开始；内联代码；按依赖顺序执行后带输出保存 |
| 看板页（可选） | `pages/N_<页面名>.py` | 数据缺失时给出生成命令，不抛异常 |
| 图与笔记 | `reports/<方向>/`、`reports/<岗位>_notes.md` | 图的文件名两个版本一致；笔记里写清楚哪些结论只在模拟数据里成立 |
| 测试 | `tests/test_<方向>_*.py` | 单元测试；**能检出植入的线索，同时不误报**；流水线测试；`test_<方向>_notebook_parity.py`（`slow`） |

完成后在本文件的“方向总览”表和“快速开始”里加上这个方向。

## 目录结构

```text
apple-mqe-interview/
├── README.md
├── pyproject.toml                    # uv、ruff、pytest 配置
├── docs/                             # 三个方向的 ADR 与补充参考文档
│   ├── adr-aoi.md
│   ├── adr-chiptest.md
│   └── adr-qpsk.md
├── scripts/
│   ├── generate_data.py              # [aoi] AOI 模拟数据生成器
│   └── generate_chip_data.py         # [chiptest] 芯片测试模拟日志生成器
├── data/
│   ├── raw/aoi_inspection.csv        # [aoi] 模拟原始数据（60,514 条检测记录，不进 git）
│   ├── raw/chip/                     # [chiptest] 模拟 ATE 日志与 truth.csv（约 21.5 MB，不进 git）
│   └── processed/
│       ├── aoi_clean.csv             # [aoi] 清洗后数据（不进 git；Notebook 02 或 aoi.pipeline 生成）
│       ├── model_metrics.json        # [aoi] 复判模型关键指标（Notebook 03 或 aoi.pipeline 生成）
│       ├── chip_parts.csv            # [chiptest] 清洗后的逐芯片数据（不进 git；Notebook 01 或 chiptest.pipeline 生成）
│       ├── chip_tests.csv            # [chiptest] 测试项目录与限值（不进 git；同上）
│       └── chip_summary.json         # [chiptest] 关键结论摘要（Notebook 02、03 或 chiptest.pipeline 生成）
├── notebooks/
│   ├── aoi/
│   │   ├── 01_quality_metrics.ipynb
│   │   ├── 02_aoi_analysis.ipynb
│   │   └── 03_ml_experiment.ipynb
│   ├── chiptest/
│   │   ├── 01_chip_log_parsing.ipynb
│   │   ├── 02_chip_yield_attribution.ipynb
│   │   └── 03_chip_pat_efficiency.ipynb
│   └── qpsk/
│       └── 01_qpsk_link_ber.ipynb
├── aoi/                              # —— aoi 的 Python 版 ——
│   ├── paths.py                      # 数据路径（可用 AOI_ROOT 覆盖）
│   ├── data.py                       # 读取、体检、清洗、派生列（Notebook 02 前半）
│   ├── metrics.py                    # 混淆矩阵指标、汇总（Notebook 01、02）
│   ├── diagnosis.py                  # 异常下钻、批次筛查、班次检验（Notebook 02 后半）
│   ├── review_model.py               # 切分、基线、阈值、泄漏演示、漂移、端到端（Notebook 03）
│   ├── plots.py                      # 9 张图，每张一个函数
│   ├── pipeline.py                   # 命令行入口：python -m aoi.pipeline
│   └── streamlit_support.py          # 看板页面共用的加载与缓存
├── chiptest/                         # —— chiptest 的 Python 版 ——
│   ├── logparse.py                   # 流式解析 STDF 风格日志，脏数据计数（Notebook 01）
│   ├── data.py                       # 体检、清洗、首测 / 最终 / 复测标记（Notebook 01）
│   ├── yield_analysis.py             # FPY、最终良率、bin Pareto（Notebook 02）
│   ├── attribution.py                # 同 lot 对照找 site / 测试机热点，生成指纹与假设（Notebook 02）
│   ├── parametric.py                 # 稳健 σ、测试机与批次效应、Cpk、残差 PAT（Notebook 02、03）
│   ├── wafer.py                      # 径向良率、成片失败的置换检验（Notebook 02）
│   ├── efficiency.py                 # 测试耗时、覆盖重叠、期望测试时间、顺序优化、删项 what-if（Notebook 03）
│   ├── evaluation.py                 # 仅模拟数据：过杀 / 漏测、PAT 精度召回（Notebook 03）
│   ├── report.py、plots.py           # 汇总结论、7 张图
│   ├── pipeline.py                   # 命令行入口：python -m chiptest.pipeline
│   └── paths.py、streamlit_support.py
├── qpsk/                             # Python 版入口 python -m qpsk.pipeline
│   └── pipeline.py
├── app.py                            # Streamlit 看板主页（aoi）
├── pages/                            # Streamlit 多页面：指标计算器、异常诊断、复判模型（aoi），芯片测试（chiptest）
├── tests/                            # pytest：单元测试、页面测试、Notebook 一致性测试
└── reports/
    ├── dashboard/                    # [aoi] 9 张分析图（PNG），两个版本生成的文件名一致
    ├── chip/                         # [chiptest] 7 张分析图（PNG），两个版本生成的文件名一致
    ├── qpsk/                         # [qpsk] BER 表、BER 曲线和接收星座图
    ├── validation_plan.md            # [aoi] 第 4 天：设备验证方案
    ├── interview_notes.md            # [aoi] 口头回答、项目介绍、模拟题、英文答案
    └── huawei_chip_test_notes.md     # [chiptest] 岗位对照、硬件经历衔接、模拟题、诚实回答
```

## 测试

```bash
uv run pytest                  # 全部（约 70 秒，含执行三个方向的 Notebook）
uv run pytest -m "not slow"    # 跳过 Notebook 一致性测试（约 40 秒），日常改代码时用这个
```

测试不依赖仓库里现成的数据：`tests/conftest.py` 用生成器在临时目录造数据，所以新克隆的仓库直接可以跑。

| 方向 | 文件 | 检查什么 |
| --- | --- | --- |
| aoi | `test_metrics.py`、`test_data.py` | 指标公式与讲义答案一致；分母为 0 返回 NaN；去重、首检、跨午夜复测 |
| aoi | `test_review_model.py` | 时间切分不跨集合；复测跨午夜仍在同一集合；**把测试期标签全部翻转，验证集预测和阈值必须不变**（阈值没有偷看测试集）；逐行切分比分组切分虚高 |
| aoi | `test_diagnosis.py` | 分析代码能找出数据里植入的线索（M03 设备、L03 反光批次、夜班），而不只是“能跑” |
| aoi | `test_pipeline.py` | 命令行产出 CSV、JSON 和 9 张图；缺少原始数据时给出可操作的提示 |
| chiptest | `test_chip_logparse.py` | 解析器：截断行、哨兵值、nan、缺 PRR、重复芯片块、WRR 计数不符、单位换算都被计数而不是抛异常；解析结果与生成器的 truth 对得上 |
| chiptest | `test_chip_analysis.py` | 分析代码能找出植入的线索（接触热点、测试机时序偏移、噪声、批次偏移、划伤、径向边缘损失），**同时不误报**（健康晶圆、未植入的测试机）；残差 PAT 优于静态阈值；复测降过杀、升漏测；期望测试时间与排序的单元测试 |
| chiptest | `test_chip_pipeline.py` | 命令行产出 CSV、JSON 和 7 张图；缺少日志时给出可操作的提示；`--no-plots`；没有 truth 文件时照常运行 |
| 共用 | `test_app_pages.py` | 每个页面无异常渲染；缺数据时的提示；滑块不改变最终阈值；分母为 0 时显示 n/a；芯片测试页默认展示噪声与划伤案例、可切换晶圆与测试项、缺数据时提示生成命令 |
| aoi | `test_notebook_parity.py`（`slow`） | **执行 Notebook 01–03**，与 Python 版比较（见“AOI 与芯片测试的两种实现”） |
| chiptest | `test_chip_notebook_parity.py`（`slow`） | **执行芯片测试 Notebook 01–03**，与 Python 版比较（见“AOI 与芯片测试的两种实现”） |
| qpsk | `test_qpsk_pipeline.py` | 命令行产物与参数校验；真实执行 Notebook，比对逐点误码数、理论 BER 和零错误上限（`slow`） |

## 关于数据与环境

- 依赖由 `uv` 管理，Python 版本固定为 3.13（见 `.python-version`）。
- `.envrc` 在进入目录时自动激活 `.venv`（需要 `direnv`，首次执行 `direnv allow`）。
- 代码格式与检查：`uv run ruff format . && uv run ruff check .`（`docs/` 已排除，不会被改动）。
- 测试：`uv run pytest`，详见上文“测试”。
