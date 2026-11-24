这 8 项可以理解成一套用于**数据分析、机器学习和展示结果**的工具。结合你的 AOI 面试练习，它们各自负责的事情如下：

| 工具           | 主要作用           | 在 AOI 练习中做什么                  |
| -------------- | ------------------ | ------------------------------------ |
| `jupyter`      | 提供交互式编程环境 | 一段一段运行代码，记录笔记和查看图表 |
| `numpy`        | 处理数组和数值计算 | 生成模拟数据、计算和处理数字         |
| `pandas`       | 处理表格数据       | 读取 CSV，按设备、日期统计误报       |
| `matplotlib`   | 绘制基础图表       | 画设备误报数量、每日趋势             |
| `seaborn`      | 更方便地绘制统计图 | 对比设备、班次和批次的数据分布       |
| `scikit-learn` | 机器学习和模型评估 | 训练分类模型、计算混淆矩阵和召回率   |
| `scipy`        | 科学计算和统计分析 | 做统计检验、优化和信号处理           |
| `streamlit`    | 制作交互式网页应用 | 做一个可以筛选设备和日期的质量看板   |

其中，**Jupyter 是运行和组织练习的环境，其余大多是供 Python 代码调用的工具库。**

## 1. Jupyter：你的练习工作台

Jupyter 让你在 Notebook 中把代码拆成一小段一小段运行。

例如：

```python
print("开始学习 AOI")
```

运行后，结果直接显示在当前代码下方。你还可以在同一份 Notebook 中添加：

- 学习笔记；
- 数据表格；
- 图表；
- 对结果的解释。

你的练习文件一般是：

```text
aoi_practice.ipynb
```

你可以在 VS Code 中打开和运行它。

**对你目前来说：每天都会用到。**

## 2. NumPy：处理数字和数组

NumPy 擅长处理大量数字，例如一组检测分数、一批测量值。

```python
import numpy as np

scores = np.array([0.2, 0.5, 0.8])

print(scores.mean())  # 平均分
```

在我们前面的模拟练习中，NumPy 用来生成随机数据，例如：

```python
rng = np.random.default_rng(42)
```

它可以帮助模拟不同工件的亮度、对比度和 AOI 分数。

**对你目前来说：先知道它用于数值和数组处理，能看懂基础代码即可。**

## 3. Pandas：处理表格数据

Pandas 可以理解为 Python 中处理表格数据的主要工具。

假设你有一个 CSV 文件：

| 设备 | 检测结果 | 参考结果 |
| ---- | -------- | -------- |
| M01  | NG       | OK       |
| M02  | OK       | OK       |
| M03  | NG       | NG       |

Pandas 可以读取它：

```python
import pandas as pd

df = pd.read_csv("data/aoi_simulated.csv")

df.head()
```

然后按设备分组，统计每台设备的检测数量：

```python
df.groupby("machine_id").size()
```

在 AOI 练习中，Pandas 主要负责：

- 读取 CSV。
- 检查空值和重复数据。
- 筛选 AOI 报警记录。
- 按设备、日期、班次汇总。
- 计算误报数量和误报率。

**对你目前来说：这是最值得优先学习的库之一。**

## 4. Matplotlib：画基础图表

Matplotlib 可以把数字变成图表，例如：

- 每天的 AOI 报警数量趋势；
- 3 台设备的误报数量对比；
- 不同阈值下的模型结果。

```python
import matplotlib.pyplot as plt

machines = ["M01", "M02", "M03"]
alarms = [120, 135, 240]

plt.bar(machines, alarms)
plt.xlabel("Machine")
plt.ylabel("Alarm count")
plt.show()
```

**对你目前来说：先会画柱状图、折线图，并设置标题和坐标轴即可。**

## 5. Seaborn：更方便地画统计图

Seaborn 也是画图工具，它建立在 Matplotlib 之上，和 Pandas 表格配合比较方便。

例如，Pandas 整理好设备数据后，可以用 Seaborn 快速绘制对比图：

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data=df, x="machine_id", y="aoi_score")
plt.show()
```

它也常用于画分布图、箱线图和热力图。

**对你目前来说：可以暂时作为辅助工具。先熟悉 Matplotlib，再按需要学习 Seaborn。**

## 6. scikit-learn：训练模型和评估结果

`scikit-learn` 是机器学习工具库。它可以用来：

- 划分训练集和测试集；
- 处理缺失数据；
- 训练分类模型；
- 计算混淆矩阵、召回率和精确率。

例如：

```python
from sklearn.metrics import confusion_matrix
```

这里有个容易困惑的地方：

> **安装名称是 `scikit-learn`，代码中的导入名称是 `sklearn`。**

这两个名称指向同一个项目。[PyPI](https://pypi.org/project/sklearn/?utm_source=chatgpt.com)

在你的练习中，它用于训练一个简单模型，尝试判断：

> AOI 报警后，这条记录更可能是真实缺陷，还是误报？

**对你目前来说：先理解模型输入、预测结果、训练集和测试集，再学习如何评估误报与漏检。**

## 7. SciPy：统计和科学计算

SciPy 提供更多科学计算工具，例如：

- 统计检验；
- 数学优化；
- 信号处理；
- 数值计算。

在质量分析中，它可以辅助研究：

> 某次设备调整前后的测量数据，差异有多大？是否可能受随机波动影响？

不过，统计分析需要先确认样本来源和适用条件。**对于当前 5 天准备，你不需要优先深入 SciPy。**

它与 NumPy 的关系可以简单理解为：

- **NumPy**：提供数组和基础数值计算能力。
- **SciPy**：在此基础上提供更多科学计算方法。

## 8. Streamlit：制作可以操作的看板

Streamlit 可以用 Python 制作简单的网页应用。例如一个 AOI 看板：

- 选择设备；
- 选择时间范围；
- 查看报警和误报趋势；
- 筛选检测记录。

使用者可以通过浏览器操作，而不需要自己写代码。

不过，Streamlit 通常需要单独编写 `.py` 文件并启动应用。对当前学习阶段来说，**它是可选项**：你可以先在 Notebook 中完成数据分析和图表，再考虑制作看板。

---

## 它们在练习中怎么配合？

你可以把整个过程想成：

1. 用 **Jupyter** 打开练习本。
2. 用 **NumPy** 生成模拟检测数据。
3. 用 **Pandas** 整理和统计数据。
4. 用 **Matplotlib / Seaborn** 把结果画出来。
5. 用 **scikit-learn** 尝试训练模型并评估。
6. 需要更深入的统计分析时使用 **SciPy**。
7. 需要交互式看板时使用 **Streamlit**。

## 版本号是什么意思？

例如：

```toml
"pandas>=3.1.1"
```

表示要求安装 **3.1.1 或更高的兼容版本**。`>=` 是“最低版本要求”，并不表示只能安装 `3.1.1`。

实际能安装哪个版本，还取决于你的 Python 版本、操作系统以及其他依赖是否兼容。如果这些内容是在 `pyproject.toml` 中，使用 `uv` 时通常可以通过 `uv.lock` 记录解析后的具体版本。

对于你的准备，我建议优先关注：

> **Jupyter → Pandas → Matplotlib → scikit-learn**

NumPy 跟着练习了解基础用法即可；Seaborn、SciPy 和 Streamlit 等用到时再学。这样能更快进入 AOI 分析，而不必先把 8 个工具全部学完。
