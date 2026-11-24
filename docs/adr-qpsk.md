# 从比特到曲线的完整链路仿真

## 项目目标

用 Python 复现一条 QPSK 复基带链路。自己生成比特，用 Gray 映射发射符号，通过 AWGN 信道，判决回比特，再把 Monte Carlo BER 与理论曲线叠在同一张图上。最终交付一个可重跑的 [Notebook](../notebooks/qpsk/01_qpsk_link_ber.ipynb)、一份可独立运行的 [Python 脚本](../qpsk/pipeline.py)、一张 BER 图和一段能独立讲清楚的实验说明。[默认参数生成的 BER 图](../reports/qpsk/qpsk_ber.png)

你已经会用 C++ 写循环和函数。第一周先练 macOS 终端和项目环境，此后用熟悉的编程概念学习 Python 数组、函数与数值实验。通信概念按实验顺序补。复用仓库现有的 Python 3.13、uv、NumPy、SciPy、Matplotlib 和 Jupyter。

基线采用理想同步下的符号级复基带模型。成形滤波、匹配滤波、定时与载波同步、衰落和纠错编码留作后续研究。理论 BER 使用相干 Gray QPSK 在 AWGN 下的公式。[MathWorks 的 QPSK 理论表达式](https://www.mathworks.com/help/comm/ug/analytical-expressions-used-in-berawgn-function-and-bit-error-rate-analysis.html#mpsk)

## 六周安排

前四周跑通作品，后两周用于独立复现和申请表达。把 CV 和 PS 的固定时段从第一周就留出来。以 12 月底完成内部交付为目标，给 1 月初申请材料检查和提交留缓冲。

| 周 | 亲手完成 | 带教验收 |
| --- | --- | --- |
| 1 | 用终端找目录、创建和移动文件。用 uv 同步环境，跑 Python 脚本，启动 Jupyter，查看 Git 改动。 | 关掉终端重开后，能进入仓库并运行 Notebook。配置卡住半天以内由带教接手。 |
| 2 | 用四组固定比特手算 I/Q 坐标，再用 NumPy 实现映射和按符号判决。 | 四个星座点和手算一致，无噪声逐比特还原。 |
| 3 | 补复数、正态分布、方差和 dB。比较 0 dB、8 dB 的接收星座，推导每路噪声标准差。 | 能解释为何每路方差是 $N_0/2$、为何每符号能量是每比特能量的两倍。 |
| 4 | 扫 $E_b/N_0$，输出每个点的错误数和总比特数，叠画理论曲线。 | 能解释低误码区的采样波动和零错误上限。保留第一张可用于作品集的图。 |
| 5 | 改样本量，复现稀有误码的波动。故意把噪声功率加倍，读出曲线偏移，再恢复正确参数。 | 从新内核和 Python 命令行各跑一次，对照误码数，记录种子、样本量与模型假设。 |
| 6 | 用自己的话写 CV 项目条目与 PS 片段，练习两分钟英文讲解。 | 能从笔记本第一格重跑，并用 BER 图回答一次追问。 |

## 第 1 周：从终端开始

### 第一次打开 Terminal

按 **Command + 空格**，搜索「终端」或「Terminal」，按回车。一次粘贴一行命令，按回车后等命令跑完。下面先在个人主目录查看路径。

```bash
pwd
ls
cd ~
pwd
```

- `pwd` 显示当前位置，`ls` 列出当前位置的内容，`cd` 改变位置。
- `~` 表示你的主目录。以 `/` 开头的是绝对路径，从当前位置出发的叫相对路径。 `.` 是当前目录，`..` 是上一级。

在 Finder 里找到带教给你的项目目录。终端先输入 `cd` 和一个空格，把该文件夹拖进终端，让 macOS 填入实际路径，然后按回车。运行以下两行，直到 `ls` 能看到 `README.md`、`pyproject.toml` 和 `notebooks`。

```bash
pwd
ls
```

### 用一个小脚本练习创建、移动和运行

在仓库根目录逐行运行。 `mkdir -p` 创建目录，`printf` 生成一行 Python，`>` 将文字写入文件，`mv` 移动文件。

```bash
mkdir -p terminal-practice/scripts
printf 'print("Hello, QPSK!")\n' > terminal-practice/hello.py
mv terminal-practice/hello.py terminal-practice/scripts/hello.py
ls terminal-practice/scripts
man mv
```

`man mv` 打开手册，按 **q** 返回命令行。之后还可以运行 `uv --help` 查询 uv 子命令。练习目录里只有你自己的文件，后面提交 Notebook 时只选择需要的文件。

### 安装并验证项目环境

先运行 `uv --version` 和 `git --version`。如果找不到 uv，请带教按本机情况完成安装，然后重新打开 Terminal。已安装 Homebrew 的电脑可以执行 `brew install uv`。Git 触发 macOS 开发者工具安装时也交给带教处理。[uv 官方安装方式](https://docs.astral.sh/uv/getting-started/installation/)

回到项目根目录，逐行执行。 `uv sync --locked` 按仓库锁文件准备 `.venv`，之后 `uv run` 会使用项目环境。

```bash
uv sync --locked
uv run python --version
uv run python terminal-practice/scripts/hello.py
uv run python -c "import numpy, scipy, matplotlib; print('imports OK')"
git status --short
```

预期能看到 Python 3.13、`Hello, QPSK!` 和 `imports OK`。 `git status --short` 显示本地改动，`??` 表示新文件还没有被 Git 跟踪。另开一个终端时，重复进入项目根目录即可继续用 `uv run`。想亲眼看看手动激活如何改变 Python 路径，可以试这三行。

```bash
source .venv/bin/activate
command -v python
deactivate
```

如果卡在安装、下载、解释器或内核选择，发给带教你执行的命令、`pwd` 输出和报错末尾。最多用半天定位，之后由带教接手配置，学生继续手算四个星座点。

### 运行 Notebook

```bash
uv run jupyter lab notebooks/qpsk/
```

打开 JupyterLab 左侧的 `01_qpsk_link_ber.ipynb`。选 **Python 3 (ipykernel)**，按 **Shift + Enter** 从第一格往下运行。浏览器没自动打开时，在这台电脑的浏览器打开终端给出的本地链接。退出时回到 Terminal 按 **Control + C**，按提示确认。[JupyterLab 启动文档](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html)

会用终端后，可以在仓库根目录从新内核重跑并保存结果。

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/qpsk/01_qpsk_link_ber.ipynb
git status --short
git diff --stat
```

### 看懂 Git 的三个状态

运行 `git status --short` 先确认改了哪些文件，`git diff --stat` 显示已跟踪文件的改动量。准备记录自己的实验时，先让带教确认本地工作分支和 Git 身份，再只暂存 Notebook。

```bash
git add -- notebooks/qpsk/01_qpsk_link_ber.ipynb
git diff --cached --stat
git commit -m "记录 QPSK 链路实验"
```

`git add` 选择下次提交的内容，`git commit` 在本机留下一条记录。用 `git log -1 --oneline` 检查刚才的记录。

## 用 Python 版独立复现实验

第四周完成 Notebook 后，在仓库根目录执行以下命令。

```bash
uv run python -m qpsk.pipeline
ls reports/qpsk/
```

`python -m qpsk.pipeline` 会运行 [Python 版源码](../qpsk/pipeline.py)，在 `reports/qpsk/` 里生成 [BER 表](../reports/qpsk/qpsk_ber.csv)、[BER 曲线](../reports/qpsk/qpsk_ber.png)和[接收星座图](../reports/qpsk/qpsk_constellation.png)。默认每点 40 万比特，随机种子 2026。实验时可保留默认结果，把新的参数组合写到另一处。

```bash
uv run python -m qpsk.pipeline --bits 40000 --seed 2026 --output-dir terminal-practice/qpsk-small
```

把新旧 CSV 并排看，先比较 `bit_errors` 和 `total_bits`，再比较 `simulated_ber`。零错误的行还记录了单侧 95% 上限 `zero_error_upper_95`。你也可以把 Notebook 的 `N_BITS` 改成同样的数，从新内核运行并核对结果。

## 作品验收

- **链路**。从全新内核按顺序执行。无噪声时能还原四组比特，加入噪声后能指出星座点跨越哪条轴导致哪一位翻转。
- **数据**。记下随机种子、每点比特数、扫点范围和错误数。用同一份 Notebook 更改样本量，说明为什么高信噪比端更难估准。
- **图**。横轴标 $E_b/N_0$（dB），纵轴标对数 BER。模拟点、理论线和零错误置信上限有各自的图例。用 Notebook 绘图单元末尾的可选命令导出 PNG。
- **讲解**。解释两个比特如何构成符号、噪声方差如何设定、理论和仿真在哪些条件下可以比较。把自己独立复现的结果写入 CV 和 PS。

完成后可据实改写这条英文 CV 描述：

> Implemented a Gray-coded QPSK baseband link in Python, compared Monte Carlo BER with the theoretical AWGN curve, and investigated the effect of sample size on low-BER estimates.

PS 用一段话讲清硬件背景如何帮助自己理解 I/Q，再举一个亲手做过的实验，例如用噪声功率翻倍检查 $E_b/N_0$ 的归一化。写下自己的观察、解释和下一步想学的通信问题。
