# CartPole 多节倒立平衡摆

这是一个基于 **pymunk 2D**、Gymnasium 和 Stable-Baselines3 SAC 的强化学习项目。系统由沿水平滑轨移动的 cart 和 N 节被动摆杆组成，支持加速度控制、目标速度控制、pygame 可视化以及重力课程学习。

## 1. 环境要求

- Windows 10/11
- Python 3.10 或更高版本

建议在项目目录中创建虚拟环境：

```powershell
cd D:\Desktop\CartPole
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 禁止执行激活脚本，可以直接使用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

主要依赖包括：

- `pymunk`：二维刚体物理仿真
- `gymnasium`：强化学习环境接口
- `stable-baselines3` 和 `torch`：SAC 训练
- `pygame`：评估和手动控制窗口
- `numpy`：数值计算

## 3. 训练模型

使用默认参数训练：

```powershell
python train.py
```

训练单节摆：

```powershell
python train.py --segments 1 --lengths 0.5 --train-steps 300000
```

训练两节摆，并设置更大的最大加速度：

```powershell
python train.py `
  --segments 2 `
  --lengths 0.4 0.3 `
  --max-acceleration 40 `
  --train-steps 500000 `
  --save-path sac_two_link
```

训练完成后，模型默认保存为 `sac_pendulum.zip`。使用 `--save-path` 可以修改保存路径。

### 3.1 重力课程学习

课程学习会在每次 episode 重置时，从逐渐扩大的重力区间中采样重力：

```powershell
python train.py `
  --segments 1 `
  --gravity 9.81 `
  --gravity-start 0.0 `
  --curriculum-steps 200000 `
  --train-steps 300000
```

训练日志会显示：

```text
[INFO] 启用重力课程学习：0.0 → 9.81 over 200000 steps
```

当 `--curriculum-steps 0` 时，不启用课程学习，环境固定使用 `--gravity`。

### 3.2 速度控制模式

默认模式是加速度控制。切换为目标速度控制：

```powershell
python train.py `
  --segments 1 `
  --mode velocity `
  --max-velocity 5 `
  --train-steps 300000 `
  --save-path sac_velocity
```

## 4. 可视化与手动控制

评估默认模型：

```powershell
python evaluate.py --segments 1
```

如果模型使用了非默认结构，评估时必须传入相同的结构参数：

```powershell
python evaluate.py `
  --segments 2 `
  --lengths 0.4 0.3 `
  --gravity 9.81 `
  --load-path sac_two_link
```

评估时会把课程学习关闭，并直接使用终端重力 `--gravity`，模拟训练完成后的部署环境。

### 键盘操作

| 按键 | 行为 |
| --- | --- |
| `←` 或 `A` | 手动向左控制 |
| `→` 或 `D` | 手动向右控制 |
| 两侧同时按或都不按 | 使用策略控制（加载模型时） |
| `R` | 重置 episode |
| `Esc` | 退出窗口 |

按住方向键时，手动控制优先于策略控制；未加载模型时，松开方向键会保持空动作。

## 5. 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--segments` | `2` | 摆杆节数 |
| `--lengths` | 自动均分 | 每节摆杆长度 |
| `--total-length` | `1.0` | 未指定 `--lengths` 时的总长度 |
| `--gravity` | `9.81` | 终端重力 |
| `--gravity-start` | `0.0` | 课程学习起始重力范围下限 |
| `--curriculum-steps` | `0` | 课程学习持续的训练步数 |
| `--mode` | `acceleration` | `acceleration` 或 `velocity` |
| `--max-acceleration` | `20.0` | 加速度模式的最大加速度 |
| `--max-velocity` | `5.0` | 速度模式的最大目标速度 |
| `--slider-range` | `2.0` | cart 可移动范围为 `[-range, range]` |
| `--wall-penalty` | `0.0` | cart 碰到滑轨边界时扣除的奖励 |
| `--max-episode-steps` | `2000` | episode 最大控制步数 |
| `--train-steps` | `300000` | SAC 总训练步数 |
| `--save-path` | `sac_pendulum` | 模型保存路径 |
| `--load-path` | 空 | 评估时优先加载的模型路径 |

例如，在火星重力下运行单节摆：

```powershell
python evaluate.py --segments 1 --lengths 0.5 --gravity 3.71
```

训练时启用碰壁惩罚，例如每个撞墙控制步扣 `0.1`：

```powershell
python train.py --segments 1 --wall-penalty 0.1 --train-steps 300000
```

碰壁不会终止或重置 episode；cart 会停在滑轨边界，环境仅从该控制步的 reward 中扣除惩罚。

## 6. 项目文件

- `env.py`：Gymnasium 环境、pymunk 物理、动作映射、观测和奖励
- `config.py`：命令行参数和环境参数构造
- `train.py`：SAC 训练和重力课程学习回调
- `evaluate.py`：pygame 渲染、模型推理和方向键控制
- `requirements.txt`：Python 依赖

训练过程不需要图形窗口；只有运行 `evaluate.py` 时才会打开 pygame 窗口。

## 7. 后续开发计划

以下功能计划在后续版本中实现：

### 7.1 增加垂直参考线与平衡奖励

- 在 cart 的当前位置绘制一条垂直参考线，作为摆杆平衡目标。
- 增加摆杆末端或整体质心相对于参考线的偏差计算。
- 当摆杆保持在参考线附近时，提高奖励；偏离参考线时降低奖励。
- 保留当前直立角度奖励，避免仅依赖位置误差导致摆杆倒立或方向错误。
- 明确定义参考线目标、允许误差范围和奖励权重，并通过参数支持调节。

### 7.2 手动控制时固定参考线

- `evaluate.py` 进入手动控制后显示垂直参考线。
- 当用户松开方向键、cart 停止输入时，记录当时的 cart 位置作为参考线位置。
- 后续手动操作过程中保持该参考线不随 cart 移动，便于观察 cart 是否回到目标位置以及摆杆是否在目标附近平衡。
- 按 `R` 重置 episode 时同步重置参考线状态。
- 在信息面板中显示参考线位置及当前 cart 与参考线的水平偏差。

### 7.3 可视化优化

- 优化 cart、摆杆、关节、滑轨和参考线的颜色、线宽与层次关系。
- 增加参考线、目标区域和当前状态的图例或视觉提示。
- 改进多节摆杆的颜色区分、关节标记和末端标记。
- 优化信息面板布局，显示模式、控制来源、重力、动作、cart 位置、参考线位置和偏差。
- 优化方向键提示及按键高亮状态。
- 确保不同摆杆数量、长度、滑轨范围和窗口尺寸下均能正确缩放显示。