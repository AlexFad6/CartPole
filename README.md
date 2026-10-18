# CartPole 多节倒立平衡摆

基于 pymunk、Gymnasium、Stable-Baselines3 SAC 和 pygame 的 2D cart-pendulum 项目。

## 快速开始

```powershell
cd D:\Desktop\CartPole
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 训练

默认训练：

```powershell
python train.py
```

单节摆：

```powershell
python train.py --segments 1 --lengths 0.5 --train-steps 300000
```

课程学习：

```powershell
python train.py --segments 1 --gravity 9.81 --gravity-start 0 --curriculum-steps 200000
```

速度控制：

```powershell
python train.py --segments 1 --mode velocity
```

GPU 训练：

```powershell
python train.py --device cuda
```

`--device` 可选 `auto`、`cuda` 或 `cpu`。pymunk 物理仿真仍运行在 CPU，GPU 主要加速 SAC 网络训练。

并行训练 8 个独立环境：

```powershell
python train.py --num-envs 8 --device cuda
```

`--train-steps` 表示 vectorized environment 的步数；8 个环境时每一步会收集 8 条 transition。Windows 下使用 `spawn` 启动子进程。

训练模型默认保存为 `sac_pendulum.zip`，可用 `--save-path` 修改。

## 可视化与手动控制

```powershell
python evaluate.py --segments 1
```

加载指定模型：

```powershell
python evaluate.py --segments 2 --lengths 0.4 0.3 --load-path sac_two_link
```

| 按键 | 操作 |
| --- | --- |
| `←` / `A` | 向左控制 |
| `→` / `D` | 向右控制 |
| `R` | 重置 |
| `Esc` | 退出 |

没有按键输入时，有模型则使用策略，没有模型则使用零动作。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--segments` | `2` | 摆杆节数 |
| `--lengths` | 自动均分 | 各节长度 |
| `--total-length` | `1.0` | 自动均分时的总长度 |
| `--gravity` | `9.81` | 终端重力 |
| `--gravity-start` | `0.0` | 课程学习起始重力 |
| `--curriculum-steps` | `0` | 课程学习步数 |
| `--mode` | `acceleration` | 加速度或速度控制 |
| `--max-acceleration` | `20.0` | 最大加速度 |
| `--max-velocity` | `5.0` | 最大目标速度 |
| `--slider-range` | `2.0` | cart 滑轨范围 |
| `--wall-penalty` | `0.1` | 碰壁奖励惩罚 |
| `--train-steps` | `300000` | 训练步数 |
| `--num-envs` | `1` | 并行物理环境数量 |
| `--physics-iterations` | `10` | 每个 pymunk 物理步的约束求解迭代次数 |
| `--device` | `auto` | SAC 运行设备 |
| `--save-path` | `sac_pendulum` | 模型保存路径 |
| `--load-path` | 空 | 评估时加载的模型 |

碰壁不会终止 episode，只会扣除 `wall-penalty`。

### 性能建议

- 多环境并行是主要加速方式；每个环境拥有独立的 pymunk `Space`，不会共享物理状态。
- `--physics-iterations 10` 保持默认物理精度。可在确认关节稳定性后尝试 `5`，但过低可能增加约束误差和抖动。
- GPU 只加速 SAC 的神经网络更新，pymunk 仍由 CPU 执行；通常应先增加 `--num-envs`，再考虑 GPU。

## 项目文件

- `env.py`：物理环境、动作、观测和奖励
- `config.py`：命令行参数
- `train.py`：SAC 训练
- `evaluate.py`：pygame 可视化和手动控制
- `requirements.txt`：依赖列表

## 开发计划

- [x] 增加固定世界坐标参考线 `x=0` 的奖励项
- [x] 增加可配置的碰壁惩罚
- [ ] 在 evaluate 中显示垂直参考线
- [ ] 手动输入停止时记录并固定参考线位置
- [ ] 显示 cart 与参考线的偏差
- [ ] 优化 cart、摆杆、关节、滑轨和参考线的视觉效果
- [ ] 优化信息面板和方向键状态提示
- [ ] 适配不同摆杆数量、长度和窗口尺寸的自适应缩放
