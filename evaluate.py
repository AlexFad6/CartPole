"""Evaluate a trained policy with the cart and pendulum GUI."""

from __future__ import annotations

from stable_baselines3 import SAC

from config import build_env_kwargs, build_parser
from env import CartPendulumEnv
from gui import run


def load_policy(args, env: CartPendulumEnv):
    path = args.load_path or args.save_path
    try:
        return SAC.load(path, env=env)
    except ValueError:
        print("[INFO] 模型与本环境观测维度不兼容，请重新训练")
    except (FileNotFoundError, OSError):
        print("[INFO] 未找到模型，将使用纯手动模式")
    except Exception as exc:
        print(f"[INFO] 模型加载失败，将使用纯手动模式：{exc}")
    return None


def main() -> None:
    args = build_parser().parse_args()
    kwargs = build_env_kwargs(args)
    kwargs["curriculum_steps"] = 0
    env = CartPendulumEnv(**kwargs)
    model = load_policy(args, env)
    run(env, model, args)


if __name__ == "__main__":
    main()
