"""Train a SAC policy for the 2D cart-pendulum environment."""

from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from config import build_env_kwargs, build_parser
from env import CartPendulumEnv


class GravityCurriculumCallback(BaseCallback):
    """Inject the current SAC timestep before environments reset."""

    def _on_step(self) -> bool:
        self.training_env.env_method("set_train_progress", self.num_timesteps)
        return True


def main() -> None:
    args = build_parser().parse_args()
    env_kwargs = build_env_kwargs(args)
    check_env(CartPendulumEnv(**env_kwargs), warn=True)
    env = DummyVecEnv([lambda: CartPendulumEnv(**env_kwargs)])
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=5000,
        buffer_size=300000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        verbose=1,
    )
    callbacks = None
    if args.curriculum_steps > 0:
        print(
            f"[INFO] 启用重力课程学习：{args.gravity_start} → {args.gravity} "
            f"over {args.curriculum_steps} steps"
        )
        callbacks = [GravityCurriculumCallback()]
    model.learn(total_timesteps=args.train_steps, callback=callbacks)
    model.save(args.save_path)
    print(f"[INFO] 模型已保存：{args.save_path}.zip")
    env.close()


if __name__ == "__main__":
    main()
