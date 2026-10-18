"""Pygame visualizer with optional SAC and keyboard control."""

from __future__ import annotations

import math
import numpy as np
import pygame
from stable_baselines3 import SAC

from config import build_env_kwargs, build_parser
from env import CartPendulumEnv


WIDTH, HEIGHT = 1000, 600
BACKGROUND = (245, 245, 245)
COLORS = [(40, 110, 200), (60, 150, 90), (190, 100, 45), (130, 80, 170)]


def load_policy(args, env: CartPendulumEnv):
    path = args.load_path or args.save_path
    try:
        model = SAC.load(path, env=env)
        return model
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
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Cart + Passive Pendulum")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    scale = min(0.35 * WIDTH / args.slider_range, 0.25 * HEIGHT / sum(env.segment_lengths))
    ground_y = int(HEIGHT * 0.70)
    obs, _ = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, _ = env.reset()

        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        if right and not left:
            action, source = np.array([1.0], dtype=np.float32), "MANUAL"
        elif left and not right:
            action, source = np.array([-1.0], dtype=np.float32), "MANUAL"
        elif model is not None:
            action, _ = model.predict(obs, deterministic=True)
            source = "POLICY"
        else:
            action, source = np.array([0.0], dtype=np.float32), "IDLE"
        obs, _, _, truncated, _ = env.step(action)
        if truncated:
            obs, _ = env.reset()

        screen.fill(BACKGROUND)
        world_to_screen = lambda x, y: (int(WIDTH / 2 + x * scale), int(ground_y - y * scale))
        pygame.draw.line(screen, (90, 90, 90), (0, ground_y), (WIDTH, ground_y), 2)
        rail_y = ground_y
        pygame.draw.line(
            screen, (190, 190, 190),
            world_to_screen(-args.slider_range, 0), world_to_screen(args.slider_range, 0), 3
        )
        cart_x = float(obs[0])
        previous = (cart_x, 0.0)
        angles = [math.atan2(obs[4 + 3 * i], obs[3 + 3 * i]) for i in range(env.num_segments)]
        for index, (length, angle) in enumerate(zip(env.segment_lengths, angles)):
            endpoint = (
                # Pymunk angles are positive counter-clockwise: at angle 0,
                # the local +Y rod direction maps to (-sin(angle), cos(angle)).
                previous[0] - length * math.sin(angle),
                previous[1] + length * math.cos(angle),
            )
            pygame.draw.line(screen, COLORS[index % len(COLORS)],
                             world_to_screen(*previous), world_to_screen(*endpoint), 4)
            pygame.draw.circle(screen, COLORS[index % len(COLORS)], world_to_screen(*previous), 5)
            previous = endpoint
        pygame.draw.circle(screen, (40, 110, 200), world_to_screen(cart_x, 0), max(6, int(0.03 * scale)))

        action_value = float(np.asarray(action).reshape(-1)[0])
        lines = [
            f"mode={args.mode}  source={source}  gravity={env.current_gravity:.2f}/{args.gravity:.2f}",
            f"segments={env.num_segments}  lengths={env.segment_lengths}",
            f"action={action_value:+.3f}  cart_x={cart_x:+.3f}",
        ]
        for index, text in enumerate(lines):
            screen.blit(font.render(text, True, (30, 30, 30)), (15, 15 + index * 23))
        screen.blit(font.render("LEFT/A                 RIGHT/D", True, (70, 70, 70)), (WIDTH // 2 - 130, HEIGHT - 30))
        pygame.display.flip()
        clock.tick(60)
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
