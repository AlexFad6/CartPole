"""Pygame GUI for the cart and passive pendulum environment."""

from __future__ import annotations

import math
from argparse import Namespace
from typing import Any

import numpy as np
import pygame

from env import CartPendulumEnv


WIDTH, HEIGHT = 1000, 600
BACKGROUND = (9, 24, 34)
GROUND = (20, 39, 51)
RAIL = (47, 70, 82)
TEXT = (213, 224, 229)
ACCENT = (0, 190, 215)


def run(env: CartPendulumEnv, model: Any, args: Namespace) -> None:
    """Run the interactive visualization for an environment and optional policy."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Cart + Passive Pendulum")
    font = pygame.font.Font(None, 22) if args.verbose else None
    clock = pygame.time.Clock()
    scale = min(0.35 * WIDTH / args.slider_range, 0.52 * HEIGHT / sum(env.segment_lengths))
    ground_y = int(HEIGHT * 0.78)
    obs, _ = env.reset()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
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

            _draw_frame(screen, env, obs, action, args, source, font, scale, ground_y)
            pygame.display.flip()
            clock.tick(60)
    finally:
        env.close()
        pygame.quit()


def _draw_frame(
    screen: pygame.Surface,
    env: CartPendulumEnv,
    obs: np.ndarray,
    action: np.ndarray,
    args: Namespace,
    source: str,
    font: pygame.font.Font | None,
    scale: float,
    ground_y: int,
) -> None:
    screen.fill(BACKGROUND)
    world_to_screen = lambda x, y: (int(WIDTH / 2 + x * scale), int(ground_y - y * scale))
    rail_top = ground_y - 2
    pygame.draw.rect(screen, GROUND, (0, rail_top, WIDTH, HEIGHT - rail_top))
    pygame.draw.line(screen, RAIL, (0, rail_top), (WIDTH, rail_top), 3)
    for x in range(-20, WIDTH + 40, 28):
        pygame.draw.line(screen, (38, 59, 71), (x, rail_top + 2), (x - 13, rail_top + 15), 2)
    pygame.draw.line(
        screen,
        (60, 86, 98),
        world_to_screen(-args.slider_range, 0),
        world_to_screen(args.slider_range, 0),
        2,
    )

    cart_x = float(obs[0])
    previous = (cart_x, 0.0)
    angles = [math.atan2(obs[4 + 3 * i], obs[3 + 3 * i]) for i in range(env.num_segments)]
    for length, angle in zip(env.segment_lengths, angles):
        endpoint = (
            previous[0] - length * math.sin(angle),
            previous[1] + length * math.cos(angle),
        )
        start = world_to_screen(*previous)
        end = world_to_screen(*endpoint)
        pygame.draw.line(screen, (185, 202, 209), start, end, 3)
        pygame.draw.circle(screen, (8, 20, 29), start, 8)
        pygame.draw.circle(screen, (218, 229, 232), start, 6, 2)
        previous = endpoint

    cart_center = world_to_screen(cart_x, 0)
    cart_width = max(54, int(0.24 * scale))
    cart_height = max(24, int(0.11 * scale))
    cart_rect = pygame.Rect(
        cart_center[0] - cart_width // 2,
        cart_center[1] - cart_height // 2,
        cart_width,
        cart_height,
    )
    pygame.draw.rect(screen, (43, 57, 66), cart_rect, border_radius=2)
    pygame.draw.rect(screen, (118, 137, 145), cart_rect, width=1, border_radius=2)
    pygame.draw.circle(screen, (11, 31, 40), (cart_rect.left + 12, cart_rect.bottom + 3), 5)
    pygame.draw.circle(screen, (11, 31, 40), (cart_rect.right - 12, cart_rect.bottom + 3), 5)

    action_value = float(np.asarray(action).reshape(-1)[0])
    if abs(action_value) > 0.01:
        direction = 1 if action_value > 0 else -1
        arrow_x = cart_center[0] + direction * (cart_width // 2 + 16)
        pygame.draw.polygon(
            screen,
            ACCENT,
            [
                (arrow_x, cart_center[1]),
                (arrow_x - direction * 10, cart_center[1] - 5),
                (arrow_x - direction * 10, cart_center[1] + 5),
            ],
        )

    if args.verbose and font is not None:
        lines = [
            f"mode={args.mode}  source={source}  gravity={env.current_gravity:.2f}/{args.gravity:.2f}",
            f"segments={env.num_segments}  lengths={env.segment_lengths}",
            f"action={action_value:+.3f}  cart_x={cart_x:+.3f}",
        ]
        for index, text in enumerate(lines):
            screen.blit(font.render(text, True, TEXT), (15, 15 + index * 23))
