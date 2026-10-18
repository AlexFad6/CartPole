"""Gymnasium environment for a 2D cart and a passive multi-link pendulum."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium import spaces


class CartPendulumEnv(gym.Env):
    """A cart constrained to a rail with N freely rotating passive links."""

    metadata = {"render_modes": [None]}

    def __init__(
        self,
        num_segments: int = 1,
        segment_lengths: list[float] | None = None,
        gravity: float = 9.81,
        gravity_start: float = 0.0,
        curriculum_steps: int = 0,
        control_mode: str = "acceleration",
        max_acceleration: float = 20.0,
        max_velocity: float = 5.0,
        cart_mass: float = 1.0,
        link_mass: float = 0.1,
        link_radius: float = 0.02,
        slider_range: float = 2.0,
        dt: float = 1 / 240,
        frame_skip: int = 4,
        max_episode_steps: int = 2000,
        wall_penalty: float = 0.1,
        reference_x: float = 0.0,
        reference_reward_weight: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments must be at least 1")
        if segment_lengths is None:
            segment_lengths = [0.5] * num_segments
        if len(segment_lengths) != num_segments:
            raise ValueError("segment_lengths must contain num_segments values")
        if any(length <= 0 for length in segment_lengths):
            raise ValueError("segment lengths must be positive")
        if gravity_start > gravity:
            raise ValueError("gravity_start must not exceed gravity")
        if control_mode not in {"acceleration", "velocity"}:
            raise ValueError("control_mode must be 'acceleration' or 'velocity'")
        if dt <= 0 or frame_skip < 1 or max_episode_steps < 1:
            raise ValueError("dt, frame_skip, and max_episode_steps must be positive")

        self.num_segments = int(num_segments)
        self.segment_lengths = [float(length) for length in segment_lengths]
        self.gravity = float(gravity)
        self.gravity_start = float(gravity_start)
        self.curriculum_steps = int(curriculum_steps)
        self.control_mode = control_mode
        self.max_acceleration = float(max_acceleration)
        self.max_velocity = float(max_velocity)
        self.cart_mass = float(cart_mass)
        self.link_mass = float(link_mass)
        self.link_radius = float(link_radius)
        self.slider_range = float(slider_range)
        self.dt = float(dt)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.wall_penalty = float(wall_penalty)
        self.reference_x = float(reference_x)
        self.reference_reward_weight = float(reference_reward_weight)
        self.render_mode = render_mode
        self._train_progress = 0
        self._step_count = 0
        self.current_gravity = self.gravity

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3 + 3 * self.num_segments,), dtype=np.float32
        )
        self.space = pymunk.Space()
        self.space.damping = 1.0
        self.cart_body: pymunk.Body
        self.link_bodies: list[pymunk.Body] = []
        self._build_physics()

    def _build_physics(self) -> None:
        self.cart_body = pymunk.Body(self.cart_mass, float("inf"))
        self.cart_body.position = (0.0, 0.0)
        cart_shape = pymunk.Circle(self.cart_body, 0.03)
        cart_shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(self.cart_body, cart_shape)

        # GrooveJoint supplies the hard horizontal rail and its two endpoints.
        rail = pymunk.GrooveJoint(
            self.space.static_body,
            self.cart_body,
            (-self.slider_range, 0.0),
            (self.slider_range, 0.0),
            (0.0, 0.0),
        )
        self.space.add(rail)

        for length in self.segment_lengths:
            body = pymunk.Body(self.link_mass, self.link_mass * length * length / 12.0)
            shape = pymunk.Segment(body, (0.0, -length / 2), (0.0, length / 2), self.link_radius)
            shape.filter = pymunk.ShapeFilter(group=1)
            self.link_bodies.append(body)
            self.space.add(body, shape)

        # Pivot anchors use each link's local lower/upper endpoint.
        first_joint = pymunk.PivotJoint(
            self.cart_body, self.link_bodies[0], (0.0, 0.0), (0.0, -self.segment_lengths[0] / 2)
        )
        self.space.add(first_joint)
        for index in range(self.num_segments - 1):
            joint = pymunk.PivotJoint(
                self.link_bodies[index],
                self.link_bodies[index + 1],
                (0.0, self.segment_lengths[index] / 2),
                (0.0, -self.segment_lengths[index + 1] / 2),
            )
            self.space.add(joint)

    def set_train_progress(self, t: int) -> None:
        """Set total training steps used by the reset-time gravity curriculum."""
        self._train_progress = int(t)

    def _get_obs(self) -> np.ndarray:
        values = [
            self.cart_body.position.x,
            self.cart_body.velocity.x,
            self.current_gravity / 9.81,
        ]
        for body in self.link_bodies:
            values.extend((math.cos(body.angle), math.sin(body.angle), body.angular_velocity))
        return np.asarray(values, dtype=np.float32)

    def _pendulum_tip_x(self) -> float:
        """Return the final pendulum endpoint x coordinate in world space."""
        tip_x = self.cart_body.position.x
        for body, length in zip(self.link_bodies, self.segment_lengths):
            tip_x -= length * math.sin(body.angle)
        return tip_x

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.curriculum_steps == 0:
            self.current_gravity = self.gravity
        else:
            progress = min(1.0, self._train_progress / self.curriculum_steps)
            gravity_max = self.gravity_start + (self.gravity - self.gravity_start) * progress
            self.current_gravity = float(self.np_random.uniform(0.0, gravity_max))
        self.space.gravity = (0.0, -self.current_gravity)

        self._set_rest_state()
        for _ in range(5):
            self.space.step(self.dt)
        # The solver's settling impulses can leave tiny velocities/position
        # errors. Restore the exact analytical equilibrium before step 0.
        self._set_rest_state()
        self._step_count = 0
        return self._get_obs(), {}

    def _set_rest_state(self) -> None:
        """Place every body at the exact vertical, zero-velocity equilibrium."""
        self.cart_body.position = (0.0, 0.0)
        self.cart_body.velocity = (0.0, 0.0)
        self.cart_body.force = (0.0, 0.0)
        cumulative = 0.0
        for body, length in zip(self.link_bodies, self.segment_lengths):
            body.position = (0.0, -cumulative - length / 2.0)
            body.angle = math.pi
            body.velocity = (0.0, 0.0)
            body.angular_velocity = 0.0
            body.force = (0.0, 0.0)
            cumulative += length

    def step(self, action: np.ndarray):
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        clipped_action = float(np.clip(action_array[0], -1.0, 1.0))
        if self.control_mode == "acceleration":
            total_mass = self.cart_mass + self.link_mass * self.num_segments
            force_x = clipped_action * self.max_acceleration * total_mass
            self.cart_body.force = (0.0, 0.0)
            self.cart_body.apply_force_at_world_point((force_x, 0.0), self.cart_body.position)
        else:
            self.cart_body.velocity = (clipped_action * self.max_velocity, 0.0)

        for _ in range(self.frame_skip):
            self.space.step(self.dt)
            for body in self.link_bodies:
                body.angular_velocity *= 1.0 - 0.002 * self.dt

            # GrooveJoint is authoritative; this prevents numerical drift past the rail.
            x = self.cart_body.position.x
            if x > self.slider_range:
                self.cart_body.position = (self.slider_range, 0.0)
                self.cart_body.velocity = (min(0.0, self.cart_body.velocity.x), 0.0)
            elif x < -self.slider_range:
                self.cart_body.position = (-self.slider_range, 0.0)
                self.cart_body.velocity = (max(0.0, self.cart_body.velocity.x), 0.0)

        hit_wall = abs(self.cart_body.position.x) >= self.slider_range - 1e-3
        angular_velocity_cost = sum(body.angular_velocity ** 2 for body in self.link_bodies)
        upright_reward = sum(math.cos(body.angle) for body in self.link_bodies)
        reference_error = self._pendulum_tip_x() - self.reference_x
        reward = (
            upright_reward
            - self.reference_reward_weight * reference_error ** 2
            - 0.001 * angular_velocity_cost
            - 0.0005 * self.cart_body.position.x ** 2
            - 0.0001 * clipped_action ** 2
            - (self.wall_penalty if hit_wall else 0.0)
        )
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        return self._get_obs(), float(reward), False, truncated, {"hit_wall": hit_wall}

    def close(self) -> None:
        self.space = pymunk.Space()
