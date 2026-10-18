"""Command-line parsing and shared environment configuration."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D cart and passive pendulum")
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--lengths", type=float, nargs="+", default=None)
    parser.add_argument("--total-length", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--gravity-start", type=float, default=0.0)
    parser.add_argument("--curriculum-steps", type=int, default=0)
    parser.add_argument("--mode", choices=["acceleration", "velocity"], default="acceleration")
    parser.add_argument("--max-acceleration", type=float, default=20.0)
    parser.add_argument("--max-velocity", type=float, default=5.0)
    parser.add_argument("--slider-range", type=float, default=2.0)
    parser.add_argument("--wall-penalty", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--train-steps", type=int, default=300000)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--save-path", type=str, default="sac_pendulum")
    parser.add_argument("--load-path", type=str, default=None)
    return parser


def build_env_kwargs(args: argparse.Namespace) -> dict:
    if args.lengths is not None and len(args.lengths) != args.segments:
        raise ValueError("--lengths 的数量必须等于 --segments")
    if args.curriculum_steps > 0 and args.gravity_start > args.gravity:
        raise ValueError("--gravity-start 不能大于 --gravity")
    lengths = args.lengths
    if lengths is None:
        lengths = [args.total_length / args.segments] * args.segments
    return {
        "num_segments": args.segments,
        "segment_lengths": lengths,
        "gravity": args.gravity,
        "gravity_start": args.gravity_start,
        "curriculum_steps": args.curriculum_steps,
        "control_mode": args.mode,
        "max_acceleration": args.max_acceleration,
        "max_velocity": args.max_velocity,
        "slider_range": args.slider_range,
        "wall_penalty": args.wall_penalty,
        "max_episode_steps": args.max_episode_steps,
    }
