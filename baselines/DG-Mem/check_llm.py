#!/usr/bin/env python3
"""Make a minimal connectivity call for an experiment LLM profile."""

from __future__ import annotations

import argparse

from src.config import ExperimentConfig
from src.llm import build_chat_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check configured LLM connectivity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--llm-profile", default=None)
    parser.add_argument("--role", choices=["agent", "judge", "both"], default="agent")
    parser.add_argument("--prompt", default="Reply with exactly OK")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = ExperimentConfig.load(args.config, profile=args.llm_profile)
    if experiment.llm_profile is None:
        raise ValueError("No LLM profile is selected in the config or --llm-profile")
    roles = ("agent", "judge") if args.role == "both" else (args.role,)
    for role in roles:
        settings = getattr(experiment.llm_profile, role)
        model = build_chat_model(settings)
        # Use the profile's configured completion budget. Reasoning models can
        # consume a small fixed budget before emitting any visible content.
        answer = model.complete([{"role": "user", "content": args.prompt}])
        if not answer:
            raise RuntimeError(f"{role} returned an empty response")
        print(f"{role}_ok provider={settings.provider} model={settings.model} response={answer!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
