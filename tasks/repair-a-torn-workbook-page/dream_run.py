"""Run an Inspect Robots policy on the workbook episode."""

import argparse
import json
from pathlib import Path

import numpy as np

from dream_episode import run_episode
from replay import Trial
from scene import ARMS, CAMERAS
from score import ACTION_LIMIT, score


INSTRUCTION = "Repair the torn workbook page and leave it turned over, with the print visible."
TASK_SLUG = "repair-a-torn-workbook-page"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=ACTION_LIMIT)
    args = parser.parse_args()
    if not 0 < args.limit <= ACTION_LIMIT:
        parser.error(f"Use 1 through {ACTION_LIMIT} actions.")
    if not args.dry_run and args.limit != ACTION_LIMIT:
        parser.error("Use the declared limit for a scored policy trial.")

    trial = Trial()

    def step(action):
        commands = trial.data.ctrl.copy()
        for side, targets in zip(ARMS, np.asarray(action).reshape(2, 7), strict=True):
            _, _, joints, fingers, _ = trial.arms[side]
            commands[joints] = targets[:6]
            commands[fingers] = .0475 * targets[6]
        trial.step(commands, "policy")

    def summary():
        return {**score(trial.trace), "actions": trial.steps}

    result = run_episode(
        trial.model, trial.data, step, summary, instruction=INSTRUCTION,
        task_id=TASK_SLUG, action_limit=args.limit, out=args.out,
        cameras=CAMERAS, dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
