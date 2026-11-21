"""Run the dog mannequin task through the standard dual-YAM policy interface."""

import argparse
import json
from pathlib import Path

import numpy as np

from dream_episode import run_episode
from probe import ACTION_LIMIT, Episode
from scene import CAMERAS, SIDES


INSTRUCTION = ("Fit the harness on the mannequin so both front legs are through their own openings "
               "and the buckle is closed over its back. Leave the fitted mannequin standing on the "
               "table and release the harness.")
TASK_SLUG = "fit-a-step-in-harness-on-a-dog-mannequin"


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
    trial = Episode(with_latch=True, inward=True)

    def step(action):
        commands = trial.data.ctrl.copy()
        for side, targets in zip(SIDES, np.asarray(action).reshape(2, 7), strict=True):
            commands[trial.actuators[side]] = targets[:6]
            commands[trial.fingers[side]] = targets[6] * .0475
        trial.step(commands, "policy")

    result = run_episode(trial.model, trial.data, step, trial.result, instruction=INSTRUCTION,
                         task_id=TASK_SLUG, action_limit=args.limit, out=args.out,
                         cameras=CAMERAS, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
