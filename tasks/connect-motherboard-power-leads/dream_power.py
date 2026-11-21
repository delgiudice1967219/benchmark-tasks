"""Run the same workstation with the calibrated vision-and-joint policy API."""

import argparse
from pathlib import Path

from episode import ACTION_LIMIT, CAMERAS, INSTRUCTION, TASK_ID, Workstation
from dream_episode import run_episode


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    task = Workstation()
    result = run_episode(task.model, task.data, task.step, task.score, instruction=INSTRUCTION,
                         task_id=TASK_ID, action_limit=ACTION_LIMIT, out=args.out,
                         cameras=CAMERAS, dry_run=args.dry_run)
    if result["status"] != "complete":
        raise SystemExit(1)
