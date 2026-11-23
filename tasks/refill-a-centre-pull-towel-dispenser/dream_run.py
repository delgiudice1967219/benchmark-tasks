"""Use the same scored native episode for a Dream-1 evaluation."""

import argparse
import json
from pathlib import Path

from dream_episode import run_episode

import numpy as np

from run import ACTION_LIMIT, Trial
from scene import CAMERAS, CASE_RESET_Y


TASK_SLUG = "refill-a-centre-pull-towel-dispenser"


def main():
    task = json.loads(Path(__file__).with_name("task.json").read_text())
    limit = task["action_limit"]
    if limit != ACTION_LIMIT:
        raise ValueError("The packaged action limit differs from the scene action limit.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scripted-evidence", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=limit)
    args = parser.parse_args()
    if not 0 < args.limit <= limit:
        parser.error(f"Use 1 to {limit:,} actions.")
    if not args.dry_run:
        if args.limit != limit:
            parser.error(f"A scored Dream-1 trial uses all {limit:,} actions.")
        if args.scripted_evidence is None:
            parser.error("Point to the approved frozen scripted replay evidence.")
        summary = json.loads((args.scripted_evidence/"summary.json").read_text())
        movie = args.scripted_evidence/"scripted_demo.mp4"
        if summary["score"] != 1. or not movie.is_file() or not movie.stat().st_size:
            parser.error("Dream-1 requires a reviewed, full-score scripted video.")

    trial = Trial(on_side=True, with_paper=True, case_on_side=True, case_y=CASE_RESET_Y, measure=True)

    def step(action):
        trial.step(np.asarray(action, dtype=float).reshape((2,7)), "policy joint action")

    def score():
        return trial.scoring.observe(trial.data)

    result = run_episode(trial.model, trial.data, step, score, instruction=task["instruction"],
                         task_id=TASK_SLUG, action_limit=args.limit, out=args.out,
                         cameras=CAMERAS, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
