"""Launch the Inspect Robots policy against a fresh, scored YAM ribbon trial."""

import argparse
import json
from pathlib import Path

from dream_episode import run_episode
from episode import ACTION_LIMIT, INSTRUCTION, TASK_SLUG, RibbonTrial
from scene import CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Do not make a model service request.')
    parser.add_argument('--limit', type=int, default=ACTION_LIMIT)
    options = parser.parse_args()
    if options.limit <= 0 or options.limit > ACTION_LIMIT:
        parser.error(f'--limit must be from 1 through {ACTION_LIMIT}')
    if options.limit != ACTION_LIMIT and not options.dry_run:
        parser.error('Use the declared action limit for the recorded model trial.')
    trial = RibbonTrial(art=True)
    report = run_episode(trial.model, trial.data, trial.step, trial.summary,
                         instruction=INSTRUCTION, task_id=TASK_SLUG, action_limit=options.limit,
                         out=options.out, cameras=CAMERAS, dry_run=options.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
