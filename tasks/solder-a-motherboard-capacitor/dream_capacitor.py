"""Run Dream-1 with the same reset, action path, and solder grader as the replay."""
import argparse
import json
from pathlib import Path

from dream_episode import run_episode
from run import ACTION_LIMIT, Trial
from scene import CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    task = json.loads(Path(__file__).with_name('task.json').read_text())
    trial = Trial()

    def step(action):
        trial.step(action)
        trial.material.camera_material()

    result = run_episode(trial.model, trial.data, step, trial.report,
                         instruction=task['instruction'], task_id='solder-a-motherboard-capacitor',
                         action_limit=8 if args.dry_run else ACTION_LIMIT,
                         out=args.out, cameras=CAMERAS, dry_run=args.dry_run)
    trial.save(args.out)
    print(json.dumps(result, indent=2))
    if result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
