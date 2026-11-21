"""Run Dream-1 on the same native contacts, heat model and scorer as replay."""

import argparse
import json
from pathlib import Path

from dream_episode import run_episode

import mujoco as mj

from run import ACTION_LIMIT, Runner
from splice import CAMERAS, Splice, build_xml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Check cameras and control without a service call")
    args = parser.parse_args()
    metadata = json.loads(Path(__file__).with_name("task.json").read_text())
    model = mj.MjModel.from_xml_string(build_xml())
    data = mj.MjData(model)
    physics = Splice(model, data, "solder")
    # The shared runner creates the output path after the robot and task reset.
    from io import StringIO
    trace = StringIO()
    trial = Runner(model, data, physics, trace)
    trial.phase = "policy"

    def step(action):
        trial.tick(action.reshape(2, 7))

    summary = run_episode(
        model, data, step, trial.report, instruction=metadata["instruction"],
        task_id="solder-robot-motor-lead", action_limit=8 if args.dry_run else ACTION_LIMIT,
        out=args.out, cameras=CAMERAS, dry_run=args.dry_run,
    )
    (args.out / "trace.jsonl").write_text(trace.getvalue())
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
