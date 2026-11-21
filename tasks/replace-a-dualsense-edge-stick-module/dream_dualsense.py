"""Evaluate Dream-1 using the same fresh controller scene and native scorer."""

import argparse
from io import StringIO
import json
from pathlib import Path

from run import ACTION_LIMIT, CAMERAS, HERE, Trial
from scene import Mechanism, build, mj

from dream_episode import run_episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--task-id",default="dualsense-stick-module")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true",help="Check camera and action formats offline")
    mode.add_argument("--attempt",action="store_true",help="Run the official policy against the service")
    args=parser.parse_args()
    task=json.loads((HERE/"task.json").read_text())
    model=mj.MjModel.from_xml_string(build("full"))
    data=mj.MjData(model)
    mechanism=Mechanism(model,data,"full")
    simulation=Trial(model,data,mechanism,StringIO())
    result=run_episode(
        model,data,simulation.step,mechanism.score,
        instruction=task["instruction"],task_id=args.task_id,
        action_limit=8 if args.dry_run else ACTION_LIMIT,out=args.out,
        cameras=CAMERAS,dry_run=args.dry_run,
    )
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result["status"]=="complete" else 1


if __name__=="__main__":
    raise SystemExit(main())
