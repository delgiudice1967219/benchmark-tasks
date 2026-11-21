"""Render verified joint targets with the standard station camera layout."""

import argparse
import json
from pathlib import Path

from review_media import StationViews, review_writer, save_setup

import mujoco as mj
import numpy as np

from scene import CAMERAS, DT, model_for_trial, reset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    summary = json.loads((directory / "summary.json").read_text())
    if summary["score"] != 1.0 or not summary["proof_pull"]["verified"]:
        raise ValueError("Media requires a verified full score")
    model = model_for_trial(with_latch=summary["harness_has_physical_latch"],
                            inward=summary["starts_with_buckle_halves_facing_inward"],
                            dual_adjust=summary["dual_adjust"])
    data = reset(model)
    action = np.load(directory / "actions.npy")
    views = StationViews(model, CAMERAS)
    try:
        save_setup(views.capture(data), directory / "setup.png")
        with review_writer(directory / "scripted-demo.mp4") as write_frame:
            for target in action:
                data.ctrl[:] = target
                mj.mj_step(model, data, nstep=round(1 / 20 / DT))
                write_frame(views.capture(data))
        save_setup(views.capture(data), directory / "ending.png")
    finally:
        views.close()


if __name__ == "__main__":
    main()
