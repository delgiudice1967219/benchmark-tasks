"""Score frozen stock arm controls and optionally capture all four camera views."""

import argparse
from contextlib import ExitStack
from pathlib import Path

from review_media import StationViews, review_writer, save_setup

import numpy as np

from run import ACTION_LIMIT, Trial
from scene import CAMERAS, CASE_RESET_Y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    actions = np.load(args.actions, allow_pickle=False)
    if actions.ndim != 3 or actions.shape[1:] != (2,7) or len(actions) > ACTION_LIMIT:
        raise ValueError(f"Expected at most {ACTION_LIMIT:,} × 2 × 7 joint controls")
    if not np.isfinite(actions).all() or np.min(actions[:,:,6]) < 0 or np.max(actions[:,:,6]) > 1:
        raise ValueError("The recorded joint commands or gripper openings are not finite and valid")

    trial = Trial(on_side=True, with_paper=True, case_on_side=True, case_y=CASE_RESET_Y, measure=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if args.video:
            views = StationViews(trial.model, CAMERAS)
            stack.callback(views.close)
            initial_views = views.capture(trial.data)
            save_setup(initial_views, args.out/"setup.jpg")
            write_frame = stack.enter_context(review_writer(args.out/"scripted_demo.mp4"))
            write_frame(initial_views)
        for controls in actions:
            trial.step(controls,"frozen joint replay")
            if args.video:
                write_frame(views.capture(trial.data))
    trial.save(args.out,None)


if __name__ == "__main__":
    main()
