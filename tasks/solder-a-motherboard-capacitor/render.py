"""Render every completed control state on the fixed four-view review layout."""
import argparse
from pathlib import Path

from review_media import StationViews, review_writer, save_setup

import mujoco as mj
import numpy as np

from run import Trial
from scene import CAMERAS


def render(folder, trial):
    # A new scene keeps joint constraints and their solver state out of rendering.
    rendered = Trial(enabled=trial.material.enabled, staged=trial.material.staged)
    with np.load(folder/'states.npz', allow_pickle=False) as trace:
        states, wetted = trace['qpos'], trace['wetted']
    visuals = StationViews(rendered.model, CAMERAS)
    try:
        with review_writer(folder/'review.mp4') as write:
            for index, (qpos, fillets) in enumerate(zip(states, wetted, strict=True)):
                rendered.data.qpos[:] = qpos
                rendered.material.wetting[:] = fillets
                rendered.material.camera_material()
                mj.mj_forward(rendered.model, rendered.data)
                frame = visuals.capture(rendered.data)
                if index == 0:
                    save_setup(frame, folder/'setup.png')
                write(frame)
        save_setup(frame, folder/'final.png')
    finally:
        visuals.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    render(args.run, Trial())
