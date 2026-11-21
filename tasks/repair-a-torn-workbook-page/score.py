"""Score a released, readable repair and a functional page turn at 20 Hz."""

import numpy as np

from scene import HZ, PAGE_DEPTH, TAPE_STEP, TEAR_START


ACTION_LIMIT = 680


def repaired(frame):
    if any(frame["liner_bonds"]) or max(frame["tear_gap_mm"]) > 2.0:
        return False
    if min(frame["tear_width_mm"]) < .10:
        return False
    segments = np.asarray(frame["tape_local"])
    # A repair bridges the edges continuously without presenting folded tape over the print.
    near_seam = bool(max(abs(segments[:, 0])) <= .0055)
    tidy = bool(min(segments[:, 4]) >= .91 and min(segments[:, 5]) >= .92
                and max(abs(segments[:, 2])) < .0025)
    start = segments[0, 1] - TAPE_STEP / 2
    end = segments[-1, 1] + TAPE_STEP / 2
    coverage = bool(start <= TEAR_START + .004 and PAGE_DEPTH - .006 <= end <= PAGE_DEPTH + .004)
    # The first segment begins on the intact sheet. The three subsequent segments span both flanks.
    anchors = all(all(frame["adhesion"][side][1:]) for side in ("left", "right"))
    return near_seam and tidy and coverage and anchors


def released(frame):
    return sum(frame["robot_page_contact_n"]) < .008


def single_flank_grasp(frame):
    left, right = frame["flank_contact_n"]
    return (left > .025 and right < .008) or (right > .025 and left < .008)


def score(trace):
    if not trace or len(trace) > ACTION_LIMIT:
        return {"score": 0., "patch": False, "turn": False, "release": False}
    intact = [repaired(frame) for frame in trace]
    motion = [f["angles_deg"]["turn_page"] for f in trace]
    flat_start = None
    for index in range(9, len(trace)):
        window = range(index - 9, index + 1)
        if all(intact[i] and released(trace[i]) and motion[i] < 15 for i in window):
            flat_start = index
            break
    final_window = range(max(0, len(trace) - 2 * HZ), len(trace))
    final_patch = len(final_window) == 2 * HZ and all(intact[i] for i in final_window)
    final_release = final_patch and all(released(trace[i]) for i in final_window)
    turn = False
    if flat_start is not None and final_release and all(motion[i] >= 165 for i in final_window):
        lifting = [i for i in range(flat_start + 1, len(trace))
                   if 45 < motion[i] < 120 and single_flank_grasp(trace[i])]
        if len(lifting) >= 3 and all(intact[flat_start:]):
            turn = True
    partial = flat_start is not None and final_release
    return {"score": 1. if turn else .40 if partial else 0.,
            "patch": partial, "turn": turn, "release": final_release}
