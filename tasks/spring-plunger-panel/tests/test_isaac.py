"""In Isaac Sim: the scripted demo reaches full score, and the three controls score 0.

Runs ``demo.py`` once for all four policies (Isaac Sim starts once). Needs an RTX GPU and the Isaac Lab-Arena
environment of the simulation walkthrough; skipped without it. Select or skip it with ``-m isaac`` / ``-m "not isaac"``.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaac
HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if importlib.util.find_spec("isaaclab_arena") is None:
        pytest.skip("needs the Isaac Lab-Arena environment")
    out = tmp_path_factory.mktemp("isaac")
    subprocess.run([sys.executable, str(HERE / "demo.py"), "--policy", "demo", "hold", "pull_only", "early_release",
                    "--no-video", "--out", str(out)], check=True, cwd=HERE, timeout=3600)
    return {r["policy"]: r for r in json.loads((out / "result.json").read_text())}


def test_demo_reaches_full_score_without_forcing(results):
    demo = results["demo"]
    assert demo["success"] and demo["task_score"] == 1.0
    assert not demo["forced"] and demo["worst_pin_overlap_mm"] < 1.0
    assert demo["steps"] <= 600 and demo["max_plunger_travel_mm"] >= 12.0


def _check_control(results, policy):
    control = results[policy]
    assert control["task_score"] == 0.0 and not control["success"]
    assert control["steps"] == 600  # a whole trial
    assert control["max_angle_deg"] < 2.0


def test_hold_scores_0(results):
    _check_control(results, "hold")


def test_pulling_the_latched_panel_scores_0(results):
    _check_control(results, "pull_only")
    assert results["pull_only"]["max_plunger_travel_mm"] < 0.5  # the plunger was never touched


def test_letting_go_of_the_knob_early_scores_0(results):
    _check_control(results, "early_release")
    assert results["early_release"]["max_plunger_travel_mm"] >= 12.0  # the pin was fully back before the release
