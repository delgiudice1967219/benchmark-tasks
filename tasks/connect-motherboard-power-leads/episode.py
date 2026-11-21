"""Run a fixed native action tape on the motherboard workstation."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import mujoco as mj
import numpy as np

from review_media import StationViews, review_writer, save_setup
from scene import scene_xml
from scoring import RELEASE_HOLD, UnloadedLatchGrade
from terminal_model import use_internal_tin_terminals

CONTROL_HZ = 20
ACTION_LIMIT = 2100
ARMS = ("left", "right")
HOME = np.array([0., 1., 1., -.6, 0., 0., 1.])
LAYOUT_SHIFT = np.array((-.020, .065, 0.))
CAMERAS = {"top_cam": "top", "left_cam": "left", "right_cam": "right", "judging_front": "case_detail"}
INSTRUCTION = "Connect the power supply’s 24-pin main lead and paired 4+4-pin CPU lead to the motherboard. Leave both connectors fully seated and latched when you let go."
TASK_ID = "connect-motherboard-power-leads"


class Arm:
    def __init__(self, model, side):
        self.site = model.site(f"{side}_tcp").id
        self.joint_names = [f"{side}_joint{i}" for i in range(1, 7)]
        self.address = [model.joint(name).qposadr[0] for name in self.joint_names]
        self.actuators = [model.actuator(name).id for name in self.joint_names]
        self.finger_names = [f"{side}_finger_{i}" for i in (-1, 1)]
        self.fingers = [model.actuator(name).id for name in self.finger_names]

    def command(self, data, action):
        data.ctrl[self.actuators] = action[:6]
        data.ctrl[self.fingers] = action[6] * .0475

    def reset(self, data):
        data.qpos[self.address] = HOME[:6]
        for name in self.finger_names:
            data.joint(name).qpos[:] = .0475
        self.command(data, HOME)


class Workstation:
    def __init__(self):
        root = ET.fromstring(scene_xml(layout_shift=LAYOUT_SHIFT, open_bench=True))
        root.find("option").attrib.update(timestep=".001", integrator="implicit")
        use_internal_tin_terminals(root)
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mj.MjData(self.model)
        self.arms = {side: Arm(self.model, side) for side in ARMS}
        for arm in self.arms.values():
            arm.reset(self.data)
        mj.mj_forward(self.model, self.data)
        self.grade = UnloadedLatchGrade(self.model, self.data, self.arms)
        self.substeps = round(1 / (CONTROL_HZ * self.model.opt.timestep))
        self.steps = 0
        self.maximum_joint_effort = np.zeros((2, 6))
        self.maximum_cross_arm_load = 0.
        self.geom_arm = np.zeros(self.model.ngeom, dtype=np.int8)
        for geom in range(self.model.ngeom):
            name = self.model.geom(geom).name
            if name.startswith("left_"):
                self.geom_arm[geom] = 1
            elif name.startswith("right_"):
                self.geom_arm[geom] = 2
        self.contact_wrench = np.empty(6)

    def step(self, action):
        if self.steps >= ACTION_LIMIT:
            raise RuntimeError("Episode exceeded the action limit")
        action = np.asarray(action).reshape(len(ARMS), 7)
        for arm, command in zip(self.arms.values(), action):
            arm.command(self.data, command)
        for _ in range(self.substeps):
            mj.mj_step(self.model, self.data)
            self.grade.update()
            for index, arm in enumerate(self.arms.values()):
                effort = abs(self.data.actuator_force[arm.actuators])
                np.maximum(self.maximum_joint_effort[index], effort, out=self.maximum_joint_effort[index])
            load = 0.
            for slot, (first, second) in enumerate(self.data.contact.geom):
                if min(first, second) >= 0 and self.geom_arm[first] * self.geom_arm[second] == 2:
                    mj.mj_contactForce(self.model, self.data, slot, self.contact_wrench)
                    load += self.contact_wrench[0]
            self.maximum_cross_arm_load = max(self.maximum_cross_arm_load, load)
        self.steps += 1

    def score(self):
        proof = self.grade.proof()
        values = [.50 if result["passed"] else .25 if hold >= RELEASE_HOLD else 0.
                  for result, hold in zip(proof, self.grade.partial)]
        report = self.grade.report()
        report.update(score=sum(values), connector_scores=values, three_newton_checks=proof,
                      control_hz=CONTROL_HZ, actions=self.steps, action_limit=ACTION_LIMIT,
                      peak_joint_effort_nm=self.maximum_joint_effort.tolist(),
                      peak_arm_to_arm_contact_n=self.maximum_cross_arm_load)
        return report


def replay(commands, out, record):
    workstation = Workstation()
    with np.load(commands, allow_pickle=False) as file:
        actions = file["actions"]
    if actions.ndim != 3 or actions.shape[1:] != (len(ARMS), 7) or not np.isfinite(actions).all():
        raise ValueError("Commands must be finite 2 by 7 joint targets")
    out.mkdir(parents=True)
    with ExitStack() as stack:
        if record:
            views = StationViews(workstation.model, CAMERAS)
            stack.callback(views.close)
            video = stack.enter_context(review_writer(out / "demo.mp4"))
            save_setup(views.capture(workstation.data), out / "setup.png")
        trace = stack.enter_context((out / "trace.jsonl").open("w"))
        for action in actions:
            workstation.step(action)
            trace.write(json.dumps(workstation.grade.report(), allow_nan=False) + "\n")
            if record:
                video(views.capture(workstation.data))
        if record:
            save_setup(views.capture(workstation.data), out / "final.png")
    result = workstation.score()
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--commands", type=Path, default=Path(__file__).with_name("commands.npz"))
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--record", action="store_true")
    options = cli.parse_args()
    replay(options.commands, options.out, options.record)
