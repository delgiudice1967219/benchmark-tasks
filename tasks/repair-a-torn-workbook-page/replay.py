"""Replay stock YAM pose commands and evaluate contact-driven paper repair."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scene import (ARMS, BOOK_X, CAMERAS, DT, HOME, HZ, PAGE_DEPTH, PAPER_Z,
                   SPINE_Y, TAPE_COUNT, TAPE_STEP, TAPE_TIP, TAPE_X, TAPE_Z,
                   TEAR_START, build_model, reset)
from score import score


def attitude(hand, angle=0., turn=0, vertical=False):
    sign = 1 if hand == "left" else -1
    if vertical:
        forward = np.array((0., 0., -1.))
        pinch = np.array((np.cos(angle), np.sin(angle), 0.))
    else:
        outward = np.array((sign * np.cos(angle), np.sin(angle), .0))
        forward = -outward
        pinch = np.array((0., 0., 1.))
    matrix = np.column_stack((np.cross(pinch, forward), pinch, forward))
    return Rotation.from_rotvec(np.array((turn, 0., 0.))).as_matrix() @ matrix


class Trial:
    def __init__(self, frame=None):
        self.model = build_model()
        self.data = reset(self.model)
        self.scratch = mj.MjData(self.model)
        self.frame = frame
        self.arms = {}
        for side in ARMS:
            joints = [self.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
            fingers = [self.model.actuator(f"{side}_finger_{i}").id for i in (-1, 1)]
            actuators = [self.model.actuator(f"{side}_joint{i}").id for i in range(1, 7)]
            self.arms[side] = (self.model.jnt_qposadr[joints], self.model.jnt_range[joints],
                               np.array(actuators), np.array(fingers), self.model.site(f"{side}_tcp").id)
        self.names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        self.adhesive = {(i, side): self.model.equality(f"patch_{i}_{side}").id
                         for i in range(TAPE_COUNT) for side in ARMS}
        self.wrenches = np.zeros(6)
        self.liner_clock = np.zeros(TAPE_COUNT - 1)
        self.liner_force = np.zeros_like(self.liner_clock)
        self.liner = [self.model.equality(f"release_bond_{i}").id for i in range(TAPE_COUNT - 1)]
        self.clock = {key: 0. for key in self.adhesive}
        self.pressure_peak = {key: 0. for key in self.adhesive}
        self.hand_contact = np.zeros((2, 3))
        self.robot_page_contact = np.zeros(2)
        self.flank_contact = np.zeros(2)
        self.steps = 0
        self.trace = []
        self.actions = []

    def solve(self, side, point, orientation, seed):
        indices, ranges, _, _, tcp = self.arms[side]
        self.scratch.qpos[:] = self.data.qpos

        def residual(q):
            self.scratch.qpos[indices] = q
            mj.mj_kinematics(self.model, self.scratch)
            matrix = self.scratch.site_xmat[tcp].reshape(3, 3)
            point_error = (self.scratch.site_xpos[tcp] - point) * 65
            angle_error = Rotation.from_matrix(orientation.T @ matrix).as_rotvec()
            return np.r_[point_error, angle_error, .001 * (q - seed)]

        lo, hi = ranges.T
        result = least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi),
                               max_nfev=90, ftol=1e-8, gtol=1e-8, xtol=1e-8)
        err = np.linalg.norm(result.fun[:3]) / 65
        if err > .0025 or np.linalg.norm(result.fun[3:6]) > .04:
            fresh = least_squares(residual, np.clip(HOME, lo, hi), bounds=(lo, hi), max_nfev=110)
            if np.linalg.norm(fresh.fun[:6]) < np.linalg.norm(result.fun[:6]):
                result = fresh
                err = np.linalg.norm(result.fun[:3]) / 65
        return result.x, float(err)

    def bond_here(self, bond, location):
        model, data = self.model, self.data
        first, second = model.eq_obj1id[bond], model.eq_obj2id[bond]
        for index, body in enumerate((first, second)):
            rotation = data.xmat[body].reshape(3, 3)
            model.eq_data[bond, 3 * index:3 * (index + 1)] = rotation.T @ (location - data.xpos[body])
        data.eq_active[bond] = True

    def material_step(self):
        model, data = self.model, self.data
        equality = data.efc_type[:data.nefc] == mj.mjtConstraint.mjCNSTR_EQUALITY
        for i, bond in enumerate(self.liner):
            if not data.eq_active[bond]:
                continue
            next_segment = data.body(f"tissue_{i + 1}")
            # Only tensile contact along the removal side of the backing releases the active peel front.
            front = i == len(self.liner) - 1 or not data.eq_active[self.liner[i + 1]]
            relevant = equality & (data.efc_id[:data.nefc] == bond)
            force = np.linalg.norm(data.efc_force[:data.nefc][relevant])
            self.liner_force[i] = max(self.liner_force[i], force)
            lifted = next_segment.xmat.reshape(3, 3)[2, 1] > .065
            if front and force > .033 and lifted:
                self.liner_clock[i] += DT
            else:
                self.liner_clock[i] = 0.
            if self.liner_clock[i] >= .004:
                data.eq_active[bond] = False
        self.hand_contact.fill(0.)
        self.robot_page_contact.fill(0.)
        self.flank_contact.fill(0.)
        if not data.ncon:
            return
        region = {}
        pressure = False
        for contact_id, (a, b) in enumerate(data.contact.geom):
            if a < 0 or b < 0:
                continue
            ga, gb = self.names[a], self.names[b]
            if ga.startswith(("left_", "right_")) and gb.startswith(("flap_", "bound_page", "adhesive_")):
                self.page_contact(ga, gb, contact_id)
            elif gb.startswith(("left_", "right_")) and ga.startswith(("flap_", "bound_page", "adhesive_")):
                self.page_contact(gb, ga, contact_id)
            tape_first = gb.startswith("adhesive_") and not ga.startswith("adhesive_")
            if "_tip_" in gb or (tape_first and "_tip_" not in ga):
                ga, gb = gb, ga
            if "_tip_" in ga:
                hand = int(ga.startswith("right"))
                category = (0 if gb.startswith("adhesive_") else
                            1 if gb.startswith(("flap_", "bound_page")) else 2)
                mj.mj_contactForce(model, data, contact_id, self.wrenches)
                self.hand_contact[hand, category] += max(0., self.wrenches[0])
                pressure |= category == 0 and self.wrenches[0] > .014
            elif ga.startswith("adhesive_") and gb.startswith("flap_"):
                _, i, side = ga.split("_")
                if gb.startswith(f"flap_{side}_"):
                    key = int(i), side
                    mj.mj_contactForce(model, data, contact_id, self.wrenches)
                    normal = max(0., float(self.wrenches[0]))
                    point = data.contact.pos[contact_id].copy()
                    along = (point[0] - BOOK_X) * (1 if side == "left" else -1)
                    if .001 < along < .016 and (key not in region or normal > region[key][0]):
                        region[key] = (normal, point)
        if any(data.eq_active[self.liner]):
            return
        for key, eq in self.adhesive.items():
            if data.eq_active[eq]:
                continue
            measurement = region.get(key)
            if measurement is None or not pressure:
                self.clock[key] = max(0., self.clock[key] - DT * .005)
                continue
            normal, contact = measurement
            i, side = key
            self.pressure_peak[key] = max(self.pressure_peak[key], normal)
            segment = data.body(f"tissue_{i}")
            upright = segment.xmat.reshape(3, 3)[2, 2] > .89
            impulse = self.clock[key] + DT * (normal - .003)
            self.clock[key] = max(0., impulse) if upright else 0.
            if self.clock[key] >= .00032:
                self.bond_here(eq, contact)

    def page_contact(self, robot, object_name, contact_id):
        mj.mj_contactForce(self.model, self.data, contact_id, self.wrenches)
        load = max(0., self.wrenches[0])
        self.robot_page_contact[int(robot.startswith("right"))] += load
        if object_name.startswith(("flap_left_", "flap_right_")):
            self.flank_contact[int(object_name.startswith("flap_right_"))] += load

    def sample(self):
        d = self.data
        tape = []
        local = []
        page = d.body("bound_page")
        frame = page.xmat.reshape(3, 3)
        flap_origin = d.body("flap_left").xpos
        bend = sum(float(d.joint(f"flap_hinge_{side}").qpos[0]) for side in ARMS) / 2
        surface = frame @ Rotation.from_rotvec((bend, 0., 0.)).as_matrix()
        for i in range(TAPE_COUNT):
            body = d.body(f"tissue_{i}")
            rotation = body.xmat.reshape(3, 3)
            centre = body.xpos + rotation @ np.array((0., TAPE_STEP / 2, 0.))
            tape.append(centre.tolist())
            relative = surface.T @ (centre - flap_origin) + (0., TEAR_START, 0.)
            local.append([*relative, float(rotation[:, 0] @ surface[:, 0]),
                          float(rotation[:, 2] @ surface[:, 2]), float(rotation[:, 1] @ surface[:, 1])])
        margins = []
        widths = []
        for i in range(8):
            a = d.site(f"tear_left_{i}").xpos
            b = d.site(f"tear_right_{i}").xpos
            margins.append(float(np.linalg.norm(a - b)))
            widths.append(float((a - b) @ frame[:, 0]))
        adherence = {side: [bool(d.eq_active[self.adhesive[(i, side)]]) for i in range(TAPE_COUNT)]
                     for side in ARMS}
        angles = {name: float(np.rad2deg(d.joint(name).qpos)[0])
                  for name in ("turn_page", "flap_hinge_left", "flap_hinge_right")}
        return {
            "tape_xyz": tape,
            "tape_local": local,
            "tape_axis": [d.body(f"tissue_{i}").xmat.reshape(3, 3)[:, 1].tolist() for i in range(TAPE_COUNT)],
            "liner_bonds": [bool(d.eq_active[i]) for i in self.liner],
            "adhesion": adherence,
            "tear_gap_mm": [gap * 1000 for gap in margins],
            "tear_width_mm": [width * 1000 for width in widths],
            "angles_deg": angles,
            "contact_n": self.hand_contact.tolist(),
            "robot_page_contact_n": self.robot_page_contact.tolist(),
            "flank_contact_n": self.flank_contact.tolist(),
            "tcp_xyz": {side: d.site(f"{side}_tcp").xpos.tolist() for side in ARMS},
        }

    def step(self, commands, label):
        self.data.ctrl[:] = commands
        for _ in range(round(1 / HZ / DT)):
            mj.mj_step(self.model, self.data)
            self.material_step()
        self.steps += 1
        if abs(self.data.time - self.steps / HZ) > 1e-7:
            raise RuntimeError(f"Simulation was unstable at step {self.steps} ({label})")
        self.actions.append(commands.copy())
        report = self.sample()
        self.trace.append({"stage": label, **report})
        if self.frame is not None:
            self.frame(self.model, self.data)

    def phase(self, label, seconds, goals=(), openings=()):
        target = self.data.ctrl.copy()
        for side, xyz, rotation in goals:
            actuators = self.arms[side][2]
            angles, error = self.solve(side, xyz, rotation, target[actuators])
            target[actuators] = angles
            if error > .0015:
                print(f"{label} {side} IK error {error * 1000:.2f} mm", flush=True)
        for side, opening in openings:
            target[self.arms[side][3]] = opening / 2
        previous = self.data.ctrl.copy()
        steps = round(seconds * HZ)
        for index in range(1, steps + 1):
            t = index / steps
            self.step(previous + (target - previous) * t * t * (3 - 2 * t), label)
        last = self.trace[-1]
        print(label, np.array(last["tape_xyz"]).round(4).tolist(),
              last["liner_bonds"], last["adhesion"],
              np.array(last["contact_n"]).round(3).tolist(),
              np.array(list(last["angles_deg"].values())).round(2).tolist(), flush=True)


def peel(trial, no_grip=False):
    pose = attitude("right", np.pi / 2)
    grip = np.array((TAPE_X, TAPE_TIP + .005, TAPE_Z + .001))
    trial.phase("settle", .45)
    trial.phase("approach free tissue edge", 2.1,
                (("right", grip + (0., .028, .048), pose),), (("right", .013),))
    trial.phase("descend to free tissue edge", 1.2, (("right", grip + (0., .019, 0.), pose),))
    trial.phase("slide in open fingers", .8, (("right", grip, pose),))
    trial.phase("pinch the free tissue end", .5, openings=(("right", .013 if no_grip else .0),))
    for dx, dy, dz in ((0., -.001, .012), (0., -.003, .024), (0., -.005, .045), (0., -.008, .063)):
        trial.phase("peel tissue upward", .55, (("right", grip + (dx, dy, dz), pose),))
    trial.phase("hold peeled tissue", .30)
    return grip, pose


def flatten_flap(trial):
    pad = np.array((BOOK_X + .091, SPINE_Y + PAGE_DEPTH - .019, PAPER_Z + .0107))
    pose = Rotation.from_rotvec((0., .48, 0.)).as_matrix() @ attitude("left", np.pi / 2, vertical=True)
    trial.phase("approach the curled tear flank", 1.35,
                (("left", pad + (0., 0., .045), pose),), (("left", .012),))
    trial.phase("align the curled paper with the other flank", .85, (("left", pad, pose),))
    return pad, pose


def place_tissue(trial, one_flank=False, skew=False):
    pad, pad_rotation = flatten_flap(trial)
    out = np.array((BOOK_X + (.013 if one_flank else 0), SPINE_Y + PAGE_DEPTH + .005, PAPER_Z))
    pose = attitude("right", np.pi / 2)
    tilted = attitude("right", np.pi / 2, turn=np.deg2rad(15))
    if skew:
        tilted = Rotation.from_rotvec((0., 0., .34)).as_matrix() @ tilted
    trial.phase("carry the tissue over the seam", 1.65, (("right", out + (0., .002, .055), pose),))
    trial.phase("lower the loose end to the aligned tear", 1.3,
                (("right", out + (0., -.001, .011), tilted),))
    trial.phase("press the loose end against the paper", .95,
                (("right", out + (-.003, -.004, .0055), tilted),))
    trial.phase("release repair tissue in place", .30, openings=(("right", .013),))
    trial.phase("retract the fingers from the edge", .50,
                (("right", out + (-.003, .035, .0065), tilted),))
    trial.phase("lift clear before reorienting fingers", .52,
                (("right", out + (-.055, .040, .090), tilted),))
    burnish_pose = attitude("right", 0., vertical=True)
    trial.phase("rotate the fingers above the seam", .70,
                (("right", out + (-.055, -.026, .095), burnish_pose),), (("right", .006),))
    trial.phase("bring the press to the inner tip of the tissue", .75,
                (("right", out + (0., -.067, .030), burnish_pose),))
    for x, y in ((0., -.054), (0., -.036), (0., -.018), (.004, -.054), (.004, -.036), (.008, -.019)):
        trial.phase("burnish the tape into the paper", .50,
                    (("right", out + (x, y, .0091 if x > .007 else .00965), burnish_pose),))
        trial.phase("hold pressure", .23)
        trial.phase("lift off after each press", .24,
                    (("right", out + (x, y, .018), burnish_pose),))
    trial.phase("retract pressing fingers from the book", .75,
                (("right", out + (-.116, .033, .090), burnish_pose),))
    trial.phase("release the supporting fingers", .55,
                (("left", pad + (.010, .010, .065), pad_rotation),))
    trial.phase("inspect the released repair before turning", .60)


def turn_page(trial, hold=False):
    turn_arm = "right"
    pivot = np.array((BOOK_X - .063, SPINE_Y, PAPER_Z))
    initial = np.array((0., PAGE_DEPTH + .002, .0009))
    pose = attitude(turn_arm, np.pi / 2)
    edge = pivot + initial
    trial.phase("approach the paper's free edge", 1.30,
                ((turn_arm, edge + (0., .028, .034), pose),), ((turn_arm, .014),))
    trial.phase("level the jaws beside the free edge", .75,
                ((turn_arm, edge + (0., .019, 0.), pose),))
    trial.phase("insert two jaws around the free edge", .80, ((turn_arm, edge, pose),))
    trial.phase("pinch only one flank of the page", .45, openings=((turn_arm, 0.),))
    waypoints = (9, 25, 43, 61) if hold else (9, 25, 43, 61, 78, 94, 110, 126)
    for degrees in waypoints:
        angle = np.deg2rad(degrees)
        rotation = Rotation.from_rotvec((angle, 0., 0.)).as_matrix()
        trial.phase(f"turn the page past {degrees} degrees", .38,
                    ((turn_arm, pivot + rotation @ initial, rotation @ pose),))
    if hold:
        trial.phase("continue gripping the page", 2.2)
        return
    trial.phase("release the page over center", .25, openings=((turn_arm, .019),))
    trial.phase("lift the fingers clear of the released page", .75,
                ((turn_arm, pivot + (-.086, -.070, .17), attitude(turn_arm, np.pi / 2, vertical=True)),))
    trial.phase("observe the released page at rest", 2.15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--no-grip", action="store_true")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--one-flank", action="store_true")
    placement.add_argument("--skew", action="store_true")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--turn-probe", action="store_true")
    parser.add_argument("--peel-only", action="store_true")
    parser.add_argument("--place-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        frame = None
        if args.video or args.setup:
            from review_media import StationViews, review_writer, save_setup
            video = stack.enter_context(review_writer(args.video)) if args.video else None
            media = None

            def frame(model, data):
                nonlocal media
                if media is None:
                    media = StationViews(model, CAMERAS)
                    stack.callback(media.close)
                    if args.setup:
                        views = media.capture(data)
                        save_setup(views, args.setup)
                        if video:
                            video(views)
                        return
                if video:
                    video(media.capture(data))

        trial = Trial(frame)
        if not args.turn_probe:
            peel(trial, args.no_grip)
            if not args.peel_only:
                place_tissue(trial, args.one_flank, args.skew)
        if not args.peel_only and not args.place_only:
            turn_page(trial, args.hold)
        elif args.place_only:
            trial.phase("confirm the flat released patch remains secure", 1.50)
    summary = {"actions": trial.steps, **score(trial.trace), "final": trial.trace[-1],
               "peak_liner_tension_n": trial.liner_force.tolist(),
               "peak_adhesive_contact_n": {str(key): val for key, val in trial.pressure_peak.items()}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "trace.json").write_text(json.dumps(trial.trace, separators=(",", ":")) + "\n")
    np.save(args.out / "actions.npy", np.stack(trial.actions))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
