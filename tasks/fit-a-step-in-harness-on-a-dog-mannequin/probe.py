"""Measure native mannequin grasp and two-paw passage on the stock station."""

import argparse
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scene import DT, HOME, SIDES, TABLE_Z, model_for_trial, reset, stance


HZ = 20
ACTION_LIMIT = 1000


def in_polygon(point, polygon):
    """Ray crossing in the laid-flat harness location."""
    inside = False
    x, y = point
    for (xa, ya), (xb, yb) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (ya > y) != (yb > y) and x < xa + (xb - xa) * (y - ya) / (yb - ya):
            inside = not inside
    return inside


class Episode:
    def __init__(self, *, with_latch=False, inward=False, dual_adjust=False):
        self.with_latch = with_latch
        self.inward = inward
        self.model = model_for_trial(with_latch=with_latch, inward=inward, dual_adjust=dual_adjust)
        self.data = reset(self.model)
        self.scratch = mj.MjData(self.model)
        self.joints, self.actuators, self.fingers, self.bounds = {}, {}, {}, {}
        for side in SIDES:
            names = [f"{side}_joint{i}" for i in range(1, 7)]
            ids = [self.model.joint(name).id for name in names]
            self.joints[side] = self.model.jnt_qposadr[ids]
            self.bounds[side] = self.model.jnt_range[ids].T
            self.actuators[side] = [self.model.actuator(name).id for name in names]
            self.fingers[side] = [self.model.actuator(f"{side}_finger_{sign}").id for sign in (-1, 1)]
        self.touch = np.zeros(6)
        names = tuple(self.model.geom(i).name for i in range(self.model.ngeom))
        self.robot_geometry = np.array([name.startswith(("left_", "right_")) and "_contact_" in name
                                        for name in names])
        self.subject_geometry = np.array([
            name.endswith(("web", "shell"))
            or name in {"chest", "neck", "head", "haunches", "muzzle", "tail"}
            or any(token in name for token in ("_fore_", "_hind_", "_ear"))
            for name in names
        ])
        self.trace = []
        self.actions = []
        self.entered = {side: False for side in SIDES}
        self.previous_clear = {side: False for side in SIDES}

    def ik(self, side, xyz, rot):
        self.scratch.qpos[:] = self.data.qpos
        start = self.data.ctrl[self.actuators[side]]
        bounds = self.bounds[side]

        def residual(angles):
            self.scratch.qpos[self.joints[side]] = angles
            mj.mj_kinematics(self.model, self.scratch)
            tcp = self.scratch.site(f"{side}_tcp")
            heading = Rotation.from_matrix(rot.T @ tcp.xmat.reshape(3, 3)).as_rotvec()
            return np.r_[(tcp.xpos - xyz) * 45, heading, (angles - start) * .004]

        candidates = [least_squares(residual, q, bounds=bounds, max_nfev=60)
                      for q in (np.clip(start, *bounds), np.clip(HOME, *bounds))]
        best = min(candidates, key=lambda choice: np.linalg.norm(choice.fun[:6]))
        return best.x, np.linalg.norm(best.fun[:3]) / 45

    def phase(self, name, steps, *, goals=(), gaps=()):
        seed = self.data.ctrl.copy()
        end = seed.copy()
        for side, xyz, attitude in goals:
            angles, error = self.ik(side, xyz, attitude)
            print(f"{name} {side} IK {1000*error:.2f} mm", flush=True)
            end[self.actuators[side]] = angles
        for side, gap in gaps:
            end[self.fingers[side]] = gap / 2
        for index in range(1, steps + 1):
            phase = index / steps
            self.step(seed + (end - seed) * (phase**2 * (3 - 2 * phase)), name)
        print(name, self.trace[-1]["body_mm"], self.trace[-1]["pad_loads"], flush=True)

    def pad_loads(self):
        loads = {side: dict(mannequin=0., harness=0., table=0., other=0.) for side in SIDES}
        for index, (a, b) in enumerate(self.data.contact.geom):
            names = tuple(self.model.geom(int(item)).name for item in (a, b))
            for finger, target in (names, names[::-1]):
                for side in SIDES:
                    if finger.startswith(f"{side}_tip_") and "_contact_" in finger:
                        destination = ("table" if target == "table" else
                                       "harness" if target.endswith(("web", "shell")) else
                                       "mannequin" if target in {"chest", "neck", "head", "haunches", "muzzle", "tail"}
                                       or any(token in target for token in ("_fore_", "_hind_", "_ear"))
                                       else "other")
                        mj.mj_contactForce(self.model, self.data, index, self.touch)
                        loads[side][destination] += max(0., float(self.touch[0]))
        return {side: {k: round(v, 4) for k, v in values.items()} for side, values in loads.items()}

    def contour(self, side, data):
        front = [data.site(f"{side}_front_seg_{i}_origin").xpos.copy() for i in range(5)]
        rear = [data.site(f"{side}_rear_seg_{i}_origin").xpos.copy() for i in range(5)]
        end = data.site(f"{side}_buckle_end").xpos.copy()
        return np.array(front + [end] + rear[::-1])

    def limb_threaded(self, side, outline, data):
        """Check the paw, hinge and chest axis against a surface spanning the loop."""
        toe = data.site(f"{side}_fore_sole").xpos
        knee = data.body(f"{side}_fore_lower").xpos
        shoulder = data.body(f"{side}_fore_thigh").xpos
        chest = data.body("mannequin")
        frame = chest.xmat.reshape(3, 3)
        offset = .008 if side == "left" else -.008
        core = chest.xpos + frame @ np.array((offset, -.085, .025))
        pelvis = chest.xpos + frame @ np.array((offset, .125, .025))
        origin = outline.mean(axis=0)
        a = outline - origin
        b = np.roll(outline, -1, axis=0) - origin
        count = 0
        for start, finish in zip((toe, knee, shoulder, core), (knee, shoulder, core, pelvis)):
            segment = finish - start
            scale = np.einsum("ij,ij->i", a, np.cross(segment, b))
            valid = np.abs(scale) > 1e-10
            rel = start - origin
            cross = np.cross(rel, a[valid])
            u = np.cross(segment, b[valid]) @ rel / scale[valid]
            v = cross @ segment / scale[valid]
            t = np.einsum("ij,ij->i", b[valid], cross) / scale[valid]
            crossings = np.unique(t[(u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t <= 1)].round(8))
            count += len(crossings)
        return bool(count % 2)

    def buckle_alignment(self):
        female = self.data.body("left_buckle")
        male = self.data.body("right_buckle")
        frame = female.xmat.reshape(3, 3)
        axis = male.xmat.reshape(3, 3)[:, 0]
        return {
            "male_in_female_mm": ((male.xpos - female.xpos) @ frame * 1000).round(3).tolist(),
            "axes_error_deg": round(math.degrees(math.acos(np.clip(-axis @ frame[:, 0], -1, 1))), 3),
        }

    def robot_touching_subject(self, data):
        for contact in data.contact[:data.ncon]:
            a, b = contact.geom
            if min(a, b) >= 0 and contact.dist <= 0 and (
                self.robot_geometry[a] and self.subject_geometry[b]
                or self.robot_geometry[b] and self.subject_geometry[a]
            ):
                return True
        return False

    def standing(self):
        upright = self.data.body("mannequin").xmat.reshape(3, 3)[2, 2] >= math.cos(math.radians(18))
        soles = [self.data.site(f"{side}_{leg}_sole").xpos[2]
                 for side in SIDES for leg in ("fore", "hind")]
        return bool(upright and all(abs(z - TABLE_Z) <= .004 for z in soles))

    def result(self):
        released_window = self.trace[-HZ:]
        released = (HZ <= len(self.trace) <= ACTION_LIMIT
                    and all(frame["released"] and frame["standing"] for frame in released_window))
        retained = {
            side: released and self.entered[side]
            and all(frame["limb_threaded"][side] for frame in released_window)
            for side in SIDES
        }
        pull = self.proof_pull() if all(retained.values()) else {"verified": False, "ran": False}
        return {"actions": len(self.actions), "score": .25 * sum(retained.values()) + .50 * pull["verified"],
                "forepaws_retained": retained, "released": released, "proof_pull": pull}

    def proof_pull(self):
        """Check the final state with no actuator inputs from the demonstrator."""
        last = self.trace[-1]
        dog = self.data.body("mannequin")
        center = np.mean([self.data.body(f"{side}_buckle").xpos for side in SIDES], axis=0)
        x, y, z = (center - dog.xpos) @ dog.xmat.reshape(3, 3)
        on_back = abs(x) <= .030 and -.085 <= y <= .045 and .030 <= z <= .090
        if (not self.with_latch or not all(last["paw_entered"].values())
                or not all(last["limb_threaded"].values()) or last["buckle_separation_mm"] > 25
                or not on_back
                or any(last["pad_loads"][side]["harness"] > .01 for side in SIDES)):
            return dict(verified=False, ran=False)
        proof = mj.MjData(self.model)
        mj.mj_copyData(proof, self.model, self.data)
        left, right = (proof.body(f"{side}_buckle") for side in SIDES)
        direction = left.xpos - right.xpos
        direction /= np.linalg.norm(direction)
        maximum_gap = 0.
        contacts, full_contacts = set(), 0
        ramp_s, hold_s = .20, 1.
        for index in range(round((ramp_s + hold_s) / DT)):
            force = min(1., index * DT / ramp_s) * 5 * direction
            proof.xfrc_applied[left.id, :3] = force
            proof.xfrc_applied[right.id, :3] = -force
            mj.mj_step(self.model, proof)
            if self.robot_touching_subject(proof):
                return {"verified": False, "ran": True, "force_assisted_by_robot": True}
            maximum_gap = max(maximum_gap, float(np.linalg.norm(left.xpos - right.xpos)))
            if index >= round((ramp_s + hold_s - .2) / DT):
                sample = set()
                for geom_pair in proof.contact.geom:
                    names = sorted(self.model.geom(int(item)).name for item in geom_pair)
                    if "catch" in names[0] and "tooth" in names[1]:
                        sample.add(tuple(names))
                contacts.update(sample)
                full_contacts += int(len(sample) == 2)
        retained = {side: self.limb_threaded(side, self.contour(side, proof), proof) for side in SIDES}
        verified = full_contacts >= round(.15 / DT) and maximum_gap <= .025 and all(retained.values())
        return dict(verified=verified, ran=True, contacts=sorted(contacts), legs_after_pull=retained,
                    maximum_gap_mm=round(maximum_gap*1000, 3), applied_hold_seconds=hold_s)

    def step(self, ctrl, name):
        self.data.ctrl[:] = ctrl
        for _ in range(round(1 / HZ / DT)):
            mj.mj_step(self.model, self.data)
        self.actions.append(ctrl.copy())
        if not math.isclose(self.data.time, len(self.actions) / HZ, abs_tol=1e-7):
            raise ValueError(f"Simulation clock reset during {name}")
        paws, inside, height, threaded = {}, {}, {}, {}
        for side in SIDES:
            polygon = self.contour(side, self.data)
            sole = self.data.site(f"{side}_fore_sole").xpos
            paws[side] = list((sole * 1000).round(2))
            height[side] = float(sole[2] - np.median(polygon[:, 2]))
            inside[side] = bool(in_polygon(sole[:2], polygon[:, :2]))
            threaded[side] = self.limb_threaded(side, polygon, self.data)
            cleared = inside[side] and height[side] >= .010
            if self.previous_clear[side] and inside[side] and height[side] < -.001:
                self.entered[side] = True
            if cleared:
                self.previous_clear[side] = True
            elif not inside[side]:
                self.previous_clear[side] = False
        seated = {side: bool(self.entered[side] and inside[side] and paws[side][2] <= TABLE_Z*1000 + 4.)
                  for side in SIDES}
        self.trace.append(dict(stage=name, body_mm=(self.data.body("mannequin").xpos * 1000).round(2).tolist(),
                               paws_mm=paws, paw_inside=inside,
                               heights_above_web_mm={side: round(height[side]*1000, 3) for side in SIDES},
                               paw_entered=dict(self.entered), seated=seated, limb_threaded=threaded,
                               released=not self.robot_touching_subject(self.data), standing=self.standing(),
                               buckle_mm={side: (self.data.body(f"{side}_buckle").xpos * 1000).round(2).tolist()
                                          for side in SIDES},
                               buckle_separation_mm=float(np.linalg.norm(self.data.body("left_buckle").xpos
                                   - self.data.body("right_buckle").xpos) * 1000),
                               tcp_mm={side: (self.data.site(f"{side}_tcp").xpos * 1000).round(2).tolist()
                                       for side in SIDES},
                               buckle_alignment=self.buckle_alignment(),
                               pad_loads=self.pad_loads()))


def try_place(episode, side, z_adjustment, angle, extra_y, grip_gap, lift, placement_offset):
    episode.phase("settle", 18)
    dog = episode.data.body("mannequin").xpos.copy()
    chest_point = dog + (0., -.035 + extra_y, z_adjustment)
    rotation = stance(side, angle=angle)
    episode.phase("approach chest", 45, goals=((side, chest_point+(0, 0, .114), rotation),),
                  gaps=((side, .095),))
    episode.phase("align around chest", 58, goals=((side, chest_point, rotation),))
    episode.phase("grasp chest", 20, gaps=((side, grip_gap),))
    episode.phase("lift mannequin", 52, goals=((side, chest_point + (0, 0, lift), rotation),))
    episode.phase("hold above bench", 16)
    paw_y = np.mean([episode.data.site(f"{s}_fore_sole").xpos[1] for s in SIDES])
    harness = episode.data.body("harness_bridge").xpos
    foreleg_shift = float(harness[1] - paw_y)
    above_harness = chest_point + (placement_offset, foreleg_shift, lift)
    episode.phase("carry forepaws above the two loops", 65, goals=((side, above_harness, rotation),))
    episode.phase("lower paws through loops", 62, goals=((side, above_harness - (0, 0, lift), rotation),))
    episode.phase("place weight onto paws", 18)
    episode.phase("release mannequin", 16, gaps=((side, .095),))
    episode.phase("withdraw from mannequin", 22, goals=((side, above_harness+(0, 0, .08), rotation),))
    episode.phase("observe released feet", 40)


def try_clip(episode, grip_height, grip_gap, pitch, *, align=False, lower=False):
    signs = dict(left=1., right=-1.)
    centers = {side: episode.data.site(f"{side}_buckle_grasp").xpos.copy() for side in SIDES}

    def attitude(side, angle):
        sign = signs[side]
        flat = Rotation.from_euler("z", sign * 90, degrees=True).as_matrix() @ stance(side)
        tilted = Rotation.from_euler("y", sign * pitch * (1 if episode.inward else -1),
                                    degrees=True).as_matrix() @ flat
        return Rotation.from_euler("z", sign * (angle - 90), degrees=True).as_matrix() @ tilted

    def goals(points, angle):
        return tuple((side, points[side], attitude(side, angle)) for side in SIDES)

    if pitch:
        episode.phase("make room beside the mannequin", 48,
                      goals=goals({side: point + (signs[side] * .110, 0, .250)
                                   for side, point in centers.items()}, 90),
                      gaps=tuple((side, .047) for side in SIDES))
    episode.phase("approach the two buckle halves", 44,
                  goals=goals({side: point + (0, 0, .082) for side, point in centers.items()}, 90),
                  gaps=tuple((side, .047) for side in SIDES))
    episode.phase("descend to the two buckle halves", 55,
                  goals=goals({side: point + (0, 0, grip_height) for side, point in centers.items()}, 90))
    episode.phase("pinch the housings", 20, gaps=tuple((side, grip_gap) for side in SIDES))
    turn = 0 if episode.inward else 180
    for step in range(1, 5):
        progress = step / 4
        lift_points = {side: point + (-signs[side] * .048 * progress, .060 * progress, .227 * progress)
                       for side, point in centers.items()}
        episode.phase(f"raise and turn the housings {step}", 32, goals=goals(lift_points, 90+turn*progress))
    y = float(episode.data.body("mannequin").xpos[1] - .040)
    z = float(episode.data.body("mannequin").xpos[2] + .072)
    episode.phase("bring the halves over the back", 46,
                  goals=goals({side: np.array([signs[side] * .021, y, z]) for side in SIDES}, 90+turn))
    if align:
        latch_center = np.mean([episode.data.body(f"{side}_buckle").xpos for side in SIDES], axis=0)
        latch_center += (0, .008, .009)

        def align_housings(separation):
            targets = []
            for side in SIDES:
                clip = episode.data.body(f"{side}_buckle")
                tcp = episode.data.site(f"{side}_tcp")
                clip_rotation = clip.xmat.reshape(3, 3)
                target_rotation = np.diag((-1., -1., 1.)) if side == "left" else np.eye(3)
                grip_in_clip = target_rotation @ clip_rotation.T
                target_grip = (latch_center + (signs[side] * separation / 2, 0, 0)
                               + grip_in_clip @ (tcp.xpos - clip.xpos))
                targets.append((side, target_grip, grip_in_clip @ tcp.xmat.reshape(3, 3)))
            return targets

        episode.phase("clear the buckle nose and square the housings", 35, goals=align_housings(.039))
        episode.phase("bring both tabs to the socket", 28, goals=align_housings(.033))
        episode.phase("press the latch together", 52, goals=align_housings(.007))
        if lower:
            latch_center[2] = episode.data.geom("chest").xpos[2] + .054
            episode.phase("lower the closed buckle onto the back", 28, goals=align_housings(.012))
            episode.phase("relax the press while holding alignment", 15, goals=align_housings(.0205))
    else:
        episode.phase("press the latch together", 30,
                      goals=goals({side: np.array([signs[side] * .016, y, z]) for side in SIDES}, 90+turn))
    episode.phase("release the buckle", 18, gaps=tuple((side, .051) for side in SIDES))
    episode.phase("observe the fitted harness", 36,
                  goals=goals({side: np.array([signs[side] * .10, y, z+.090]) for side in SIDES}, 90+turn))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument("--z-mm", type=float, default=3)
    parser.add_argument("--y-mm", type=float, default=0)
    parser.add_argument("--angle", type=float, default=0)
    parser.add_argument("--grip-mm", type=float, default=55)
    parser.add_argument("--lift-mm", type=float, default=48)
    parser.add_argument("--placement-offset-mm", type=float, default=0)
    parser.add_argument("--try-latch", action="store_true")
    parser.add_argument("--inward-latch", action="store_true")
    parser.add_argument("--dual-adjust", action="store_true")
    parser.add_argument("--clip-height-mm", type=float, default=6)
    parser.add_argument("--clip-grip-mm", type=float, default=13)
    parser.add_argument("--clip-pitch", type=float, default=0)
    parser.add_argument("--align-clips", action="store_true")
    parser.add_argument("--lower-clip", action="store_true")
    parser.add_argument("--replay", type=Path, help="Replay a previously saved action trajectory directory")
    args = parser.parse_args()
    if args.replay:
        plan = json.loads((args.replay / "trace.json").read_text())
        actions = np.load(args.replay / "actions.npy")
        episode = Episode(with_latch=args.try_latch, inward=args.inward_latch, dual_adjust=args.dual_adjust)
        for action, frame in zip(actions, plan, strict=True):
            episode.step(action, frame["stage"])
    else:
        episode = Episode(with_latch=args.try_latch, inward=args.inward_latch, dual_adjust=args.dual_adjust)
        try_place(episode, args.side, args.z_mm / 1000, args.angle, args.y_mm / 1000,
                  args.grip_mm / 1000, args.lift_mm / 1000, args.placement_offset_mm / 1000)
        if args.try_latch and all(episode.trace[-1]["seated"].values()):
            try_clip(episode, args.clip_height_mm / 1000, args.clip_grip_mm / 1000, args.clip_pitch,
                     align=args.align_clips, lower=args.lower_clip)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(**episode.result(), harness_has_physical_latch=args.try_latch,
                  starts_with_buckle_halves_facing_inward=args.inward_latch, dual_adjust=args.dual_adjust,
                  stage_pairs={item["stage"]: item for item in episode.trace})
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "trace.json").write_text(json.dumps(episode.trace, separators=(",", ":")) + "\n")
    np.save(args.out / "actions.npy", np.stack(episode.actions))


if __name__ == "__main__":
    main()
