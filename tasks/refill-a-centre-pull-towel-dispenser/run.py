"""Run isolated native physical stages from a stock station reset."""

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from motion import ARMS, HOME, Arm, direction
from paper import CreaseSet, TearablePerforation
from scene import ROLL_RADIUS, SIDE_LID_OPEN, TABLE, build
from score import Observer


HZ = 20
ACTION_LIMIT = 3000


class Trial:
    def __init__(self, *, on_side=False, with_paper=False, case_on_side=False, case_x=.028, case_y=-.499,
                 measure=False, pace=1.):
        self.on_side = on_side
        self.model = mj.MjModel.from_xml_string(build(on_side=on_side, with_paper=with_paper,
                                                      case_on_side=case_on_side,case_x=case_x,case_y=case_y))
        self.data = mj.MjData(self.model)
        self.planning = mj.MjData(self.model)
        self.arms = {side: Arm(self.model, side) for side in ARMS}
        self.actions = []
        self.trace = []
        self.targets = np.stack([HOME, HOME])
        self.orientation = {}
        self.pace = pace
        self.geoms = {
            name: {i for i in range(self.model.ngeom)
                   if self.model.geom(i).name.startswith(prefix)}
            for name, prefix in (("core", "core_"), ("roll", "roll_"), ("lid", ("cover_","lid_lip","nozzle_")),
                                 ("first_wipe", "first_wipe_patch_"), ("next_wipe", "next_wipe_patch_"))
        }
        self.pads = {
            side: tuple({i for i in range(self.model.ngeom)
                         if self.model.geom(i).name.startswith(f"{side}_tip_{pad}_contact_")}
                        for pad in ("left", "right"))
            for side in ARMS
        }
        self.data.joint("lid_hinge").qpos[:] = SIDE_LID_OPEN if case_on_side else 1.75
        for arm in self.arms.values():
            arm.reset(self.data)
        mj.mj_forward(self.model, self.data)
        self.substeps = round(1 / (HZ * self.model.opt.timestep))
        self.paper = TearablePerforation(self.model) if with_paper else None
        self.scoring = Observer(self.model) if measure else None
        self.crease = CreaseSet(self.model) if with_paper else None

    def loads(self, side, target, data=None):
        data = self.data if data is None else data
        force = np.zeros(6)
        sums = np.zeros(2)
        for i, contact in enumerate(data.contact[:data.ncon]):
            pair = set(contact.geom)
            if not pair & self.geoms[target] or contact.efc_address < 0:
                continue
            for j, pad in enumerate(self.pads[side]):
                if pair & pad:
                    mj.mj_contactForce(self.model, data, i, force)
                    sums[j] += max(0, force[0])
        return sums.tolist()

    def sample(self, label):
        data = self.data
        roll = data.body("roll")
        core = data.body("core")
        row = {
            "action": len(self.actions),
            "phase": label,
            "time_s": float(data.time),
            "core_center_m": core.xpos.tolist(),
            "roll_center_m": roll.xpos.tolist(),
            "dispenser_center_m": data.body("dispenser").xpos.tolist(),
            "lid_angle_rad": float(data.joint("lid_hinge").qpos[0]),
            "latch_angle_rad": float(data.joint("latch_deflection").qpos[0]),
            "core_grip_m": data.site("core_grip").xpos.tolist(),
            "wrist_m": {side: self.tcp(side).tolist() for side in ARMS},
            "pad_n": {
                f"{side}_{target}": self.loads(side, target)
                for side in ARMS for target in self.geoms
            },
        }
        if self.paper is not None:
            row["first_wipe_leader_m"] = data.site("first_wipe_leader").xpos.tolist()
            row["next_wipe_leader_m"] = data.site("next_wipe_leader").xpos.tolist()
            row["tension_n"] = self.paper.tension_n
            row["first_wipe_tear_s"] = self.paper.break_time_s
        if self.scoring is not None:
            row.update(self.scoring.observe(data))
        return row

    def step(self, action, label):
        if len(self.actions) >= ACTION_LIMIT:
            raise RuntimeError("Exceeded the control-action limit")
        for arm, physical_targets in zip(self.arms.values(), action):
            arm.command(self.data, physical_targets)
        for _ in range(self.substeps):
            mj.mj_step(self.model, self.data)
            if self.paper is not None:
                self.paper.observe_substep(self.data)
            if self.scoring is not None:
                self.scoring.audit_substep(self.data)
        if self.crease is not None:
            self.crease.observe(self.data)
        self.actions.append(action.copy())
        if abs(self.data.time - len(self.actions) / HZ) > 1e-7:
            raise RuntimeError(f"Physics clock reset during {label}")
        self.targets = action.copy()
        self.trace.append(self.sample(label))

    def ramp(self, targets, frames, label):
        frames = max(7, round(frames*self.pace))
        initial = self.targets.copy()
        for j in range(1, frames+1):
            u = j / frames
            self.step(initial + (u*u*(3-2*u)) * (targets-initial), label)
        print(json.dumps(self.trace[-1]), flush=True)

    def tcp(self, side):
        return self.data.site(f"{side}_tcp").xpos.copy()

    def move(self, side, xyz, frames, label, *, gap=None, frame=None):
        if frame is not None:
            self.orientation[side] = frame
        targets = self.targets.copy()
        i = ARMS.index(side)
        self.planning.qpos[:] = self.data.qpos
        targets[i,:6] = self.arms[side].inverse(
            self.planning, np.asarray(xyz), self.orientation[side], targets[i,:6]
        )
        if gap is not None:
            targets[i,6] = gap/.095
        self.ramp(targets, frames, label)

    def fingers(self, side, gap, frames, label):
        targets = self.targets.copy()
        targets[ARMS.index(side),6] = gap/.095
        self.ramp(targets, frames, label)

    def extract_core(self):
        self.ramp(self.targets,12,"settle the tabletop reset")
        grip = self.data.site("core_grip").xpos.copy()+[-.010,0,0]
        self.move("right",grip+[0,.100,.137],37,"move above the open end",frame=direction(0,45),gap=.021)
        self.move("right",grip+[0,.100,.063],25,"lower clear of the roll",frame=direction(0,32))
        self.move("right",grip+[0,.100,0],25,"level with the roll opening",frame=direction(0,10))
        self.move("right",grip+[0,.049,0],36,"approach the cardboard seam")
        self.move("right",grip,18,"straddle the cardboard seam")
        self.fingers("right",0.,11,"grasp the cardboard edge")
        for draw in (.018,.041,.082,.111,.134,.165,.190,.215):
            centering = .008*np.clip((draw-.083)/.04,0,1)
            support = max(0.,draw-.153)*.42
            self.move("right",grip+[centering,draw,support],17,"peel and extract the cardboard sleeve")
        at = self.tcp("right")
        self.move("right",at+[-.023,.008,.049],26,"clear the cardboard from the opening",frame=direction(-7,16))
        self.move("right",[-.291,-.044,.922],28,"carry the sleeve away from the refill",frame=direction(-8,18))
        if self.scoring is not None and self.scoring.observe(self.data)["core_set_aside"]:
            self.move("right",[-.336,-.023,.955],25,"withdraw beside the removed sleeve",frame=direction(-8,18))
            self.move("right",[-.368,-.035,1.007],25,"leave the removed sleeve on the tabletop",frame=direction(-14,30))
            self.ramp(self.targets,15,"let the removed sleeve settle")
            return
        self.move("right",[-.291,-.056,.844],23,"lower the sleeve to the table",frame=direction(-5,29))
        self.fingers("right",.094,19,"set aside the removed sleeve")
        self.move("right",[-.362,-.068,.895],25,"clear the dropped sleeve sideways",frame=direction(-17,32))
        self.move("right",[-.339,-.074,1.000],24,"retreat above the tabletop",frame=direction(-12,39))
        self.move("right",[-.223,.015,1.010],26,"leave the fallen sleeve clear of the gripper",
                  frame=direction(-13,30))
        self.ramp(self.targets,28,"let the extracted sleeve settle on the table")

    def move_pair(self, points, frames, label, *, gap=None, orientation=None):
        commands = self.targets.copy()
        for index, side in enumerate(ARMS):
            if orientation is not None:
                self.orientation[side] = orientation[side]
            self.planning.qpos[:] = self.data.qpos
            commands[index,:6] = self.arms[side].inverse(
                self.planning, points[side], self.orientation[side], commands[index,:6]
            )
            if gap is not None:
                commands[index,6] = gap/.095
        self.ramp(commands, frames, label)

    def core(self, yaw, pitch, offset, *, lift_only=False):
        self.ramp(self.targets, 12, "settle the tabletop reset")
        grip = self.data.site("core_grip").xpos.copy() + np.asarray(offset)
        r = direction(yaw, pitch)
        axial = np.array([0,1,0] if self.on_side else [0,0,1])
        if self.on_side:
            self.move("right", grip+[0,.100,.137], 37, "move above the open end", frame=direction(0,45), gap=.021)
            self.move("right", grip+[0,.100,.063], 25, "lower clear of the roll", frame=direction(0,32))
            self.move("right", grip+[0,.100,.000], 25, "level with the centre", frame=r)
        self.move("right", grip + .049*axial, 36, "approach the cardboard peel edge", frame=r, gap=.021)
        self.move("right", grip, 18, "straddle the cardboard peel edge")
        self.fingers("right", 0., 11, "close about the core edge")
        if self.on_side:
            for distance in (.018,.041,.082):
                self.move("right", grip+distance*axial, 15, "peel and start extracting the core")
            core = self.data.body("core").xpos.copy()
            around = core + [0,.155,.003]
            self.fingers("right", .088, 18, "open to regrip the exposed core")
            self.move("right", around+[0,.071,.048], 19, "withdraw from the peel edge", frame=direction(0,17))
            self.move("right", around+[0,.071,0], 17, "line up the exposed core", frame=direction(0,10))
            self.move("right", around, 26, "straddle the exposed core")
            self.fingers("right", .035, 20, "grasp the exposed cardboard sleeve")
            for distance in (.035,.070,.103,.129):
                self.move("right", around+[0,distance,max(0,distance-.100)*.65], 17, "extract the core axially")
            for attempt in range(2):
                core = self.data.body("core").xpos.copy()
                roll = self.data.body("roll").xpos.copy()
                required = roll[1]+.206-core[1]
                if required <= 0:
                    break
                center = core+[0,.127+.008*attempt,.002]
                self.fingers("right", .091, 19, "open for the exposed cardboard circumference")
                self.move("right", center+[0,.055,.066], 24, "approach the partly extracted core",
                          frame=direction(-8,24))
                self.move("right", center+[0,.054,0], 21, "level with the partly extracted core",
                          frame=direction(-3,11))
                self.move("right", center, 26, "straddle the rigid part of the sleeve")
                self.fingers("right", .018, 24, "regrasp the cardboard to support its full width")
                for part in (.33,.66,1.):
                    self.move("right", center+[0,required*part,.020*max(0,part-.58)],
                              24,"complete extraction without rocking the refill")
            if self.data.body("core").xpos[1] < self.data.body("roll").xpos[1]+.184:
                raise RuntimeError("The cardboard sleeve was not fully removed")
        else:
            for distance in (.018,.041,.082,.125,.166,.202):
                self.move("right", grip+distance*axial, 12, "extract the core axially")
        if lift_only:
            return
        at = self.tcp("right")
        if self.on_side:
            self.move("right", at+[0,.012,.035], 21, "clear the end of the roll", frame=direction(0,11))
            self.move("right", [-.291,-.040,.916], 28, "carry the removed core to the side", frame=direction(0,16))
            self.move("right", [-.291,-.055,.850], 23, "lower the core to the tabletop", frame=direction(0,28))
            self.fingers("right", .094, 19, "leave the removed core beside the roll")
            self.move("right", [-.291,-.06,.970], 21, "withdraw from the extracted core", frame=direction(0,38))
            return
        # Position above a clear region of the existing tabletop.
        self.move("right", at+[-.126,.023,0.], 27, "carry the extracted core clear")
        height = .038 if self.on_side else .176
        self.move("right", self.tcp("right")+[0,0,-height], 22, "lower the core to the table")
        self.fingers("right", .047, 12, "set the core aside")
        self.move("right", self.tcp("right")+[0,0,.052], 17, "withdraw from the extracted core")

    def load(self):
        roll = self.data.body("roll").xpos.copy()
        grip = roll + [0,.179,.048]
        self.move("right", grip+[0,.093,.105], 34, "move above the paper roll", frame=direction(0,43),gap=.074)
        self.move("right", grip+[0,.093,0], 30, "lower clear of the paper roll",frame=direction(0,10))
        self.move("right", grip, 23, "straddle the roll wall")
        self.fingers("right", .028, 19, "grasp the intact roll")
        self.move("right", grip+[0,0,.045], 25, "lift the intact roll")
        self.move("right", grip+[0,.108,.045], 29, "carry the roll clear of the case mouth")
        case = self.data.body("dispenser").xpos.copy()
        grip = np.array([case[0],grip[1]+.108,case[2]+.048])
        self.move("right", grip+[0,0,.031], 32, "bring the roll into the loading lane")
        self.move("right", grip, 22, "align the roll with the case")
        self.move("right", [grip[0],case[1]+.195,grip[2]], 65, "seat the roll through the open mouth")
        self.fingers("right", .060, 16, "release the seated refill")
        self.move("right", self.tcp("right")+[0,.093,.023], 23, "withdraw from the refill")

    def carry_roll(self):
        centre = self.data.body("roll").xpos.copy() + [0,.105,.003]
        orient = {"left": direction(70,10), "right": direction(-70,10)}

        def points(radius, delta):
            return {side: centre + [radius*sign,0,0] + delta
                    for side,sign in (("left",1),("right",-1))}

        self.move_pair(points(.133,[0,0,.121]), 40, "approach both sides of the roll", gap=.021,
                       orientation={"left":direction(50,32),"right":direction(-50,32)})
        self.move_pair(points(.133,[0,0,.048]), 29, "lower both grippers beside the roll", orientation=orient)
        self.move_pair(points(.125,[0,0,0]), 30, "line up the side contacts")
        self.move_pair(points(ROLL_RADIUS-.010,[0,0,0]), 32, "brace the roll between both grippers")
        self.move_pair(points(ROLL_RADIUS-.010,[0,0,.032]), 30, "lift the roll with both grippers")
        for shift in (.046,.090,.123):
            self.move_pair(points(ROLL_RADIUS-.010,[0,shift,.032]), 23, "move the roll clear of the case")
        case = self.data.body("dispenser").xpos.copy()
        for fraction in (.35,.70,1.):
            self.move_pair(points(ROLL_RADIUS-.010,[fraction*(case[0]-centre[0]),.123,.032]), 30,
                           "carry the roll across to the case")
        x_shift = case[0]-centre[0]
        self.move_pair(points(ROLL_RADIUS-.010,[x_shift,.123,.043]), 23,
                       "line up the roll with the mouth")
        for insertion in (.039,.076,.110,.141,.165,.180):
            actual = self.data.body("roll").xpos.copy()
            x_shift += np.clip(case[0]-actual[0],-.004,.004)
            height = .043 + np.clip((case[2]-actual[2])*.5,-.006,.006)
            self.move_pair(points(ROLL_RADIUS-.010,[x_shift,.123-insertion,height]), 25,
                           "guide the front edge into the case")
        self.finish_loading()

    def finish_loading(self):
        roll = self.data.body("roll").xpos.copy()
        case = self.data.body("dispenser").xpos.copy()
        if roll[1] > case[1]+.162:
            raise RuntimeError("The roll did not reach a stable partial insertion")

        current = {side: self.tcp(side) for side in ARMS}
        self.move_pair({"right":current['right']+[-.039,.031,.012],
                        "left":current['left']+[.041,.031,.012]}, 25, "let the sleeve support the roll")
        self.move("left", self.tcp("left")+[0,.087,.112], 28, "raise the left gripper away", frame=direction(35,35))
        self.push_roll_to_base()

    def push_roll_to_base(self):
        roll = self.data.body("roll").xpos.copy()
        case = self.data.body("dispenser").xpos.copy()
        push = roll + [0,.189,.052]
        self.move("right", push+[0,.086,.068], 30, "approach the exposed end of the refill",
                  frame=direction(-25,24),gap=.009)
        self.move("right", push+[0,.062,0], 23, "line up with the outside paper wall", frame=direction(-7,8))
        self.move("right", push, 25, "place the gripper against the refill end")
        depth = np.clip((roll[1]-case[1])-.006,0.,.195)
        for fraction in (.33,.66,1.):
            self.move("right", push+[0,-fraction*depth,0], 28, "push the refill home inside the case")
        self.move("right", self.tcp("right")+[0,.089,.031], 25, "withdraw after loading")

    def save(self, directory, fault):
        directory.mkdir(exist_ok=True, parents=True)
        np.save(directory/"actions.npy", np.asarray(self.actions))
        state = np.empty(mj.mj_stateSize(self.model, mj.mjtState.mjSTATE_INTEGRATION))
        mj.mj_getState(self.model, self.data, state, mj.mjtState.mjSTATE_INTEGRATION)
        np.save(directory/"native_state.npy",state)
        if self.crease is not None:
            np.save(directory/"bending_rest.npy", self.crease.rest)
            np.save(directory/"bending_stiffness.npy", self.crease.stiffness)
        with (directory/"trace.jsonl").open("w") as handle:
            for row in self.trace:
                handle.write(json.dumps(row, allow_nan=False)+"\n")
        (directory/"summary.json").write_text(json.dumps(self.sample("terminal") | {"fault":fault},indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--yaw",type=float,default=0.)
    parser.add_argument("--pitch",type=float,default=66.)
    parser.add_argument("--offset-mm",type=float,nargs=3,default=(0.,0.,0.))
    parser.add_argument("--lift-only",action="store_true")
    parser.add_argument("--side",action="store_true")
    parser.add_argument("--paper",action="store_true")
    parser.add_argument("--case-side",action="store_true")
    parser.add_argument("--case-x",type=float,default=.028)
    parser.add_argument("--case-y",type=float,default=-.499)
    parser.add_argument("--load",action="store_true")
    parser.add_argument("--carry",action="store_true")
    args=parser.parse_args()
    trial=Trial(on_side=args.side, with_paper=args.paper, case_on_side=args.case_side,case_x=args.case_x,case_y=args.case_y)
    fault=None
    try:
        trial.core(args.yaw,args.pitch,np.asarray(args.offset_mm)/1000,lift_only=args.lift_only)
        if args.load:
            trial.load()
        if args.carry:
            trial.carry_roll()
    except (RuntimeError,ValueError) as error:
        fault=str(error)
        print(fault,flush=True)
    trial.save(args.out,fault)


if __name__=="__main__":
    main()
