"""Run capped dual-YAM mechanics with repeatable joint targets."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import numpy as np
from scipy.spatial.transform import Rotation

from control import ARMS, HOME, Arm, direction
from scene import HERE, MOUNT, TABLE, Mechanism, build, mj

CONTROL_HZ = 20
EFFORT = np.array([27., 27., 27., 7., 7., 7.])
ACTION_LIMIT = 2400
CAMERAS={"top_cam":"top","left_cam":"left","right_cam":"right","judging_front":"detail"}


def grip_frame(yaw, pitch, roll=-90.):
    return direction(yaw, pitch) @ Rotation.from_euler("z", roll, degrees=True).as_matrix()


class Trial:
    def __init__(self, model, data, mechanism, stream, capture=None):
        self.model, self.data, self.mechanism = model, data, mechanism
        self.arms = [Arm(model, side) for side in ARMS]
        self.phase = "reset"
        self.stream, self.capture = stream, capture
        self.now = np.stack([HOME, HOME])
        self.target_rotation = [None, None]
        self.plan = mj.MjData(model)
        self.actions = []
        self.peak_effort = np.zeros((2, 6))
        self.physics_steps = round(1/(CONTROL_HZ*model.opt.timestep))
        for arm in self.arms:
            model.actuator_forcerange[arm.actuators] = np.column_stack((-EFFORT, EFFORT))
            model.actuator_forcerange[arm.fingers] = -40., 40.
            arm.reset(data)
        mj.mj_forward(model, data)

    def step(self, action):
        targets = np.asarray(action, dtype=float).reshape(2, 7)
        if len(self.actions) >= ACTION_LIMIT:
            raise ValueError("action limit exceeded")
        for arm, target in zip(self.arms, targets):
            arm.command(self.data, target)
        for _ in range(self.physics_steps):
            self.mechanism.step()
            mj.mj_step(self.model, self.data)
            self.mechanism.observe()
            for i, arm in enumerate(self.arms):
                self.peak_effort[i] = np.maximum(self.peak_effort[i],
                                                abs(self.data.actuator_force[arm.actuators]))
        self.now = targets.copy()
        self.actions.append(self.now.copy())
        row = self.mechanism.score()
        row.update(time_s=self.data.time, action=len(self.actions), phase=self.phase,
                   tcp_m=[self.tcp(i).tolist() for i in (0, 1)],
                   objects_m={name: self.data.body(name).xpos.tolist() for name in
                              ("controller", "face_cover", "old_module", "new_module")})
        self.stream.write(json.dumps(row)+"\n")
        if self.capture:
            self.capture()

    def ramp(self, target, duration, phase):
        self.phase = phase
        initial = self.now.copy()
        count = round(duration*CONTROL_HZ)
        for k in range(1, count+1):
            fraction = k/count
            blend = fraction*fraction*(3-2*fraction)
            self.step(initial+(target-initial)*blend)

    def tcp(self, side):
        return self.data.site_xpos[self.arms[side].site].copy()

    def tcp_rotation(self, side):
        return self.data.site_xmat[self.arms[side].site].reshape(3, 3).copy()

    def move(self, side, point, duration, phase, gap=None, rot=None):
        point = np.asarray(point)
        aiming = self.target_rotation[side] if rot is None else rot
        if aiming is None:
            aiming=self.tcp_rotation(side)
        old = self.now[side,:6]
        joint_target = None
        for candidate in (aiming, aiming @ np.diag([-1., -1., 1.])):
            for seed in (old, HOME[:6]):
                try:
                    joint_target = self.arms[side].inverse(self.plan, point, candidate, seed)
                except ValueError as err:
                    failure = str(err)
                else:
                    break
            if joint_target is not None:
                break
        if joint_target is None:
            raise ValueError(f"{phase}: {failure}")
        goals = self.now.copy()
        goals[side,:6] = joint_target
        if gap is not None:
            goals[side,6] = gap
        self.ramp(goals, duration, phase)
        self.target_rotation[side] = candidate
        if duration >= .35:
            error = float(np.linalg.norm(self.tcp(side)-point))*1000
            print(f"{phase}: TCP {self.tcp(side).round(4)} | {error:.1f} mm error", flush=True)

    def jaws(self, side, target, phase, duration=.45):
        goals = self.now.copy()
        goals[side,6] = target
        self.ramp(goals, duration, phase)

    def park(self, side, phase):
        goals = self.now.copy()
        goals[side] = HOME
        self.ramp(goals, 1.25, phase)

    def position(self, site):
        return self.data.site(site).xpos.copy()

    def relative_point(self, body_name, point):
        body = self.data.body(body_name)
        return body.xpos + body.xmat.reshape(3,3) @ point

    def contacts_on(self, prefix):
        grouped = set()
        force = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            if min(contact.geom1,contact.geom2)<0:
                continue
            names=[self.model.geom(geom).name for geom in (contact.geom1,contact.geom2)]
            matching=[name for name in names if name.startswith(prefix)]
            if len(matching)==1:
                mj.mj_contactForce(self.model,self.data,index,force)
                if force[0]>.3:
                    item=next(name for name in names if name!=matching[0])
                    grouped.add((item,matching[0].split("_contact_")[0]))
        return sorted(grouped)

    def brace(self, args, face_up=False):
        x = -.060 if args.mode == "full" and not face_up else .060
        point = self.relative_point("controller", [x, args.brace_y, args.brace_z])
        r = grip_frame(args.brace_yaw, args.brace_pitch, args.brace_roll)
        self.move(0, point-r[:,2]*.070, 1.30, "approach controller handle", .52, r)
        self.move(0, point, .90, "straddle the handle", .52)
        self.jaws(0, args.brace_close, "grasp controller handle")
        print("Controller grip", self.data.body("controller").xpos.round(4),
              self.mechanism.load("controller_grip",side="left"), flush=True)

    def unlatch(self, args):
        at = self.position("latch_tip")
        r = grip_frame(args.latch_yaw, args.latch_pitch, args.latch_roll)
        offset = np.array(args.latch_offset)
        start = at+offset
        self.move(1, start-r[:,2]*.080, 1.20, "approach rear release", .004, r)
        start = self.position("latch_tip")+offset
        self.move(1, start-r[:,2]*.025, .65, "align above the rear release", .004)
        for travel in (-.008, -.005, -.002, .001, .004, .007, .010, .013):
            latch_body = self.data.body("controller")
            axis = -latch_body.xmat.reshape(3,3)[:,0]
            start = self.position("latch_tip")+offset-axis*self.data.joint("latch_slide").qpos[0]
            self.move(1, start+axis*travel, .42, "slide the rear release")
            print("Rear latch", self.mechanism.score()["latch_mm"], self.mechanism.opened,
                  self.mechanism.load("release_slider",side="right"),
                  self.mechanism.load("main_chassis",side="right"),
                  "offset mm",(1000*(self.tcp(1)-self.position("latch_tip"))).round(1),flush=True)
            if self.mechanism.opened:
                break
        if not self.mechanism.opened:
            raise ValueError("rear release never reached its native travel limit under finger contact")
        self.move(1, self.tcp(1)-r[:,2]*.075, .75, "clear the release latch", .25)
        self.park(1, "move right arm clear of the cover")
        controller_start = self.data.body("controller").xpos.copy()
        self.move(0, self.tcp(0)+[0,0,.070], 1.25, "lift controller away from the loose cover")
        print("Cover release", self.mechanism.score(), "controller lift",
              (self.data.body("controller").xpos-controller_start).round(4), flush=True)

    def remove_cover(self, args):
        rot = grip_frame(args.cover_yaw, args.cover_pitch, args.cover_roll)
        offset = np.array(args.cover_offset)
        point = self.position("cover_grip")+offset
        self.move(1, point-rot[:,2]*.073, 1.20, "approach the front cover edge", .29, rot)
        self.move(1, self.position("cover_grip")+offset-rot[:,2]*.028,
                  .62, "line up with the cover edge", .22)
        self.move(1, self.position("cover_grip")+offset, .76, "straddle the cover edge", .18)
        self.jaws(1, .005, "grasp the front cover")
        print("Native cover grip", self.mechanism.load("cover_",side="right"),
              "offset mm",(1000*(self.tcp(1)-self.position("cover_grip"))).round(1),flush=True)
        at = self.tcp(1)
        face = self.data.body("controller").xmat.reshape(3,3)[:,2].copy()
        for distance in (.009,.019,.032,.043,.052):
            self.move(1, at+face*distance, .46, "detach the front cover")
            print("Cover pull", self.mechanism.rel("face_cover")[0].round(4),
                  self.mechanism.load("cover_",side="right"), flush=True)
            if self.mechanism.shown:
                break
        if not self.mechanism.shown:
            raise ValueError("the cover stayed on the controller after manual release")
        self.move(1,self.tcp(1)+[-.101,.080,.016],1.11,"carry the cover clear")
        cover_z = self.data.body("face_cover").xpos[2]
        self.move(1,self.tcp(1)+[0.,0.,TABLE+.005-cover_z],.78,"set the front cover on the table")
        self.jaws(1,.37,"release the front cover")
        self.move(1,self.tcp(1)+[0,.040,.046],.73,"clear the loose front cover")

    def lowest_controller_point(self, include_original=True):
        names = ["controller","left_other","module_lever","rear_latch"]
        if include_original:
            names.append("old_module")
        parts = [self.model.body(name).id for name in names]
        height = []
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] not in parts:
                continue
            z_axis = self.data.geom_xmat[geom].reshape(3,3)[2]
            size = self.model.geom_size[geom]
            if self.model.geom_type[geom] == mj.mjtGeom.mjGEOM_ELLIPSOID:
                extent = np.linalg.norm(z_axis*size)
            elif self.model.geom_type[geom] == mj.mjtGeom.mjGEOM_CYLINDER:
                extent = size[0]*np.linalg.norm(z_axis[:2])+size[1]*abs(z_axis[2])
            else:
                extent = abs(z_axis)@size
            height.append(self.data.geom_xpos[geom,2]-extent)
        return min(height)

    def turn_controller(self, args):
        self.park(1,"clear the right arm for the turn")
        anchor = self.tcp(0)+[0.,0.,.106]
        rotation = self.tcp_rotation(0)
        self.move(0,anchor,1.10,"lift the open controller",rot=rotation)
        azimuth = np.deg2rad(args.turn_axis)
        axis = np.array([np.cos(azimuth),np.sin(azimuth),0.])
        for angle in np.linspace(0.,args.turn_deg,9)[1:]:
            aim = Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()@rotation
            self.move(0,anchor,.44,"turn the controller face up",rot=aim)
            print("Controller face normal",self.data.body("controller").xmat.reshape(3,3)[:,2].round(3),flush=True)
        self.move(0, self.tcp(0)+[.022,.048,-.025], .78,
                  "bring the turned controller into the working area")
        for _ in range(3):
            height = self.lowest_controller_point()
            target = self.tcp(0)+[0,0,max(-.054,TABLE+.003-height)]
            self.move(0,target,.64,"lower the controller to the table")
        self.jaws(0,.76,"let the controller settle face up",duration=.57)
        self.park(0,"clear the controller handle")
        self.ramp(self.now,.68,"controller rests on the work surface")
        forward = self.data.body("controller").xmat.reshape(3,3)[2,2]
        print("Resting face normal",self.data.body("controller").xmat.reshape(3,3)[:,2].round(3),flush=True)
        if forward < .95:
            raise ValueError("controller did not settle face up")

    def lift_lever(self, args):
        at = self.position("lever_tip")
        r = grip_frame(args.lever_yaw, args.lever_pitch, args.lever_roll)
        offset = np.array(args.lever_offset)
        start = at+offset
        self.move(1,start-r[:,2]*.066,1.3,"approach the module release lever",.27,r)
        for height in (.026, .012, .005, .0, -.003):
            start = self.position("lever_tip")+offset+[0,0,height]
            self.move(1,start,.45,"straddle the module release lever",args.lever_opening)
        self.jaws(1,.00,"grip the release lever",duration=.8)
        print("Native lever grip",self.mechanism.load("lever_",side="right"),
              "offset mm",(1000*(self.tcp(1)-self.position("lever_tip"))).round(1),
              self.contacts_on("right_"),flush=True)
        pivot = self.position("lever_pivot")
        axis = self.data.body("controller").xmat.reshape(3,3)[:,0].copy()
        relative = self.tcp(1)-pivot
        gripped_rotation = self.tcp_rotation(1)
        angles = [9.,18.,28.,39.,52.,65.]
        if args.lever_tilt>65.:
            angles.append(args.lever_tilt)
        for degree in angles:
            rotation = Rotation.from_rotvec(axis*np.deg2rad(degree))
            target = self.position("lever_pivot")+rotation.apply(relative)
            self.move(1,target,.50,"rotate the release lever",rot=rotation.as_matrix()@gripped_rotation)
            print("Release lever", self.mechanism.score()["lever_deg"],
                  self.mechanism.load("lever_",side="right"), flush=True)
        if not self.mechanism.lever_seen:
            raise ValueError("module release lever was never lifted past 59 degrees")
        self.jaws(1,.32,"leave the raised release lever")
        self.move(1,self.tcp(1)+[-.018,.012,.024],.70,"clear the raised release lever",.38,rot=r)

    def remove_module(self, args):
        at = self.position("old_module_grip")
        r = grip_frame(args.module_yaw,args.module_pitch,args.module_roll)
        start = at+np.array(args.module_offset)
        self.move(1,start-r[:,2]*.056,1.10,"approach the old module's thumbstick",.22,r)
        for hover in (.032,.020,.010,.003,-.003):
            start=self.position("old_module_grip")+np.array(args.module_offset)+[0,0,hover]
            self.move(1,start,.46,"straddle the old module's thumbstick",args.module_opening)
            print("Thumbstick contacts",self.contacts_on("right_"),flush=True)
        self.jaws(1,.005,"grip the thumbstick")
        print("Thumbstick pinch",self.contacts_on("right_"),
              (1000*(self.tcp(1)-self.position("old_module_grip"))).round(1),flush=True)
        tcp_start = self.tcp(1)
        axis = self.data.body("controller").xmat.reshape(3,3)[:,1].copy()
        for distance in (.012,.024,.038,.052,.069):
            self.move(1,tcp_start+axis*distance,.34,"slide the old module by its thumbstick")
            print("Module slide", self.mechanism.rel("old_module")[0].round(4),
                  self.mechanism.load("old_module_stick",side="right"),
                  self.contacts_on("old_module"),flush=True)
        self.move(1,self.tcp(1)+[0,0,.029],.65,"lift the clear module")
        if not self.mechanism.old_out:
            raise ValueError("old module never left the guide")
        self.move(1,self.tcp(1)+[-.075,.040,0],.85,"carry the old module away from its socket")
        at = self.data.body("old_module").xpos.copy()
        self.move(1,self.tcp(1)+[0,0,TABLE+.0085-at[2]],.75,"set the old module on the table")
        self.jaws(1,.35,"release the original module")
        self.move(1,self.tcp(1)+[0,.018,.051],.65,"leave the original module clear")

    def steer_held_object(self, name, target, duration, phase, tilt=1.):
        item = self.data.body(name)
        base = self.data.body("controller").xmat.reshape(3,3)
        item_world = item.xmat.reshape(3,3)
        expected = self.relative_point("controller",target)
        spin = Rotation.from_matrix(base@item_world.T).as_rotvec()
        if np.linalg.norm(spin) > .14:
            spin *= .14/np.linalg.norm(spin)
        grip = self.tcp_rotation(1)
        aim = Rotation.from_rotvec(spin*tilt).as_matrix()@grip
        # Apply measured part error while keeping gripper jaws on the stick.
        target_tcp = self.tcp(1)+np.clip(expected-item.xpos,-.025,.025)
        self.move(1,target_tcp,duration,phase,rot=aim)

    def fit_replacement(self, args):
        r = grip_frame(args.new_yaw,args.new_pitch,args.new_roll)
        at = self.position("new_module_grip")+np.array(args.new_grip)
        self.move(1,at-r[:,2]*.062,.91,"approach the replacement module",.24,r)
        for hover in (.026,.012,.004,0.):
            at = self.position("new_module_grip")+np.array(args.new_grip)
            self.move(1,at+[0,0,hover],.43,"straddle the replacement thumbstick",.22)
        self.jaws(1,.003,"grip the replacement thumbstick")
        print("Replacement native grip", self.contacts_on("new_module"), flush=True)
        self.move(1,self.tcp(1)+[0,0,.088],.90,"lift the replacement module")
        item = self.data.body("new_module")
        hand = self.tcp_rotation(1)
        in_hand = hand.T @ (item.xpos-self.tcp(1))
        point = self.relative_point("controller",MOUNT+[0.,.069,.090])-hand@in_hand
        midway = (self.tcp(1)+point)/2
        self.move(1,midway,.75,"carry the replacement toward the housing",rot=hand)
        self.move(1,point,.75,"carry the replacement above the guide")
        if self.data.body("new_module").xpos[2]<TABLE+.06:
            raise ValueError("replacement thumbstick slipped while carrying it")
        for _ in range(8):
            self.steer_held_object("new_module",MOUNT+[0.,.063,.078],.45,
                                   "level the replacement above the entry")
            position,rotation=self.mechanism.rel("new_module")
            if (np.linalg.norm(position-(MOUNT+[0.,.063,.078]))<.0025 and
                    Rotation.from_matrix(rotation).magnitude()<np.deg2rad(2)):
                break
        for z in (.055,.035,.021):
            self.steer_held_object("new_module",MOUNT+[0.,.060,z],.52,
                                   "lower the replacement near the opening")
        for y,z in ((.058,.010),(.057,.002),(.056,.001),(.056,.001),(.056,.001),
                    (.044,.001),(.035,.001),(.025,.0007),(.020,.0007),(.014,.0007),
                    (.009,.0006),(.005,.0006),(.002,.0004)):
            self.steer_held_object("new_module",MOUNT+[args.guide_x,y,z+args.guide_z],.43,
                                   "guide the replacement into the housing")
            position,rotation = self.mechanism.rel("new_module")
            print("Replacement guide",position.round(4),
                  np.rad2deg(Rotation.from_matrix(rotation).magnitude()).round(1),
                  self.contacts_on("new_module"),self.contacts_on("right_"),flush=True)
        self.jaws(1,.28,"release the module in the grooves")
        self.move(1,self.tcp(1)+[0,.004,.048],.65,"clear the replacement thumbstick")
        self.seat_replacement(args)

    def seat_replacement(self,args):
        rotation=grip_frame(args.seat_yaw,args.seat_pitch,args.seat_roll)
        contact=self.position("new_module_grip")+np.array(args.seat_offset)
        self.move(1,contact+[0,.054,.030],.88,"approach the front of the module",.008,rotation)
        backstop=False
        for distance in (.029,.019,.012,.008,.004,.000,-.004,-.008,-.012,-.017):
            axis=self.data.body("controller").xmat.reshape(3,3)[:,1]
            self.move(1,contact+axis*distance,.45,"push the replacement to the backstop")
            backstop=self.mechanism.load("socket_stop",other="new_module_case")>.11
            print("Seating press",self.mechanism.rel("new_module")[0].round(4),
                  self.contacts_on("right_"),backstop,flush=True)
            if self.mechanism.new_in and backstop:
                break
        self.jaws(1,args.seat_release,"release the seating pressure",duration=.32)
        self.move(1,self.tcp(1)+[0.,-.002,.045],.58,"lift clear of the seated module")
        if not self.mechanism.new_in:
            raise ValueError("the replacement module did not remain fully seated")

    def secure_lever(self, args):
        current = self.position("lever_tip")
        r = grip_frame(-18.,67.,0.)
        axis = self.data.body("controller").xmat.reshape(3,3)[:,0].copy()
        initial = np.rad2deg(self.data.joint("lever_angle").qpos[0])
        distance = current-self.position("lever_pivot")
        shift = np.array(args.close_lever_offset)
        self.move(1,current+shift+[0,.019,.064],1.01,"approach the raised lever",.006,r)
        self.move(1,self.position("lever_tip")+shift+[0,0,.021],.65,"line up to lower the lever")
        for drop in np.linspace(0.,initial+5.,9):
            at=self.position("lever_pivot")+Rotation.from_rotvec(-axis*np.deg2rad(drop)).apply(distance)
            self.move(1,at+shift,.45,"lower the module release lever")
            print("Closing lever",self.mechanism.score()["lever_deg"],
                  self.contacts_on("right_"),flush=True)
            if self.mechanism.score()["lever_deg"]<8:
                break
        self.move(1,self.tcp(1)+[0,.016,.052],.76,"clear the closed release lever",.27)
        if self.mechanism.score()["lever_deg"]>10:
            raise ValueError("the module release lever remained raised")

    def pick_cover_for_reassembly(self,args):
        self.park(1,"free the right arm for the front cover")
        inward=self.tcp_rotation(0)
        self.move(0,self.tcp(0)+[0.,0.,.050],.85,"lift the repaired controller clear of the cover",rot=inward)
        self.move(0,self.tcp(0)+args.repair_offset,.85,"set the controller apart from the front cover")
        for _ in range(2):
            height=self.lowest_controller_point(include_original=False)
            self.move(0,self.tcp(0)+[0,0,TABLE+.003-height],.58,"lower the repaired controller")
        self.jaws(0,.90,"release the repaired controller")
        self.move(0,self.tcp(0)-inward[:,2]*.069,.73,"withdraw from the controller handle",.9)
        self.park(0,"free the left arm for the front cover")
        orient=grip_frame(args.cover_pick_yaw,82.,args.cover_pick_roll)
        from_cover=np.array(args.cover_pick_local)
        offset=np.array(args.cover_pick_offset)
        point=self.relative_point("face_cover",from_cover)+offset
        self.move(0,point-orient[:,2]*.072,1.10,"approach the cover ring",args.cover_pick_gap,orient)
        for clearance in (.029,.018,.008,.001,-.004):
            point=self.relative_point("face_cover",from_cover)+offset+[0,0,clearance]
            self.move(0,point,.44,"fit the fingers across the cover openings",args.cover_pick_gap)
        self.jaws(0,args.cover_pick_grip,"grip the front cover by its rim")
        print("Left cover grasp", self.contacts_on("cover_"),
              self.mechanism.load("cover_",side="left"), flush=True)
        self.move(0,self.tcp(0)+[0.,0.,.052],.73,"lift the cover clear of the bench")
        self.move(0,self.tcp(0)+args.cover_lift,.98,"bring the cover into reach for regrasp")
        if self.data.body("face_cover").xpos[2]<TABLE+.086:
            raise ValueError("front cover stayed on the table during pickup")

    def grip_cover_edge(self,args):
        point=self.position("cover_grip")+np.array(args.handoff_offset)
        panel=self.data.body("face_cover")
        panel_rot=panel.xmat.reshape(3,3)
        controller_rot=self.data.body("controller").xmat.reshape(3,3)
        grasp_delta=controller_rot@panel_rot.T
        final_tcp=self.relative_point("controller",[0.,.009,.039])+grasp_delta@(point-panel.xpos)
        # Both jaw symmetries have identical grasp geometry. Choose the one
        # with a full wrist-turn interval before closing on the part.
        roll_range=None
        for yaw in (args.handoff_yaw,args.handoff_yaw-16.,args.handoff_yaw+16.):
            for roll in (0.,180.):
                aim=grip_frame(yaw,args.handoff_pitch,roll)
                for sign in (1.,-1.):
                    seed=HOME[:6]
                    try:
                        if not args.skip_fit_precheck:
                            fit=self.arms[1].inverse(self.plan,final_tcp,grasp_delta@aim,HOME[:6])
                            self.arms[1].inverse(self.plan,final_tcp+[0.,0.,.063],grasp_delta@aim,fit)
                        for angle in np.linspace(0.,180.,13):
                            orientation=Rotation.from_rotvec(aim[:,2]*np.deg2rad(angle)*sign).as_matrix()@aim
                            seed=self.arms[1].inverse(self.plan,point,orientation,seed)
                    except ValueError:
                        continue
                    roll_range=(aim,sign)
                    break
                if roll_range:
                    break
            if roll_range:
                break
        if roll_range is None:
            raise ValueError("no reachable continuous wrist turn for the held front cover")
        aim,sign=roll_range
        travel=grip_frame(-25.,4.,0.)
        self.move(1,point-travel[:,2]*.077,1.15,"bring the receiving gripper toward the work area",.24,travel)
        self.move(1,point-aim[:,2]*.072,.82,"align the receiving gripper away from the panel",.24,aim)
        point=self.position("cover_grip")+np.array(args.handoff_offset)
        self.move(1,point-aim[:,2]*.021,.59,"line up with the panel edge",.20,aim)
        self.move(1,point,.63,"straddle the panel edge",.15,aim)
        self.jaws(1,.002,"grip the panel edge")
        print("Right cover grasp",self.contacts_on("cover_"),flush=True)
        self.jaws(0,.48,"transfer the front cover to the right gripper")
        self.move(0,self.tcp(0)+[.021,.017,.040],.79,"clear the left gripper")
        self.park(0,"clear the work area of the left arm")
        if self.data.body("face_cover").xpos[2]<TABLE+.06:
            raise ValueError("front cover slipped from the right gripper during the transfer")
        aim=self.tcp_rotation(1)
        start=self.tcp(1)
        for angle in (20.,40.,60.,80.,100.,120.,140.,160.,180.):
            rot=Rotation.from_rotvec(aim[:,2]*np.deg2rad(angle)*sign).as_matrix()@aim
            self.move(1,start,.45,"turn the front cover to face up",rot=rot)
        print("Turned face cover",self.data.body("face_cover").xmat.reshape(3,3)[:,2].round(3),flush=True)

    def refit_cover(self,args):
        self.pick_cover_for_reassembly(args)
        self.grip_cover_edge(args)
        if args.regrasp_face_up:
            self.regrasp_face_up_cover(args)
        self.align_cover(args)

    def regrasp_face_up_cover(self,args):
        position=self.tcp(1)
        hold=self.tcp_rotation(1)
        self.move(1,position+args.panel_staging,.98,"carry the turned cover to the worktop",rot=hold)
        for _ in range(2):
            height=self.data.body("face_cover").xpos[2]
            self.move(1,self.tcp(1)+[0,0,np.clip(TABLE+.021-height,-.058,.0)],
                      .70,"lower the face-up cover to the worktop")
        self.jaws(1,.47,"set the face-up cover down")
        self.move(1,self.tcp(1)+[0,.052,.047],.69,"clear the cover from the side grip")
        self.park(1,"prepare a top grip on the face cover")
        frame=self.data.body("face_cover").xmat.reshape(3,3)
        yaw=np.rad2deg(np.arctan2(-frame[1,0],frame[0,0]))
        grip=grip_frame(yaw+args.final_grip_yaw,82.,args.final_grip_roll)
        target=np.array(args.final_grip_local)
        point=self.relative_point("face_cover",target)
        self.move(1,point-grip[:,2]*.068,1.02,"approach the face-up cover from above",.64,grip)
        for dz in (.026,.016,.010,.007,.004):
            self.move(1,self.relative_point("face_cover",target)+[0.,0.,dz],
                      .45,"straddle the front and back of the cover",.63)
        self.jaws(1,args.final_grip_gap,"grip the front and back of the cover")
        print("Cover bench grasp",self.contacts_on("cover_"),flush=True)
        self.move(1,self.tcp(1)+[0,0,.065],.82,"lift the face-up cover from the bench")
        if self.data.body("face_cover").xpos[2]<TABLE+.040:
            raise ValueError("front cover slipped from the top grip")

    def align_cover(self,args):
        tool=self.tcp_rotation(1)
        held=self.data.body("face_cover")
        rotation=self.data.body("controller").xmat.reshape(3,3)@held.xmat.reshape(3,3).T@tool
        offset=tool.T@(held.xpos-self.tcp(1))
        point=self.relative_point("controller",[0.,.009,.106])-rotation@offset
        self.move(1,point,1.47,"carry the face-up cover above the controller",rot=rotation)
        for _ in range(14):
            self.steer_held_object("face_cover",[0.,.009,.103],.45,
                                   "align the face cover over both thumbsticks")
            alignment,orientation=self.mechanism.rel("face_cover")
            if (np.linalg.norm(alignment[:2]-[0,.009])<.0012 and
                    Rotation.from_matrix(orientation).magnitude()<np.deg2rad(2)):
                break
        for dy,dz in ((.000,.040),(.000,.020),(.000,.015),(.000,.010),(.000,.006),(.000,.003),(.000,.001),
                      (.000,0.),(.000,-.0008),(.000,-.0016),(.000,-.0020)):
            seal=np.array([0.,.009+dy,.036+dz])
            self.steer_held_object("face_cover",seal,.45,"align and press the front cover")
            alignment,orientation=self.mechanism.rel("face_cover")
            print("Front cover seating",alignment.round(4),
                  np.rad2deg(Rotation.from_matrix(orientation).magnitude()).round(1),
                  self.contacts_on("cover_"),self.contacts_on("right_"),flush=True)
            if self.mechanism.relatched:
                break
        if not self.mechanism.relatched:
            self.press_cover(args)
            return
        self.jaws(1,.43,"leave the front cover clipped in place")
        self.move(1,self.tcp(1)+[-.021,.038,.044],.74,"withdraw from the repaired controller")
        self.park(1,"return the right gripper to ready")
        self.finish_trial()

    def press_cover(self,args):
        for _ in range(8):
            self.steer_held_object("face_cover",[0.,.009,.0475],.44,
                                   "center the guided cover around both sticks")
            p,r=self.mechanism.rel("face_cover")
            print("Centered cover",p.round(4),np.rad2deg(Rotation.from_matrix(r).magnitude()).round(1),flush=True)
            if np.linalg.norm(p[:2]-[0.,.009])<.003 and p[2]<.052:
                break
        self.jaws(1,.32,"release the aligned cover onto the guide")
        back=self.tcp_rotation(1)[:,2]
        self.move(1,self.tcp(1)-back*.042,.76,"withdraw the cover edge grip",.45)
        self.move(1,self.tcp(1)+[-.012,.014,.074],.8,"leave the face cover free",.55)
        self.ramp(self.now,.38,"let the cover settle around the thumbsticks")
        mounted,_=self.mechanism.rel("face_cover")
        if np.linalg.norm(mounted-[0,.009,.036])>.018:
            raise ValueError("the cover left its guides when released")
        self.seating_press(args)

    def seating_press(self,args):
        if args.cover_brace:
            support=grip_frame(32.,76.,-90.)
            local=[.067,.029,.040]
            self.move(0,self.relative_point("controller",local)+[0,.014,.064],1.06,
                      "approach the outside controller handle",.005,support)
            self.move(0,self.relative_point("controller",local)+[0,0,.014],.67,
                      "line up to stabilize the handle")
            self.move(0,self.relative_point("controller",local)-[0,0,.003],.62,
                      "stabilize the controller handle for seating")
        aim=grip_frame(args.cover_press_yaw,args.cover_press_pitch,args.cover_press_roll)
        target=self.relative_point("face_cover",args.clip_point)
        self.move(1,target+[0,.018,.060],1.02,"approach to press the center of the cover",.005,aim)
        for height in (.029,.018,.011,.006,.002,.0,-.002,-.005,-.008):
            target=self.relative_point("face_cover",args.clip_point)
            self.move(1,target+[0.,0.,height],.43,"press the cover onto its retaining clips")
            print("Clip press",self.mechanism.rel("face_cover")[0].round(4),
                  np.rad2deg(Rotation.from_matrix(self.mechanism.rel("face_cover")[1]).as_rotvec()).round(2),
                  self.mechanism.relatched,self.contacts_on("right_"),flush=True)
            if self.mechanism.relatched:
                break
        if not self.mechanism.relatched:
            self.register_cover_edge(args)
        if not self.mechanism.relatched:
            raise ValueError("front cover did not latch after its physical seating press")
        self.move(1,self.tcp(1)+[0.,.018,.062],.71,"withdraw from the secured front cover",.45)
        self.park(1,"return the right gripper to ready")
        if args.cover_brace:
            self.move(0,self.tcp(0)+[.021,.030,.058],.76,"release support from the controller handle",.35)
            self.park(0,"return the left gripper to ready")
        self.finish_trial()

    def register_cover_edge(self,args):
        p,_=self.mechanism.rel("face_cover")
        if np.linalg.norm(p-[0.,.009,.036])>.011:
            return
        angle=grip_frame(-13.,78.,0.)
        position=self.relative_point("face_cover",args.clip_nudge_point)
        self.move(1,self.tcp(1)+[0,.007,.041],.58,"clear the seam before aligning the raised edge",.015,angle)
        self.move(1,position+[0,0,.026],.62,"approach a raised corner of the cover")
        self.move(1,position+[0,0,.004],.57,"line up with the raised corner")
        self.move(1,position-[0,0,.0015],.38,"engage the raised corner")
        reference=self.tcp(1)
        frame=self.data.body("controller").xmat.reshape(3,3).copy()
        for amount in (.25,.50,.8,1.15,1.5,1.9):
            target=reference+frame@(amount*np.array(args.clip_nudge))
            self.move(1,target,.44,"seat the raised front edge into its notch")
            p,r=self.mechanism.rel("face_cover")
            print("Edge fit",p.round(4),np.rad2deg(Rotation.from_matrix(r).as_rotvec()).round(1),
                  self.mechanism.relatched,self.contacts_on("right_"),flush=True)
            if self.mechanism.relatched:
                break

    def finish_trial(self):
        old,_=self.mechanism.rel("old_module")
        if np.linalg.norm(old-MOUNT)<.070:
            rotation=grip_frame(-20.,77.,8.)
            at=self.position("old_module_grip")+[.002,.002,.007]
            self.move(1,at-rotation[:,2]*.062,1.04,"approach the removed module",.25,rotation)
            self.move(1,at,.76,"straddle the removed module",.23)
            self.jaws(1,.003,"pick up the removed module")
            self.move(1,self.tcp(1)+[0.,0.,.039],.69,"lift the removed module")
            self.move(1,self.tcp(1)+[-.081,.006,0.],.85,"place the removed module away from the controller")
            height=self.data.body("old_module").xpos[2]
            self.move(1,self.tcp(1)+[0.,0.,TABLE+.0085-height],.71,"lower the original module to the tabletop")
            self.jaws(1,.41,"release the removed module on the table")
            self.move(1,self.tcp(1)+[0.,.025,.058],.67,"move the gripper clear of both items")
            self.park(1,"return both grippers to ready")
        self.ramp(self.now,1.2,"inspect the repaired controller at rest")

    def script(self, args):
        if args.mode == "hands-off":
            self.ramp(self.now, ACTION_LIMIT/CONTROL_HZ, "leave the controller untouched")
            return
        self.ramp(self.now, .55, "controller at rest")
        self.brace(args)
        if args.mode == "full":
            self.unlatch(args)
            if not self.mechanism.shown:
                self.remove_cover(args)
            self.turn_controller(args)
            self.brace(args, face_up=True)
            self.lift_lever(args)
            self.remove_module(args)
            self.fit_replacement(args)
            self.secure_lever(args)
            self.refit_cover(args)
        else:
            self.lift_lever(args)
            if args.mode == "exchange":
                self.remove_module(args)
        self.ramp(self.now, .8, "inspect mechanism")


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    mode = "full" if args.mode=="hands-off" else args.mode
    xml = build(mode)
    (args.out/"scene.xml").write_text(xml)
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    physics = Mechanism(model, data, mode)
    failure = None
    with ExitStack() as resources:
        stream = resources.enter_context((args.out/"trace.jsonl").open("w"))
        trial = Trial(model, data, physics, stream)
        if args.record:
            from review_media import StationViews, review_writer, save_setup
            renderer=StationViews(model,CAMERAS)
            resources.callback(renderer.close)
            write=resources.enter_context(review_writer(args.out/"demo.mp4"))
            save_setup(renderer.capture(data),args.out/"setup.png")
            trial.capture=lambda: write(renderer.capture(data))
        try:
            if args.replay:
                trial.phase = "recorded joint commands"
                for action in np.load(args.replay):
                    trial.step(action)
                if args.continue_lever:
                    trial.lift_lever(args)
                if args.continue_remove:
                    trial.remove_module(args)
                if args.continue_insert:
                    trial.fit_replacement(args)
                if args.continue_seat:
                    trial.seat_replacement(args)
                if args.continue_relock:
                    trial.secure_lever(args)
                if args.continue_cover:
                    trial.refit_cover(args)
                if args.continue_handoff:
                    trial.grip_cover_edge(args)
                    if args.regrasp_face_up:
                        trial.regrasp_face_up_cover(args)
                    trial.align_cover(args)
                if args.continue_regrasp:
                    trial.regrasp_face_up_cover(args)
                    trial.align_cover(args)
                if args.continue_press_cover:
                    trial.press_cover(args)
                if args.continue_seam:
                    trial.seating_press(args)
            else:
                trial.script(args)
        except ValueError as err:
            failure = str(err)
            print(failure, flush=True)
        finally:
            if args.record:
                save_setup(renderer.capture(data),args.out/"final.png")
    np.save(args.out/"actions.npy", trial.actions)
    report = physics.score()
    report.update(action_count=len(trial.actions), failure=failure, mode=args.mode,
                  control_hz=CONTROL_HZ, peak_effort_nm=trial.peak_effort.tolist())
    (args.out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full","exchange","lever","hands-off"), default="full")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--live-script",action="store_true",help="Use the experimental motion planner")
    parser.add_argument("--continue-lever", action="store_true")
    parser.add_argument("--continue-remove", action="store_true")
    parser.add_argument("--continue-insert", action="store_true")
    parser.add_argument("--continue-relock", action="store_true")
    parser.add_argument("--continue-seat", action="store_true")
    parser.add_argument("--continue-cover", action="store_true")
    parser.add_argument("--continue-handoff", action="store_true")
    parser.add_argument("--continue-regrasp", action="store_true")
    parser.add_argument("--continue-press-cover", action="store_true")
    parser.add_argument("--continue-seam", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--brace-y", type=float, default=.026)
    parser.add_argument("--brace-z", type=float, default=.019)
    parser.add_argument("--brace-yaw", type=float, default=42.)
    parser.add_argument("--brace-pitch", type=float, default=43.)
    parser.add_argument("--brace-roll", type=float, default=-90.)
    parser.add_argument("--brace-close", type=float, default=.09)
    parser.add_argument("--latch-yaw", type=float, default=-65.)
    parser.add_argument("--latch-pitch", type=float, default=74.)
    parser.add_argument("--latch-roll", type=float, default=-90.)
    parser.add_argument("--latch-offset", type=float, nargs=3, default=(-.002,.001,.007))
    parser.add_argument("--cover-yaw", type=float, default=-43.)
    parser.add_argument("--cover-pitch", type=float, default=32.)
    parser.add_argument("--cover-roll", type=float, default=0.)
    parser.add_argument("--cover-offset", type=float, nargs=3, default=(0.,.012,-.002))
    parser.add_argument("--turn-axis", type=float, default=-110.)
    parser.add_argument("--turn-deg", type=float, default=102.)
    parser.add_argument("--lever-yaw", type=float, default=-12.)
    parser.add_argument("--lever-pitch", type=float, default=29.)
    parser.add_argument("--lever-roll", type=float, default=-89.)
    parser.add_argument("--lever-offset", type=float, nargs=3, default=(-.002,.004,.004))
    parser.add_argument("--lever-opening", type=float, default=.145)
    parser.add_argument("--lever-tilt", type=float, default=65.)
    parser.add_argument("--module-yaw", type=float, default=0.)
    parser.add_argument("--module-pitch", type=float, default=80.)
    parser.add_argument("--module-roll", type=float, default=0.)
    parser.add_argument("--module-offset", type=float, nargs=3, default=(.003,.002,.002))
    parser.add_argument("--module-opening", type=float, default=.205)
    parser.add_argument("--new-grip",type=float,nargs=3,default=(.006,.002,.008))
    parser.add_argument("--new-yaw",type=float,default=-20.)
    parser.add_argument("--new-pitch",type=float,default=84.)
    parser.add_argument("--new-roll",type=float,default=23.)
    parser.add_argument("--guide-x",type=float,default=0.)
    parser.add_argument("--guide-z",type=float,default=0.)
    parser.add_argument("--seat-yaw",type=float,default=17.)
    parser.add_argument("--seat-pitch",type=float,default=46.)
    parser.add_argument("--seat-roll",type=float,default=-90.)
    parser.add_argument("--seat-release",type=float,default=.23)
    parser.add_argument("--seat-offset",type=float,nargs=3,default=(.004,.005,.008))
    parser.add_argument("--close-lever-offset",type=float,nargs=3,default=(0.,.004,.007))
    parser.add_argument("--cover-pick-yaw",type=float,default=-4.)
    parser.add_argument("--cover-pick-roll",type=float,default=-90.)
    parser.add_argument("--cover-pick-local",type=float,nargs=3,default=(0.,.005,0.))
    parser.add_argument("--cover-pick-offset",type=float,nargs=3,default=(0.,0.,.009))
    parser.add_argument("--cover-pick-gap",type=float,default=.42)
    parser.add_argument("--cover-pick-grip",type=float,default=.78)
    parser.add_argument("--cover-lift",type=float,nargs=3,default=(.057,.038,.064))
    parser.add_argument("--handoff-yaw",type=float,default=0.)
    parser.add_argument("--handoff-pitch",type=float,default=0.)
    parser.add_argument("--handoff-offset",type=float,nargs=3,default=(0.,.007,.0004))
    parser.add_argument("--skip-fit-precheck",action="store_true")
    parser.add_argument("--regrasp-face-up",action="store_true")
    parser.add_argument("--panel-staging",type=float,nargs=3,default=(-.132,-.040,0.))
    parser.add_argument("--final-grip-local",type=float,nargs=3,default=(0.,.008,0.))
    parser.add_argument("--final-grip-yaw",type=float,default=0.)
    parser.add_argument("--final-grip-roll",type=float,default=0.)
    parser.add_argument("--final-grip-gap",type=float,default=.33)
    parser.add_argument("--cover-brace",action="store_true")
    parser.add_argument("--clip-point",type=float,nargs=3,default=(0.,.004,.008))
    parser.add_argument("--cover-press-yaw",type=float,default=-13.)
    parser.add_argument("--cover-press-pitch",type=float,default=80.)
    parser.add_argument("--cover-press-roll",type=float,default=0.)
    parser.add_argument("--clip-nudge-point",type=float,nargs=3,default=(-.013,.022,.009))
    parser.add_argument("--clip-nudge",type=float,nargs=3,default=(.003,.004,-.0015))
    parser.add_argument("--repair-offset",type=float,nargs=3,default=(-.059,.034,.0))
    args=parser.parse_args()
    if args.mode=="full" and args.replay is None and not args.live_script:
        args.replay=HERE/"scripted-actions.npy"
    run(args)
