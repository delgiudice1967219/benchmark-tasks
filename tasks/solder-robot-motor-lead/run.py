"""Develop and replay joint commands for the bench soldering task."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from motion import Arm, ARMS, HOME, direction
from review_media import CONTROL_HZ, StationViews, review_writer, save_setup
from splice import CAMERAS, IRONS, Splice, build_xml

ACTION_LIMIT=1800
EFFORT=np.array((27.,27.,27.,7.,7.,7.))


class Runner:
    def __init__(self,model,data,physics,trace,record=None):
        self.model,self.data,self.physics=model,data,physics
        self.planning=mj.MjData(model)
        self.arms=[Arm(model,side) for side in ARMS]
        self.command=np.stack([HOME,HOME])
        self.poses=[direction(22,66,False), direction(-22,66,False)]
        for arm in self.arms:
            model.actuator_forcerange[arm.actuators]=np.column_stack((-EFFORT,EFFORT))
            arm.reset(data)
        mj.mj_forward(model,data)
        self.peak_effort=np.zeros((2,6))
        self.peak_arm_speed=np.zeros((2,6))
        self.peak_finger_speed=np.zeros((2,2))
        self.peak_finger_force=np.zeros((2,2))
        self.finger_dofs=[[model.joint(name).dofadr[0] for name in arm.finger_names] for arm in self.arms]
        self.targets=[]
        self.rows=[]
        self.phase="reset"
        self.trace=trace
        self.record=record
        self.steps_per_action=round(1/(CONTROL_HZ*model.opt.timestep))
        self.grip=np.zeros((3,2))
        self.pose_error=[]

    def tick(self,targets):
        if len(self.targets) >= ACTION_LIMIT:
            raise ValueError("Action limit reached")
        for index,(arm,target) in enumerate(zip(self.arms,targets)):
            arm.command(self.data,target)
        for _ in range(self.steps_per_action):
            self.physics.apply()
            mj.mj_step(self.model,self.data)
            self.grip=self.physics.observe()
            for i,arm in enumerate(self.arms):
                self.peak_effort[i]=np.maximum(self.peak_effort[i],abs(self.data.actuator_force[arm.actuators]))
                self.peak_arm_speed[i]=np.maximum(self.peak_arm_speed[i],abs(self.data.qvel[arm.dofs]))
                self.peak_finger_force[i]=np.maximum(self.peak_finger_force[i],abs(self.data.actuator_force[arm.fingers]))
                self.peak_finger_speed[i]=np.maximum(self.peak_finger_speed[i],abs(self.data.qvel[self.finger_dofs[i]]))
        self.command=targets.copy()
        self.targets.append(self.command)
        row=self.physics.measure()
        row.update(phase=self.phase,action=len(self.targets),holding_n=self.grip.tolist(),
                   poses={key:self.data.body(key).xpos.tolist() for key in ("iron","solder")},
                   tool_tips={key:self.physics.xpos(key+"_tip_point").tolist() for key in ("iron","solder")},
                   grippers=[self.data.site_xpos[arm.site].tolist() for arm in self.arms])
        self.rows.append(row)
        self.trace.write(json.dumps(row)+"\n")
        if len(self.targets)%CONTROL_HZ==0:
            print(f"{self.data.time:.1f}s {self.phase}: {row['clip_angle_deg']:.2f} deg, "
                  f"{min(row['conductor_temperatures_c']):.0f} C, {row['solder_feed_mm']:.2f} mm",
                  flush=True)
        if self.record is not None:
            self.record()

    def report(self):
        return dict(self.physics.measure(), actions=len(self.targets),
                    peak_joint_effort_nm=self.peak_effort.tolist(),
                    peak_joint_speed_rad_s=self.peak_arm_speed.tolist(),
                    peak_finger_force_n=self.peak_finger_force.tolist(),
                    peak_finger_speed_m_s=self.peak_finger_speed.tolist())

    def joints(self,goals,duration,phase,soft=True):
        self.phase=phase
        start=self.command.copy()
        steps=max(1,round(duration*CONTROL_HZ))
        for k in range(1,steps+1):
            frac=k/steps
            if soft:
                frac=frac*frac*(3-2*frac)
            self.tick(start+(goals-start)*frac)

    def hold(self,duration,phase=None):
        self.joints(self.command,duration,phase or self.phase)

    def candidates(self,side,target,desired=None,alternates=True):
        arm=self.arms[side]
        current=self.command[side,:6]
        standard=self.poses[side]
        if desired is not None:
            rotations=[desired]
        elif alternates:
            flips=(not bool(side),bool(side))
            rotations=[standard]+[direction(yaw,pitch,flip) for pitch in (65,80,56,72) for yaw in (0,-25,25,-48,48)
                                 for flip in flips]
        else:
            rotations=[standard]
        solution=None
        for rot in rotations:
            try:
                q=arm.inverse(self.planning,np.asarray(target),rot,current)
                solution=q,rot
                break
            except ValueError:
                pass
        if solution is None:
            raise ValueError(f"{arm.side} unreachable at {target}")
        return solution

    def move(self,side,point,duration,phase,opening=None,orientation=None,alternates=True):
        target=np.asarray(point,dtype=float)
        q,rotation=self.candidates(side,target,orientation,alternates)
        command=self.command.copy()
        command[side,:6]=q
        if opening is not None:
            command[side,6]=opening
        self.poses[side]=rotation
        self.joints(command,duration,phase)
        actual=self.data.site_xpos[self.arms[side].site]
        self.pose_error.append([phase,side,float(np.linalg.norm(actual-target))])

    def tcp(self,side):
        return self.data.site_xpos[self.arms[side].site].copy()

    def release(self,side,opening,phase):
        goal=self.command.copy()
        goal[side,6]=opening
        self.joints(goal,.55,phase)

    def retreat(self,side,phase):
        goal=self.command.copy()
        goal[side]=HOME
        self.joints(goal,1.4,phase)

    def set_clip(self):
        side=0
        point=self.physics.xpos("thumb_grip")
        frame=direction(48,45,False)
        self.move(side,point+(.079,.007,.032),1.25,"approach the movable clamp",.03,
                  orientation=frame,alternates=False)
        self.move(side,point+(.027,0,.002),.80,"line up outside the clamp thumb",.03,alternates=False)
        for dx in (.022,.016,.011,.006,.001,-.004,-.008,-.013,-.017,-.021,-.025,-.029,-.033,-.037,-.041,-.045):
            self.move(side,point+(dx,0,.001),.28,"swing the clamp into copper contact",alternates=False)
            if self.physics.stable_alignment>.3:
                break
        self.move(side,point+(.075,0,.036),.90,"back away from the clamp",alternates=False)
        self.retreat(side,"return left hand from clamp")

    def grasp_tool(self,side,key):
        grasp=self.physics.xpos(key+"_grip")
        front=0. if key=="iron" else .002
        frame=direction(0,79,False)
        self.move(side,grasp+(front,.010,.074),1.3,f"approach the {key}",.45,orientation=frame)
        self.move(side,grasp+(front,0.,-.006),.95,f"straddle the {key} barrel",.39,alternates=False)
        self.release(side,.06,f"grip {key}")
        print(f"{key} grip",self.grip[0 if key=="iron" else 1].tolist(),
              "TCP",self.tcp(side),"tool",self.physics.xpos(key+"_grip"),flush=True)
        self.move(side,self.tcp(side)+(0,.004,.054),1.25,f"lift {key} from holder",alternates=False)
        lifted=self.physics.xpos(key+"_grip")
        if lifted[2]<grasp[2]+.018 or min(self.grip[0 if key=="iron" else 1])<1.:
            raise ValueError(f"{key} missed stable pickup: before {grasp}, after {lifted}")
        print(f"{key} lifted",lifted,"tip",self.physics.xpos(key+"_tip_point"),"contacts",self.grip[0 if key=="iron" else 1],flush=True)

    def tool_pose(self, side, key, point):
        """Select a reachable tool axis from calibrated end-effector tracking."""
        arm=self.arms[side]
        tcp=self.tcp(side)
        tcp_rot=self.data.site_xmat[arm.site].reshape(3,3)
        body_rot=self.data.body(key).xmat.reshape(3,3)
        tip_local=tcp_rot.T@(self.physics.xpos(key+"_tip_point")-tcp)
        rel=tcp_rot.T@body_rot
        spec=IRONS[key]
        for pitch in (.18,.31,.44,.07,.56,-.08,-.24,.66):
            for yaw_adjust in (0.,.09,-.09,.20,-.20,.31,-.31):
                tool_rot=Rotation.from_euler("zy",[spec["yaw"]+yaw_adjust,pitch]).as_matrix()
                goal_rot=tool_rot @ rel.T
                goal=np.asarray(point)-goal_rot @ tip_local
                try:
                    self.candidates(side,goal,desired=goal_rot,alternates=False)
                    return goal,goal_rot
                except ValueError:
                    pass
        raise ValueError(f"{key} has no reachable contact orientation at {point}")

    def place_tip(self,side,key,point,duration,phase,choose_axis=False,correct=False):
        target=np.asarray(point)
        if choose_axis:
            position,orientation=self.tool_pose(side,key,target)
            self.move(side,position,duration,phase,orientation=orientation,alternates=False)
        else:
            offset=self.physics.xpos(key+"_tip_point")-self.tcp(side)
            self.move(side,target-offset,duration,phase,alternates=False)
        if correct:
            for _ in range(4):
                error=target-self.physics.xpos(key+"_tip_point")
                if np.linalg.norm(error)<.00046:
                    return
                self.move(side,self.tcp(side)+np.clip(error,-.006,.006),.42,phase,alternates=False)

    def warm_and_feed(self,skip_solder,miss_iron,misaligned):
        fixed=self.physics.xpos("fixed_center")
        free=self.physics.xpos("free_center")
        # Heat the overlapping bare section, touching from the outer side of
        # the fixed conductor. Approach from beyond the cold handpiece guard.
        copper_y=float((fixed[1]+free[1])/2)
        point=np.array(((fixed[0]+free[0])/2,copper_y-.0004,fixed[2]+.0030))
        if miss_iron:
            point[1]-=.013
        self.place_tip(1,"iron",point+(-.028,.0,.021),1.3,"position the iron tip in free space",
                       choose_axis=True)
        self.place_tip(1,"iron",point,.9,"approach the bare copper",choose_axis=True,correct=True)
        last_tcp=self.tcp(1)
        for dz in (.0008,.0013,.0019,.0025,.0030,.0035,.0042,.0047):
            self.move(1,last_tcp+(0,0,-dz),.30,"press the iron tip against the copper",alternates=False)
            if self.physics.recent_force[1]+self.physics.recent_force[2]>.15:
                break
        self.hold(.8,"transfer heat through the iron contact")
        if skip_solder:
            self.hold(3.0,"keep the second hand clear")
            return
        pen_point=np.array((free[0]+.0028,copper_y+.0011,free[2]+.00015))
        self.place_tip(0,"solder",pen_point+(.022,0,.020),1.3,"bring the solder to the opposite side",
                       choose_axis=True)
        self.place_tip(0,"solder",pen_point,.85,"approach the heated bare copper",choose_axis=True,correct=True)
        if misaligned:
            self.hold(1.0,"attempt to solder leads with a gap")
            return
        last_tcp=self.tcp(0)
        for dx in (.0008,.0013,.0020,.0025,.0030,.0037,.0044,.0052,.0058):
            self.move(0,last_tcp+(-dx,0.,0.),.26,"feed the solder against the heated conductors",alternates=False)
            if self.physics.fed_m>.0018:
                break
        self.hold(.45,"allow solder to wet the overlapping copper")

    def transport(self,side,points,duration,phase):
        """Use a reachable position above the bench while retaining the grasp."""
        initial=self.poses[side].copy()
        for point in points:
            for delta in ((0.,0.,0.),(0.,0.,.10),(0.,0.,-.10),
                          (0.,.09,0.),(0.,-.09,0.),(0.,0.,.20),(0.,0.,-.20)):
                orientation=Rotation.from_rotvec(delta).as_matrix()@initial
                try:
                    self.move(side,point,duration,phase,orientation=orientation,alternates=False)
                    return
                except ValueError:
                    pass
        raise ValueError(f"Cannot reach a supported approach for {phase}")

    def put_back(self,side,key):
        start=IRONS[key]["origin"]
        now=self.tcp(side)
        retreat=(-.015 if key=="iron" else .015)
        self.transport(side,[now+(retreat,.011,height) for height in (.031,.022,.040)],
                       .85,f"lift {key} clear of the joint")
        delta=start-(self.data.body(key).xpos.copy()-self.tcp(side))
        self.transport(side,[delta+(0.,.008,height) for height in (.043,.032,.055)],
                       1.25,f"carry the {key} above its stand")
        if key=="iron":
            axis=self.data.body(key).xmat.reshape(3,3)[:,0]
            yaw=np.arctan2(axis[1],axis[0])
            if abs(yaw)>.055:
                tcp=self.tcp(side)
                body_offset=self.data.body(key).xpos.copy()-tcp
                orientation=self.data.site_xmat[self.arms[side].site].reshape(3,3)
                for fraction in (1.,.72,.45):
                    correction=Rotation.from_rotvec(np.array((0.,0.,-yaw*fraction))).as_matrix()
                    try:
                        self.move(side,start+(0,0,.043)-correction@body_offset,.80,
                                  "align the iron with the guard",orientation=correction@orientation,
                                  alternates=False)
                        break
                    except ValueError:
                        pass
        delta=start-(self.data.body(key).xpos.copy()-self.tcp(side))
        self.transport(side,[delta+(0.,0.,height) for height in (.001,.005,.008)],
                       .90,f"set {key} back in its cradle")
        self.release(side,.49,f"release {key} in the holder")
        self.move(side,self.tcp(side)+(0,.008,.059),.90,f"clear the {key} holder",.63,alternates=True)
        self.retreat(side,f"withdraw from {key} holder")

    def script(self,mode):
        self.hold(.8,"settle")
        if mode=="hands-off":
            self.hold((ACTION_LIMIT-len(self.targets))/CONTROL_HZ,"no actions")
            return
        if mode!="gap":
            self.set_clip()
        if mode=="align-only":
            self.hold(10.)
            return
        self.grasp_tool(1,"iron")
        if mode!="no-solder":
            self.grasp_tool(0,"solder")
        self.warm_and_feed(mode=="no-solder",mode=="no-contact",mode=="gap")
        if mode!="no-solder":
            current=self.tcp(0)
            self.move(0,current+(.012,0,.013),.65,"stop feeding solder",alternates=False)
        current=self.tcp(1)
        self.move(1,current+(-.015,0,.014),.65,"remove heat from splice",alternates=False)
        self.put_back(1,"iron")
        if mode!="no-solder":
            self.put_back(0,"solder")
        self.hold(1.4,"arms clear")


def run(args):
    args.out.mkdir(parents=True)
    xml=build_xml()
    model=mj.MjModel.from_xml_string(xml)
    data=mj.MjData(model)
    physics=Splice(model,data,args.mode)
    status="complete"
    error=None
    with ExitStack() as resources:
        trace=resources.enter_context((args.out/"trace.jsonl").open("w"))
        trial=Runner(model,data,physics,trace)
        if args.record:
            views=StationViews(model,CAMERAS)
            resources.callback(views.close)
            write=resources.enter_context(review_writer(args.out/"review.mp4"))
            initial=views.capture(data)
            save_setup(initial,args.out/"setup.png")
            write(initial)
            trial.record=lambda: write(views.capture(data))
        try:
            if args.replay:
                actions=np.load(args.replay,allow_pickle=False)
                if actions.ndim != 3 or actions.shape[1:] != (2,7) or not np.isfinite(actions).all():
                    raise ValueError("Replay requires finite arrays of two seven-value joint targets")
                trial.phase="joint replay"
                for goals in actions:
                    trial.tick(goals)
            else:
                trial.script(args.mode)
        except ValueError as exc:
            status="motion failed"
            error=str(exc)
            print(exc,flush=True)
        if args.record:
            save_setup(views.capture(data),args.out/"final.png")
            views.renderer.update_scene(data,camera="joint_inspection",scene_option=views.options)
            import cv2
            cv2.imwrite(str(args.out/"joint-after.png"), cv2.cvtColor(views.renderer.render(),cv2.COLOR_RGB2BGR))
    np.save(args.out/"actions.npy",np.asarray(trial.targets))
    np.savez(args.out/"final_state.npz",qpos=data.qpos,qvel=data.qvel)
    report=trial.report()
    report.update(status=status,error=error,mode=args.mode,pose_error_m=trial.pose_error)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("score",report["score"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--record",action="store_true")
    parser.add_argument("--replay",type=Path)
    parser.add_argument("--mode",choices=("solder","hands-off","align-only","cold","gap","no-solder","no-contact"),
                        default="solder")
    run(parser.parse_args())
