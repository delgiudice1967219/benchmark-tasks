"""Two-arm contact simulation of mounting and hand-soldering a radial capacitor."""
import argparse
import os
from pathlib import Path
import json

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/50_mesa.json')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('EGL_PLATFORM','surfaceless')
os.environ.setdefault('MESA_SHADER_CACHE_DISABLE','true')

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from controller import ARMS, CLOSED, HOME, OPEN, Motion, direction, horizontal_pick, tool_axis, turn_tool, workspace_pair
from material import Solder
from scene import CAP_LEN, CAP_PICK, PCB_T, PCB_Z, PAD_XY, IRON_PICK, WIRE_PICK, IRON_REACH, WIRE_REACH, scene_xml

IRON_AXIS=tool_axis(-57,0)
WIRE_AXIS=tool_axis(55,0)
START_AXES={'iron':IRON_AXIS,'wire':WIRE_AXIS}
WORK_AXES={'iron':tool_axis(-80,15),'wire':tool_axis(80,15)}
TOOL_ROLL={'iron':35,'wire':-35}
WORK_TWIST={'iron':20,'wire':-20}
PICK_CENTER={'iron':IRON_PICK,'wire':WIRE_PICK}
POINTS={'iron':'iron_point','wire':'solder_point'}
REACH={'iron':IRON_REACH,'wire':WIRE_REACH}
CAP_HOLD=direction(16,20,True)@Rotation.from_euler('z',90,degrees=True).as_matrix()
CAP_SEAT=np.array((PAD_XY[:,0].mean(),PAD_XY[0,1],PCB_Z-PCB_T-CAP_LEN/2))


def body_to_hand(motion, name, side):
    point,frame=motion.frame(side)
    body=motion.data.body(name)
    body_rot=body.xmat.reshape(3,3)
    return frame.T@(body.xpos-point),frame.T@body_rot


def compatible_rotation(rotation, current_axis, wanted):
    return turn_tool(rotation,current_axis,wanted)


def move_part(motion, side, name, position, relative, rotation, seconds, label, strategy='cart'):
    offset, _=relative
    requested=np.asarray(position)-rotation@offset
    motion.cart(side,requested,rotation,CLOSED,seconds,label,strategy)


def load_capacitor(motion, args):
    pos=np.array(motion.data.body('capacitor').xpos)
    pick=pos+np.array((args.cap_x,args.cap_y,args.cap_z))
    left='left'
    before=pick-CAP_HOLD[:,2]*.056
    motion.cart(left,before+[0,0,.067],CAP_HOLD,.31,1.8,'position above capacitor','joint')
    motion.cart(left,before,CAP_HOLD,.31,1.2,'approach capacitor')
    motion.cart(left,pick,CAP_HOLD,.28,1.3,'straddle capacitor')
    motion.jaw(left,CLOSED,.55,'grip capacitor')
    relative=body_to_hand(motion,'capacitor',left)
    move_part(motion,left,'capacitor',pos+[0,0,.042],relative,CAP_HOLD,.95,'lift capacitor')
    if motion.data.body('capacitor').xpos[2] < pos[2]+.020:
        raise ValueError('Capacitor did not remain between the gripper pads during lift')
    near=CAP_SEAT+(0,.042,-.013)
    move_part(motion,left,'capacitor',near,relative,CAP_HOLD,1.55,'approach solder fixture')
    pre=CAP_SEAT+(0,0,-.0085)
    move_part(motion,left,'capacitor',pre,relative,CAP_HOLD,1.4,'align below through holes')
    for _ in range(2):
        pos0=motion.data.body('capacitor').xpos.copy()
        grip,_=motion.frame(left)
        motion.cart(left,grip+pre-pos0,CAP_HOLD,CLOSED,.35,'correct lead alignment')
    move_part(motion,left,'capacitor',CAP_SEAT+(args.seat_x,args.seat_y,args.seat_z),body_to_hand(
        motion,'capacitor',left),CAP_HOLD,1.2,'insert both leads from component side')
    motion.hold(.35,'verify seated leads')
    motion.jaw(left,.30,.45,'let go of capacitor')
    start,_=motion.frame(left)
    motion.cart(left,start+np.array((.003,.037,-.013)),CAP_HOLD,.30,1.15,'clear board edge')
    motion.move(1.2,'arm clear',left=HOME.copy())
    motion.hold(.35,'component retained by compliant leads')


def pick_tool(motion,side,name, args):
    axis=START_AXES[name]
    angle=np.deg2rad(TOOL_ROLL[name])
    rotation=Rotation.from_rotvec(axis*angle).as_matrix()@horizontal_pick(axis,True)
    body=motion.data.body(name)
    point=body.xpos.copy() + np.array((0,0,args.iron_z if name=='iron' else args.wire_z))
    true_axis=body.xmat.reshape(3,3)[:,2].copy()
    rotation=compatible_rotation(rotation,axis,true_axis)
    approach=point - rotation[:,2]*.041
    motion.cart(side,approach+[0,0,.043],rotation,.48,1.6,f'position above {name}','joint')
    motion.cart(side,approach,rotation,.48,1.2,f'approach {name}')
    motion.cart(side,point,rotation,.45,1.10,f'straddle {name}')
    motion.jaw(side,CLOSED if args.mode!='no-tools' else .38,.50,f'grip {name}')
    motion.cart(side,point+[0,0,.035],rotation,CLOSED if args.mode!='no-tools' else .38,.95,f'lift {name}')
    working_axis=WORK_AXES[name]
    work_rot=Rotation.from_rotvec(working_axis*np.deg2rad(WORK_TWIST[name])).as_matrix()@turn_tool(
        rotation,true_axis,working_axis)
    return point,rotation,work_rot


def tool_target(motion,side,name,position,rotation,seconds,label,strategy='cart',jaw=CLOSED):
    site=motion.data.site(POINTS[name]).xpos.copy()
    ee,frame=motion.frame(side)
    offset=frame.T@(site-ee)
    motion.cart(side,np.array(position)-rotation@offset,rotation,jaw,seconds,label,strategy)


def settle_tip(motion,side,name,target,rotation,label,attempts=3):
    arm=motion.arms[side]
    for index in range(attempts):
        observed=motion.data.site(POINTS[name]).xpos.copy()
        offset=target-observed
        if np.linalg.norm(offset)<.00024:
            break
        motion.planning.qpos[:]=motion.data.qpos
        motion.planning.qpos[arm.address]=motion.commands[side][:6]
        mj.mj_kinematics(motion.model,motion.planning)
        commanded=motion.planning.site_xpos[arm.site].copy()
        motion.cart(side,commanded+np.clip(offset,-.003,.003),rotation,CLOSED,
                    .20,label)

def put_down(motion,side,name,point,rotation):
    motion.cart(side,point+[0,0,.049],rotation,CLOSED,1.7,f'return {name} to holder','joint')
    motion.cart(side,point+[0,0,.0008],rotation,CLOSED,1.1,f'lower {name} into holder')
    motion.jaw(side,.42,.4,f'let go of {name}')
    motion.cart(side,point+[0,0,.047],rotation,.45,.9,f'clear {name} holder')
    motion.move(1.1,f'{side} arm clear',**{side:HOME.copy()})


def solder_joints(motion,args):
    iron_pick,iron_rotation,iron_rot=pick_tool(motion,'left','iron',args)
    wire_pick,wire_rotation,wire_rot=pick_tool(motion,'right','wire',args)
    if args.mode=='tools':
        return
    points = [(np.array((x+.00015,y,PCB_Z+args.iron_depth)),
               np.array((x+args.feed_x,y+args.feed_y,PCB_Z+args.wire_depth)))
              for x,y in PAD_XY]
    if args.search:
        iron_rot,wire_rot=workspace_pair(motion,
            {'left':points[0][0],'right':points[0][1]},
            {'left':'iron','right':'wire'})
    else:
        rotations=[]
        for side,name,angles in (
            ('left','iron',(-85,15,0)),('right','wire',(72,15,-42)),
        ):
            axis=motion.data.body(name).xmat.reshape(3,3)[:,2].copy()
            _,rotation=motion.frame(side)
            tip_axis=tool_axis(*angles[:2])
            rotations.append(Rotation.from_rotvec(tip_axis*np.deg2rad(angles[2])).as_matrix()
                             @turn_tool(rotation,axis,tip_axis))
        iron_rot,wire_rot=rotations
    for i,(hot,feed) in enumerate(points):
        j=i+1
        tool_target(motion,'left','iron',hot+[0,0,.031],iron_rot,1.4,f'align iron over land {j}','joint')
        tool_target(motion,'right','wire',feed+[0,0,.021],wire_rot,1.4,f'align solder over land {j}','joint')
        reserve=feed+np.array((-.003,0.,.0010))
        tool_target(motion,'right','wire',reserve,wire_rot,.70,f'bring solder alongside land {j}')
        settle_tip(motion,'right','wire',reserve,wire_rot,f'align solder tip to land {j}',6)
        tool_target(motion,'left','iron',hot+[0,0,.006],iron_rot,.75,f'approach land {j} closely')
        settle_tip(motion,'left','iron',hot+[0,0,.006],iron_rot,f'align tip to land {j}',5)
        settle_tip(motion,'left','iron',hot,iron_rot,f'contact and preheat land {j}',3)
        motion.hold(.36,f'preheat land {j}')
        if args.mode!='iron-only':
            settle_tip(motion,'right','wire',feed,wire_rot,f'feed solder onto heated land {j}',4)
            settle_tip(motion,'right','wire',feed,wire_rot,f'wet land {j} with lead-free solder',4)
            motion.hold(.23,f'hold solder against land {j}')
            tool_target(motion,'right','wire',feed+[0,0,.012],wire_rot,.23,f'withdraw solder from joint {j}')
        else:
            motion.hold(.98,f'heat-only negative control land {j}')
        tool_target(motion,'left','iron',hot+[0,0,.019],iron_rot,.37,f'withdraw iron from joint {j}')
        motion.hold(.50,f'cool solder joint {j}')
        if args.mode=='one-joint':
            break
    put_down(motion,'right','wire',wire_pick,wire_rotation)
    put_down(motion,'left','iron',iron_pick,iron_rotation)
    motion.hold(2.5,'cooled solder joints with both hands clear')


ACTION_LIMIT = 1000


class Trial:
    """Shared 20 Hz control path for planned motion, replay, and Dream-1."""

    def __init__(self, *, enabled=True, staged=True, reverse=False):
        self.model = mj.MjModel.from_xml_string(scene_xml(IRON_AXIS, WIRE_AXIS, reverse, staged))
        self.data = mj.MjData(self.model)
        self.material = Solder(self.model, self.data, enabled, staged)
        self.motion = Motion(self.model, self.data, self.capture)
        self.trace = []
        self.states = [self.data.qpos.copy()]
        self.fillets = [self.material.wetting.copy()]
        self.actions = [np.stack([HOME, HOME])]
        self.peak_effort = np.zeros((2, 6))
        self.peak_speed = np.zeros((2, 6))
        self.peak_jaw_speed = np.zeros(2)
        self.joint_dofs = [np.ravel([self.model.joint(f'{side}_joint{j}').dofadr for j in range(1, 7)])
                           for side in ARMS]
        self.finger_dofs = [np.ravel([self.model.joint(name).dofadr for name in arm.finger_names])
                            for arm in self.motion.arms.values()]
        self.substeps = round(.05 / self.model.opt.timestep)

    def capture(self, phase, index):
        if index > ACTION_LIMIT:
            raise ValueError('Action limit exceeded')
        for _ in range(self.substeps):
            mj.mj_step(self.model, self.data)
            self.material.update()
            for j, arm in enumerate(self.motion.arms.values()):
                self.peak_effort[j] = np.maximum(self.peak_effort[j], abs(self.data.actuator_force[arm.actuators]))
                self.peak_speed[j] = np.maximum(self.peak_speed[j], abs(self.data.qvel[self.joint_dofs[j]]))
                self.peak_jaw_speed[j] = max(self.peak_jaw_speed[j], np.max(abs(self.data.qvel[self.finger_dofs[j]])))
        row = self.material.sample()
        row.update(phase=phase, action=index,
                   bodies={name: self.data.body(name).xpos.tolist() for name in ('capacitor','iron','wire')},
                   tools={name: self.data.site(site).xpos.tolist() for name, site in POINTS.items()},
                   grippers={side: {'position': self.data.site_xpos[arm.site].tolist(),
                                    'gap': sum(float(self.data.joint(name).qpos[0]) for name in arm.finger_names)}
                             for side, arm in self.motion.arms.items()})
        self.trace.append(row)
        self.states.append(self.data.qpos.copy())
        self.fillets.append(self.material.wetting.copy())

    def step(self, action, phase='policy'):
        commands = np.asarray(action, dtype=float).reshape((2, 7))
        for arm, target in zip(self.motion.arms.values(), commands):
            arm.command(self.data, target)
        self.capture(phase, len(self.trace)+1)
        self.actions.append(commands.copy())

    def report(self):
        return dict(self.material.sample(), actions=len(self.trace), events=self.material.events,
                    peak_joint_effort_nm=self.peak_effort.tolist(),
                    peak_joint_speed_rad_s=self.peak_speed.tolist(), peak_finger_speed_m_s=self.peak_jaw_speed.tolist())

    def save(self, folder):
        with (folder/'trace.jsonl').open('w') as stream:
            for row in self.trace:
                stream.write(json.dumps(row)+'\n')
        np.savez_compressed(folder/'states.npz', qpos=self.states, wetted=self.fillets)


def run(args):
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    staged = args.staged or args.mode == 'solder-stage'
    trial = Trial(enabled=args.mode != 'cold', staged=staged, reverse=args.mode == 'reverse')
    motion = trial.motion
    commands = None
    error = None
    try:
        if args.replay:
            commands = np.load(args.replay, allow_pickle=False)
            if (commands.ndim != 3 or commands.shape[1:] != (2, 7) or len(commands)-1 > ACTION_LIMIT
                    or not np.all(np.isfinite(commands)) or not np.array_equal(commands[0], np.stack([HOME, HOME]))):
                raise ValueError('Expected a finite dual-arm command file beginning at the ready pose')
            for action in commands[1:]:
                trial.step(action, 'command replay')
        elif args.mode == 'idle':
            for _ in range(ACTION_LIMIT):
                trial.step(np.stack([HOME, HOME]), 'hands-off control')
        else:
            motion.hold(.75, 'settle')
            if args.mode != 'tools' and not staged:
                load_capacitor(motion, args)
            if args.mode != 'seat':
                solder_joints(motion, args)
    except (ValueError, RuntimeError) as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    report = dict(trial.report(), error=error, mode=args.mode)
    (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    trial.save(folder)
    actions = trial.actions if args.replay or args.mode == 'idle' else motion.replay
    np.save(folder/'actions.npy', np.asarray(actions))
    if args.record:
        from render import render
        render(folder, trial)
    print(f'Grade {report["score"]:.2f}', flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--record',action='store_true')
    p.add_argument('--search',action='store_true')
    p.add_argument('--staged',action='store_true')
    p.add_argument('--replay',type=Path)
    p.add_argument('--mode',choices=('full','idle','seat','tools','cold','iron-only',
                                     'one-joint','no-tools','reverse','solder-stage'),default='solder-stage')
    for name,value in (('cap-x',0.),('cap-y',0.),('cap-z',0.),('seat-x',0.),
                       ('seat-y',0.),('seat-z',.00012),('iron-z',0.),('wire-z',0.),
                       ('iron-depth',.00002),('wire-depth',.0025),('feed-x',.0002),('feed-y',.00033)):
        p.add_argument('--'+name,type=float,default=value)
    run(p.parse_args())
