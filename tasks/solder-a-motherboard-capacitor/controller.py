"""Local calibration-frame joint controls for the rework bench."""
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from yam_motion import ARMS, Arm, HOME, direction

CONTROL_HZ = 20
CLOSED = .003
OPEN = .35


def tool_axis(yaw, pitch):
    yaw, pitch = np.radians((yaw,pitch))
    return np.array((np.sin(yaw)*np.cos(pitch),
                     -np.cos(yaw)*np.cos(pitch),-np.sin(pitch)))


def horizontal_pick(axis, flip=False):
    x = np.asarray(axis)
    y = np.cross((0.,0.,-1.),x)
    return np.column_stack((x,y,(0.,0.,-1.))) @ np.diag(
        (-1.,-1.,1.) if flip else (1.,1.,1.))


def turn_tool(h, start, end):
    start, end = np.asarray(start), np.asarray(end)
    hinge = np.cross(start,end)
    sine = np.linalg.norm(hinge)
    if sine < 1e-9:
        return h
    return Rotation.from_rotvec(hinge/sine*np.arctan2(sine,start@end)).as_matrix()@h


class Motion:
    def __init__(self, model, data, capture):
        self.model, self.data = model,data
        self.arms = {side:Arm(model,side) for side in ARMS}
        for arm in self.arms.values():
            arm.reset(data)
        mj.mj_forward(model,data)
        self.planning = mj.MjData(model)
        self.commands = {side:HOME.copy() for side in ARMS}
        self.replay = [tuple(HOME for _ in ARMS)]
        self.capture = capture
        self.phase = 'settle'

    def frame(self,side,start_pose=None):
        if start_pose is not None:
            return start_pose
        site = self.arms[side].site
        return self.data.site_xpos[site].copy(), self.data.site_xmat[site].reshape(3,3).copy()

    def hold(self,seconds, label='hold'):
        self.move(seconds,label)

    def move(self,seconds,label,**goals):
        self.phase = label
        previous = {side: self.commands[side].copy() for side in ARMS}
        count = max(1, round(CONTROL_HZ*seconds))
        plans = {side: self.plan(side,goal,count,previous[side])
                 for side,goal in goals.items()}
        for index in range(count):
            for side,plan in plans.items():
                self.commands[side]=plan[index]
            for side,arm in self.arms.items():
                arm.command(self.data,self.commands[side])
            self.replay.append(tuple(self.commands[side].copy() for side in ARMS))
            self.capture(self.phase,len(self.replay)-1)
        print(f'{self.data.time:5.1f}s {label}',flush=True)

    def plan(self,side,goal,count,previous):
        arm = self.arms[side]
        if isinstance(goal,np.ndarray):
            fraction=np.linspace(0,1,count+1)[1:,None]
            return previous+(goal-previous)*fraction
        point, rot, jaw, strategy = goal
        self.planning.qpos[:] = self.data.qpos
        self.planning.qpos[arm.address]=previous[:6]
        mj.mj_kinematics(self.model,self.planning)
        if strategy == 'joint':
            sequence = []
            seed=previous[:6].copy()
            last_error = None
            for guess in (seed,
                          np.array([0.,1.,1.,-.6,0.,0.]),
                          np.array([.2,1.5,1.4,-.35,0.,0.]),
                          np.array([-.6,1.5,1.3,-.35,0.,0.])):
                try:
                    q=arm.inverse(self.planning,np.array(point),rot,guess)
                    sequence.append(q)
                except ValueError as error:
                    last_error=error
            if not sequence:
                raise last_error
            q=min(sequence,key=lambda a:np.linalg.norm(a-seed))
            fraction = np.linspace(0,1,count+1)[1:,None]
            fraction = fraction**2*(3-2*fraction)
            return previous+(np.r_[q,jaw]-previous)*fraction
        at = self.planning.site_xpos[arm.site].copy()
        orient = self.planning.site_xmat[arm.site].reshape(3,3).copy()
        points=np.linspace(0,1,count+1)[1:]
        slerp=Slerp([0,1],Rotation.from_matrix(np.stack((orient,rot))))
        count_of_knots = max(1,int(np.ceil(np.linalg.norm(point-at)/.004)))
        knots = np.linspace(0,1,count_of_knots+1)
        rotations=slerp(knots).as_matrix()
        seed=previous[:6].copy()
        targets=[seed]
        for path_s,rotation in zip(knots[1:],rotations[1:]):
            seed=arm.inverse(self.planning,at+(point-at)*path_s,rotation,seed)
            targets.append(seed)
        commands=np.column_stack([np.interp(points,knots,column) for column in np.array(targets).T])
        return np.column_stack((commands,previous[6]+(jaw-previous[6])*points))

    def jaw(self, side, value, seconds, label):
        target = self.commands[side].copy()
        target[6] = value
        self.move(seconds,label,**{side:target})

    def cart(self, side, point, rot, wrist, seconds, label, strategy='cart'):
        self.move(seconds,label,**{side:(np.asarray(point),rot,wrist,strategy)})


def workspace_pair(motion, targets, tools):
    """Screen solver poses for joint-accessible native collision-free endpoints."""
    from itertools import product

    model,data=motion.model,motion.data
    scratch=mj.MjData(model)
    poses={}
    ranges={'left':(-85,-72,-56,-42,-25,0), 'right':(85,72,56,42,25,0)}
    body_ids={side:{i for i in range(model.nbody) if model.body(i).name.startswith(side+'_')}
              for side in ARMS}
    all_robot_ids=body_ids['left']|body_ids['right']
    owners=model.geom_bodyid

    def blocked(check_arms=True):
        for c in scratch.contact:
            if c.dist>=-.0001:
                continue
            i,j=(owners[int(g)] for g in c.geom)
            both=i in all_robot_ids and j in all_robot_ids
            across=i in body_ids['left'] and j in body_ids['right'] or i in body_ids['right'] and j in body_ids['left']
            if across and check_arms:
                return True
            if not both and ((i in all_robot_ids or j in all_robot_ids)
                             and model.body(int(i if j in all_robot_ids else j)).name not in tools.values()):
                # Table and other task props are non-nominal obstructions to
                # the arm links, including in the approach envelope.
                return True
        return False

    for side in ARMS:
        arm=motion.arms[side]
        rest_q=data.qpos[arm.address].copy()
        live,rot=motion.frame(side)
        name=tools[side]
        tool=data.body(name)
        axis=tool.xmat.reshape(3,3)[:,2].copy()
        tip=data.site('iron_point' if name=='iron' else 'solder_point').xpos.copy()
        offset=rot.T@(tip-live)
        poses[side]=[]
        for yaw,pitch,twist in product(ranges[side],(15,25,38,52,65),(0,-20,20,-42,42,-65,65)):
            approach=tool_axis(yaw,pitch)
            rn=Rotation.from_rotvec(approach*np.deg2rad(twist)).as_matrix()@turn_tool(rot,axis,approach)
            xyz=targets[side]-rn@offset
            try:
                seed=motion.commands[side][:6]
                q=arm.inverse(scratch,xyz,rn,seed)
                h=arm.inverse(scratch,xyz+(0,0,.022),rn,q)
            except ValueError:
                continue
            scratch.qpos[:]=data.qpos
            scratch.qpos[arm.address]=q
            other=motion.arms['right' if side=='left' else 'left']
            scratch.qpos[other.address]=HOME[:6]
            mj.mj_forward(model,scratch)
            if blocked(check_arms=False):
                continue
            quality=(np.linalg.norm(q-rest_q)+.65*np.linalg.norm(h-q))
            poses[side].append((quality,q,h,rn,(yaw,pitch,twist)))
        poses[side].sort(key=lambda item:item[0])
        print('Accessible tool frames',side,len(poses[side]),flush=True)
    for left,right in sorted(product(poses['left'],poses['right']),key=lambda ij:ij[0][0]+ij[1][0]):
        scratch.qpos[:]=data.qpos
        good=True
        for l,r in ((1,1),(1,2),(2,1),(2,2)):
            scratch.qpos[motion.arms['left'].address]=left[l]
            scratch.qpos[motion.arms['right'].address]=right[r]
            mj.mj_forward(model,scratch)
            if blocked():
                good=False
                break
        if good:
            print('Selected bilateral joint approach',left[-1],right[-1],flush=True)
            return left[3],right[3]
    raise ValueError('No simultaneous collision-free tool frames in the sampled approaches')
