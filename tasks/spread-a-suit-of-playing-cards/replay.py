"""Joint-action replay and measured terminal state for the suit-ribbon task."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from episode import ACTION_LIMIT, RibbonTrial
from scene import CAMERAS, COUNT, DECK_XY, FELT_Z, HALF_LENGTH, READY, reset


HZ = 20
SIDE = 'right'
DOWN = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])


def smooth(t):
    return t*t*(3.-2.*t)


def plan(model, sweep=.247, lift=.0048, height=.01035, pitch_deg=0, guide_y=0., spread_sec=2.8,
         leading_pause=0., profile=1., negative=False, two_tips=False, roll_deg=0., clearance=.013):
    scratch = reset(model)
    ids = [model.joint(f'{SIDE}_joint{i}').id for i in range(1,7)]
    addrs = model.jnt_qposadr[ids]
    lo,hi = model.jnt_range[ids].T
    tcp = model.site(f'{SIDE}_tcp').id
    base_attitude = np.diag([1.,-1.,-1.]) if two_tips else DOWN
    attitude = Rotation.from_euler('xy',[roll_deg,pitch_deg],degrees=True).as_matrix() @ base_attitude
    # Register the stock TCP against the narrowest reach of the trailing inner pad.
    pad_offset = attitude @ np.array([.006, 0., .0102] if two_tips else [0., -.0475, .0102])
    start = np.array([DECK_XY[0]-HALF_LENGTH-clearance-pad_offset[0], DECK_XY[1]+guide_y, FELT_Z+height])
    if negative:
        start[2]+=.012

    def solve(point, seed):
        def residual(q):
            scratch.qpos[addrs] = q
            mj.mj_kinematics(model,scratch)
            rot = scratch.site_xmat[tcp].reshape(3,3)
            return np.r_[(scratch.site_xpos[tcp]-point)*90,
                         Rotation.from_matrix(attitude.T@rot).as_rotvec(), .0003*(q-seed)]
        result=least_squares(residual,np.clip(seed,lo+1e-9,hi-1e-9),bounds=(lo,hi),
                             max_nfev=100,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        return result.x, float(np.linalg.norm(result.fun[:3])/90)

    # Every command is determined from fixed reset coordinates and CAD, before the physical replay.
    coords = []
    for i in range(1,61):
        approach = smooth(i/60.)
        coords.append(('approach', start+[0,0,.075*(1.-approach)]))
    coords += [('settle',start.copy())]*18
    path_steps=max(24,round(spread_sec*HZ))
    for i in range(1,path_steps+1):
        t=i/path_steps
        x=smooth(t)
        up=max(0., (x-leading_pause)/(1.-leading_pause)) ** profile
        coords.append(('spread',start+[sweep*x,0,lift*up]))
    for i in range(1,15):
        coords.append(('release',start+[sweep,0,lift+.084*smooth(i/14.)]))
    points=[]
    q=READY
    largest=0.
    for stage, point in coords:
        q,error=solve(point,q)
        largest = max(largest,error)
        points.append((stage,q.copy()))
    if largest>.001:
        raise RuntimeError(f'IK beyond 1 mm: {largest*1000:.3f} mm')
    first = points[0][1]
    lead = [('lead-in', READY+(first-READY)*smooth(i/29.)) for i in range(30)]
    hold = [('hold',points[-1][1].copy())]*43
    return [*lead,*points,*hold], largest


def state(model,data):
    xyz = np.array([data.body(f'card_{i:02d}').xpos.copy() for i in range(COUNT)])
    directions = np.array([data.body(f'card_{i:02d}').xmat.reshape(3,3)[:,2] for i in range(COUNT)])
    angles = np.array([data.body(f'card_{i:02d}').xmat.reshape(3,3)[:,0] for i in range(COUNT)])
    gaps = np.diff(xyz[:,0])*1000
    return {'time_s': round(data.time,6), 'xyz_mm': (xyz*1000).round(3).tolist(),
            'centers_gaps_mm': gaps.round(3).tolist(), 'min_gap_mm': float(min(gaps)),
            'ordered_gaps_ge9mm': int(((gaps>=9)&(gaps<55)).sum()),
            'positive_gaps': int((gaps>0).sum()), 'orient_x': angles.round(5).tolist(),
            'up_dot': directions[:,2].round(5).tolist(),
            'tcp_mm': (data.site(f'{SIDE}_tcp').xpos*1000).round(3).tolist()}


def run(options):
    trial=RibbonTrial(art=bool(options.video))
    model, data=trial.model, trial.data
    if options.idle:
        steps,ik=[('idle', READY.copy())] * ACTION_LIMIT, 0.
    else:
        steps, ik=plan(model, options.stroke, options.rise, options.height, options.pitch,
                      options.guide_y, options.duration, options.delay, options.profile,
                      options.negative,options.two_tips,options.roll,options.clearance)
    opening = options.pad_open/.0475 if options.two_tips and not options.idle else 1.
    contact_names = {model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.startswith(f'{SIDE}_tip_')}
    trace=[]
    actions=[]
    max_push=0.
    before_sweep_motion=0.
    wrench=np.zeros(6)
    with ExitStack() as context:
        if options.video:
            from review_media import StationViews, review_writer, save_setup
            dest=Path(options.video)
            dest.parent.mkdir(exist_ok=True,parents=True)
            views=StationViews(model,CAMERAS)
            context.callback(views.close)
            frame=context.enter_context(review_writer(dest))
            save_setup(views.capture(data),dest.with_name('setup.png'))
        for stage, q in steps:
            action=np.r_[READY,1.,q,opening]
            trial.step(action)
            actions.append(action)
            if stage in ('lead-in','approach','settle','idle'):
                positions=np.array([data.body(f'card_{i:02d}').xpos[:2] for i in range(COUNT)])
                before_sweep_motion=max(before_sweep_motion,float(abs(positions-DECK_XY).max()))
            for ci in range(data.ncon):
                c=data.contact[ci]
                a,b=(model.geom(int(g)).name for g in c.geom)
                if (a in contact_names and b.startswith('card_')) or (b in contact_names and a.startswith('card_')):
                    mj.mj_contactForce(model,data,ci,wrench)
                    max_push=max(max_push,float(wrench[0]))
            if options.trace:
                trace.append({'stage':stage,**state(model,data)})
            if options.video:
                frame(views.capture(data))
        final=state(model,data)
        if options.video:
            save_setup(views.capture(data),dest.with_name('finish.png'))
            from PIL import Image

            with mj.Renderer(model,height=1080,width=1600) as photo:
                photo.update_scene(data,camera=CAMERAS['judging_front'])
                Image.fromarray(photo.render()).save(dest.with_name('finish-overhead.png'))
    final.update(action_count=len(steps), control_hz=HZ, maximum_plan_error_mm=ik*1000,
                 maximum_frame_sampled_pad_normal_force_n=max_push, solver_dt=model.opt.timestep,
                 max_before_sweep_movement_mm=round(before_sweep_motion*1000,4),**trial.summary())
    if options.trace:
        final['trace']=trace
    if options.actions:
        np.save(options.actions,np.array(actions))
    return final


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stroke',type=float,default=.247)
    parser.add_argument('--rise',type=float,default=.0048)
    parser.add_argument('--height',type=float,default=.01035)
    parser.add_argument('--pitch',type=float,default=0.)
    parser.add_argument('--guide-y',type=float,default=0.)
    parser.add_argument('--duration',type=float,default=2.8)
    parser.add_argument('--delay',type=float,default=0.)
    parser.add_argument('--profile',type=float,default=1.)
    parser.add_argument('--negative',action='store_true')
    parser.add_argument('--idle',action='store_true')
    parser.add_argument('--two-tips',action='store_true')
    parser.add_argument('--roll',type=float,default=.46)
    parser.add_argument('--pad-open',type=float,default=.018)
    parser.add_argument('--clearance',type=float,default=.013)
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--actions',type=Path)
    return parser.parse_args()


if __name__=='__main__':
    cli=arguments()
    result=run(cli)
    text=json.dumps(result,indent=2,default=np.generic.item)+'\n'
    if cli.out:
        cli.out.write_text(text)
    print(json.dumps({k:v for k,v in result.items() if k!='trace'},default=np.generic.item))
