"""Mechanical model of the user-replaceable DualSense Edge stick assembly."""
from pathlib import Path
import os
import sys
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
try:
    from dream_examples.yam_mujoco import yam_env
except ModuleNotFoundError:
    example = os.environ.get("ROBOCURVE_DREAM_YAM")
    if example is None:
        raise
    sys.path.insert(0, example)
    import yam_env

TABLE = .760
ORIGIN = np.array((.022, -.103, TABLE + .054))
MOUNT = np.array((-.026, .013, .026))
BLACK = (.105,.115,.128,1.)
INNER = (.18,.195,.21,1.)
WHITE = (.91,.92,.94,1.)
RUBBER = (.17,.18,.20,1.)
METAL = (.64,.69,.74,1.)
PALE = (.63,.66,.7,1.)
COOL = (.42,.49,.57,1.)
FRICTION = {"friction": ".75 .004 .0004"}
# Closed cover and rails use native collision where parts approach the housing.
PROFILE = {"solref": ".006 1", "solimp": ".95 .995 .001"}
RAIL_PROFILE = {**PROFILE, "priority": "1", "friction": ".27 .002 .0002"}

def nums(p):
    return " ".join(f"{float(x):.10g}" for x in p)

def body(parent, name, pos, **attrs):
    return ET.SubElement(parent,"body",name=name,pos=nums(pos),**attrs)

def geom(parent,name,kind,size,pos,color,**attrs):
    return ET.SubElement(parent,"geom",type=kind,name=name,size=nums(size),
                         pos=nums(pos),rgba=nums(color),**attrs)

def box(parent,name,dims,pos,color,**attrs):
    return geom(parent,name,"box",np.array(dims)/2,pos,color,**attrs)

def site(parent,name,pos):
    return ET.SubElement(parent,"site",name=name,pos=nums(pos),rgba="0 0 0 0",size=".001")

def ellipsoid(parent,name,radii,pos,color,**attrs):
    return geom(parent,name,"ellipsoid",radii,pos,color,**attrs)

def along_guide(point, rotation):
    delta=point-MOUNT
    return (abs(delta[0])<.004 and abs(delta[2])<.004 and
            .018<delta[1]<.034 and np.linalg.norm(rotation-np.eye(3))<.25)

def ring(parent,name,center,inside,outside):
    for i in range(16):
        a=i*np.pi/8
        rr=(inside+outside)/2
        p=np.asarray(center)+[rr*np.cos(a),rr*np.sin(a),0]
        box(parent,f"{name}_{i}",(outside-inside,.0082,.003),p,BLACK,
            euler=nums((0,0,a)),mass=".00032",**PROFILE)

def add_module(world,name,p,seat=False,mark=RUBBER,quat=None):
    opts={} if quat is None else {"quat":nums(quat)}
    mod=body(world,name,p,**opts)
    if not seat:
        ET.SubElement(mod,"freejoint",name=name+"_free")
    box(mod,name+"_case",(.027,.038,.016),(0,0,0),INNER,
        mass=".009",**FRICTION)
    for sign in (-1,1):
        box(mod,f"{name}_tongue_{sign}",(.0026,.035,.0035),
            (sign*.0141,0,-.0032),BLACK,mass=".0007",**PROFILE)
    geom(mod,name+"_post","cylinder",(.0045,.005),(0,.001,.011),BLACK,mass=".0009",
         **FRICTION)
    ellipsoid(mod,name+"_stick",(.00915,.00915,.0048),(0,.001,.0187),mark,
              mass=".0036",**FRICTION)
    box(mod,name+"_fn_tab",(.010,.004,.004),(0,.020,.0015),BLACK,
        mass=".0005")
    site(mod,name+"_grip",(0,.001,.0187))
    return mod

def build(mode="full"):
    root=ET.fromstring(yam_env.build_xml())
    root.set("model","Replace a gamepad stick module")
    root.find("option").attrib.update(timestep=".001",iterations="60",cone="elliptic")
    root.find("visual/quality").set("shadowsize","2048")
    world=root.find("worldbody")
    for lamp in world.findall("light"):
        lamp.set("castshadow","false")
    for n in ("apple","banana","orange","water_bottle","bowl","plate"):
        world.remove(world.find(f"body[@name='{n}']"))
    world.remove(world.find("camera[@name='top']"))
    yam_env.calibrated_camera(world,"top",yam_env.CALIBRATION["cameras"]["top"],world=True)
    # Same angle as the candy close-up, framed around the larger working object.
    yam_env.camera(world,"detail",(-.46,-.43,1.183),(.018,-.085,.8115))
    world.find("camera[@name='detail']").set("fovy","35")
    controller_rot=np.array([1.,0.,0.,0.]) if mode in ("exchange","lever") else np.array([0.,0.,1.,0.])
    position=ORIGIN.copy()
    if mode in ("exchange","lever"):position[2]=TABLE+.0005
    c=body(world,"controller",position,quat=nums(controller_rot))
    ET.SubElement(c,"freejoint",name="controller_free")
    box(c,"main_chassis",(.100,.069,.0176),(0,-.003,.0088),INNER,
        mass=".124",**FRICTION)
    for side in (-1,1):
        ellipsoid(c,f"controller_grip_{side}",(.0175,.0435,.0183),
                  (side*.0625,.008,.019),WHITE,euler=nums((0,0,side*.075)),
                  mass=".062",**FRICTION)
        ellipsoid(c,f"lower_grip_{side}",(.0165,.0235,.0088),
                  (side*.063,.029,.0098),BLACK,euler=nums((0,0,side*.15)),
                  mass=".009",**FRICTION)
        ellipsoid(c,f"shoulder_button_{side}",(.020,.011,.011),
                  (side*.052,-.033,.028),BLACK,mass=".006",**FRICTION)
        for i, (x,y) in enumerate(((0,.006),(.007,0),(0,-.006),(-.007,0))):
            geom(c,f"face_button_{side}_{i}","cylinder",(.0031,.0022),
                 (side*.052+x,-.019+y,.036),PALE if side<0 else BLACK,mass=".00017")
    box(c,"touch_pad",(.053,.023,.004),(0,-.022,.033),WHITE,mass=".009")
    box(c,"between_sticks",(.019,.044,.009),(0,.012,.0235),INNER,mass=".006")
    box(c,"rail_floor",(.033,.048,.007),(MOUNT[0],.007,.0146),INNER,
        mass=".008",**RAIL_PROFILE)
    for side in (-1,1):
        x=MOUNT[0]+side*.0176
        box(c,f"rail_web_{side}",(.0026,.029,.011),(x,.004,.0265),INNER,
            mass=".0012",**RAIL_PROFILE)
        box(c,f"rail_lip_{side}",(.0033,.029,.0028),(x-side*.0014,.004,.0295),
            COOL,mass=".0008",**RAIL_PROFILE)
    box(c,"socket_stop",(.027,.004,.013),(MOUNT[0],-.008,.026),BLACK,
        mass=".0018",**PROFILE)
    box(c,"port_trim",(.020,.003,.003),(0,.032,.021),METAL,mass=".0008")
    lift=body(c,"module_lever",(-.045,-.003,.0318))
    ET.SubElement(lift,"joint",name="lever_angle",type="hinge",axis="1 0 0",
                  range="0 1.4",damping=".008",armature=".0005",frictionloss=".006",
                  stiffness=".002",springref="0")
    box(lift,"lever_strip",(.0053,.041,.0023),(0,.018,0),METAL,mass=".0015",**PROFILE)
    box(lift,"lever_lock_toe",(.011,.0032,.0072),(.0036,.038,-.0035),METAL,
        mass=".0011",**PROFILE)
    box(lift,"lever_lip",(.0095,.0064,.0043),(-.0005,.026,.0019),METAL,mass=".0011",**FRICTION)
    site(lift,"lever_tip",(-.0005,.026,.0028))
    site(c,"lever_pivot",(-.045,-.003,.0318))
    site(c,"brace",(.060,.022,.019))
    latch=body(c,"rear_latch",(0,.028,.001))
    ET.SubElement(latch,"joint",name="latch_slide",type="slide",axis="-1 0 0",
                  range="0 .0048",stiffness="135",damping=".38",armature=".0001")
    box(latch,"release_slider",(.009,.0062,.003),(0,0,-.002),PALE,mass=".001",**FRICTION)
    site(latch,"latch_tip",(-.002,0,-.0034))
    add_module(c,"left_other",(-MOUNT[0],MOUNT[1],MOUNT[2]),seat=True)
    flip_axes=np.array([-1.,1.,-1.])
    add_module(world,"old_module",position+MOUNT*flip_axes if mode=="full" else position+MOUNT,
               mark=RUBBER,quat=controller_rot)
    add_module(world,"new_module",(-.098,-.074,TABLE+.011),mark=(.28,.30,.33,1))
    cover_p=position+(np.array([0.,.009,.036])*flip_axes if mode=="full" else [-.090,.094,.017])
    cover=body(world,"face_cover",cover_p,quat=nums(controller_rot if mode=="full" else (1,0,0,0)))
    ET.SubElement(cover,"freejoint",name="cover_free")
    for sign in (-1,1):
        ring(cover,f"cover_ring_{sign}",(sign*.026,.005,0),.0126,.0175)
        box(cover,f"cover_shoulder_{sign}",(.009,.032,.003),
            (sign*.0425,.007,0),BLACK,mass=".0016",**FRICTION)
    box(cover,"cover_bridge",(.024,.034,.003),(0,.005,0),BLACK,mass=".0031",**FRICTION)
    box(cover,"cover_front_lip",(.084,.008,.004),(0,.025,-.0004),
        BLACK,mass=".0038",**FRICTION)
    box(cover,"cover_front_keel",(.029,.004,.005),(0,.028,-.0029),BLACK,
        mass=".0014",**FRICTION)
    site(cover,"cover_grip",(0,.027,-.0006))
    for y in (.002,.009,.016):
        geom(cover,f"cover_control_{y:g}","cylinder",(.0021,.001),
             (0,y,.0022),INNER,mass=".0002")
    equality=root.find("equality")
    if equality is None:equality=ET.SubElement(root,"equality")
    ET.SubElement(equality,"weld",name="face_clip",body1="controller",body2="face_cover",
                  relpose="0 .009 .036 1 0 0 0",active="true" if mode=="full" else "false",
                  solref=".009 1",solimp=".97 .998 .0004")
    # Keep the bores and lever present while approximating the front cover's sprung metal clips.
    return ET.tostring(root,encoding="unicode")


class Mechanism:
    """Apply the small faceplate eject spring and measure the stock mechanical sequence."""
    def __init__(self, model, data, mode):
        self.m,self.d,self.mode=model,data,mode
        self.clip=model.equality("face_clip").id
        self.shown=mode!="full"
        self.opened=mode in ("exchange","lever")
        self.relatched=False
        self.lever_seen=False
        self.original_cleared_guide=False
        self.old_out=False
        self.replacement_at_guide=False
        self.new_in=False
        self.final_hold=0.
        self.bad=False
        self.controller=model.body("controller").id
        self.cover=model.body("face_cover").id
        # Assembly reset always uses the nominal stock assembled clip pose.
        model.eq_data[self.clip,3:10]=[0.,.009,.036,1.,0.,0.,0.]

    def rel(self,name,origin="controller"):
        body=self.d.body(origin)
        rot=body.xmat.reshape(3,3)
        target=self.d.body(name)
        return rot.T@(target.xpos-body.xpos),rot.T@target.xmat.reshape(3,3)

    def load(self, starts, other=None, side=None):
        total=0.
        force=np.zeros(6)
        for i, contact in enumerate(self.d.contact):
            if contact.geom1<0 or contact.geom2<0: continue
            names=tuple(self.m.geom(x).name for x in (contact.geom1,contact.geom2))
            for first, second in (names,names[::-1]):
                if not first.startswith(starts):continue
                if other is not None and not second.startswith(other):continue
                if side is not None and not (second.startswith(side+"_") and "_tip_" in second):continue
                mj.mj_contactForce(self.m,self.d,i,force)
                total+=max(0.,float(force[0]))
        return total

    def step(self):
        d,m=self.d,self.m
        d.qfrc_applied[:]=0.
        displacement,_=self.rel("face_cover")
        seated=displacement-np.array([0.,.009,.036])
        if not d.eq_active[self.clip] and self.opened and not self.shown:
            # Released stock clips lift the loose panel. The spring fades at 2.2 mm.
            travel=max(0.,seated[2])
            if travel<.0022 and np.linalg.norm(seated[:2])<.007:
                rot=d.body("controller").xmat.reshape(3,3)
                direction=rot[:,2]
                spring=direction*.58*(1.-travel/.0022)
                mj.mj_applyFT(m,d,spring,np.zeros(3),d.xpos[self.cover],self.cover,d.qfrc_applied)
                mj.mj_applyFT(m,d,-spring,np.zeros(3),d.xpos[self.cover],self.controller,d.qfrc_applied)

    def observe(self):
        d,m=self.d,self.m
        dial=d.joint("latch_slide").qpos[0]
        if (not self.opened and dial>=.0035 and
                sum(self.load("release_slider",side=side) for side in ("left","right"))>.08):
            self.opened=True
            d.eq_active[self.clip]=False
        cover,rotation=self.rel("face_cover")
        cover_delta=cover-[0.,.009,.036]
        if self.opened and np.linalg.norm(cover_delta)>.042:self.shown=True
        lever=d.joint("lever_angle").qpos[0]
        if self.shown and lever>1.03:self.lever_seen=True
        old,old_r=self.rel("old_module")
        new,new_r=self.rel("new_module")
        if self.lever_seen and along_guide(old,old_r):
            self.original_cleared_guide=True
        if self.original_cleared_guide and np.linalg.norm(old-MOUNT)>.064:
            self.old_out=True
        if self.old_out and along_guide(new,new_r):
            self.replacement_at_guide=True
        seated=(np.linalg.norm(new[:2]-MOUNT[:2])<.0022 and abs(new[2]-MOUNT[2])<.0017
                and np.linalg.norm(new_r-np.eye(3))<.14)
        self.new_in=bool(self.replacement_at_guide and seated)
        if (self.new_in and lever<.24 and not self.relatched
                and np.linalg.norm(cover_delta)<.0011 and np.linalg.norm(rotation-np.eye(3))<.07
                and sum(self.load("cover_",side=side) for side in ("left","right"))>.4
                and sum(self.load("cover_",other=case) for case in
                        ("left_other_case","new_module_case","between_sticks","port_trim"))>.25):
            # The approximation retains the pose reached by physical closure.
            # Enabling the clip never pulls the part to a preferred pose.
            quat=np.empty(4)
            mj.mju_mat2Quat(quat,rotation.ravel())
            m.eq_data[self.clip,3:6]=cover
            m.eq_data[self.clip,6:10]=quat
            self.relatched=True
            d.eq_active[self.clip]=True
        sites=[d.site(f"{side}_tcp").xpos for side in ("left","right")]
        clearance=min(np.linalg.norm(q-d.body("controller").xpos) for q in sites)
        old_on_table=TABLE+.006<d.body("old_module").xpos[2]<TABLE+.012
        separate=np.linalg.norm(old-MOUNT)>.064
        ready=(self.relatched and seated and lever<.18 and clearance>.09 and old_on_table and separate and
               d.body("controller").xmat.reshape(3,3)[2,2]>.99 and
               d.body("controller").xpos[2]<TABLE+.005)
        self.final_hold=self.final_hold+m.opt.timestep if ready else 0.
        if any(d.body(name).xpos[2]<TABLE-.005 for name in ("controller","face_cover","old_module","new_module")):
            self.bad=True

    def score(self):
        score=0.
        if self.shown and self.opened and self.mode=="full":score=.2
        if self.old_out:score=.45
        if self.new_in:score=.75
        if self.final_hold>=1.:score=1.
        if self.bad:score=0.
        return dict(score=score,cover_unlocked=self.opened,cover_removed=self.shown,lever_opened=self.lever_seen,
                    original_removed=self.old_out,replacement_seated=self.new_in,cover_relatched=self.relatched,
                    original_guide_travel=self.original_cleared_guide,replacement_guide_travel=self.replacement_at_guide,
                    final_hold_s=self.final_hold,violation=self.bad,
                    latch_mm=float(self.d.joint("latch_slide").qpos[0]*1000),
                    lever_deg=float(self.d.joint("lever_angle").qpos[0]*180./np.pi))
