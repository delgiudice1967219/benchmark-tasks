"""Full-size wire lap-splice contact model on the calibrated dual-YAM station."""
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

if station_source := os.environ.get("ROBOCURVE_DREAM_YAM"):
    yam_env = SimpleNamespace(**runpy.run_path(str(Path(station_source) / "yam_env.py")))
else:
    from dream_examples.yam_mujoco import yam_env

TABLE = .760
CAMERAS = {name: name for name in ("top_cam", "left_cam", "right_cam", "judging_front")}
HINGE = np.array((.00035, -.166, .851))
JOINT = np.array((0., -.102, .851))
COPPER_RADIUS = .00051
CLIP_READY = -.110
ROOM_C = 23.
TIP_SET_C = 345.
LIQUIDUS_C = 227.
NEEDED_MM = 1.45
STEEL = (.46, .50, .54, 1.)
COLD_SOLDER = (.86, .89, .91, 1.)
BLACK = (.12, .15, .18, 1.)
RED = (.78, .12, .1, 1.)
COPPER = (.76, .51, .30, 1.)
IRONS = {
    "iron": {"origin": np.array((-.191, -.025, .815)),
             "yaw": 0., "pitch": .18, "tip": .112},
    "solder": {"origin": np.array((.162, -.020, .809)),
               "yaw": np.pi, "pitch": .12, "tip": .067},
}


def nums(values):
    return yam_env.numbers(values)


def shape(parent, name, kind, size, pos=(0, 0, 0), color=BLACK, **attrs):
    return ET.SubElement(parent, "geom", name=name, type=kind, size=nums(size),
                         pos=nums(pos), rgba=nums(color), **attrs)


def box(parent, name, size, pos=(0, 0, 0), color=BLACK, **attrs):
    return shape(parent, name, "box", np.asarray(size)/2, pos, color, **attrs)


def line(parent, name, a, b, radius, color, **attrs):
    return ET.SubElement(parent, "geom", name=name, type="capsule", size=str(radius),
                         fromto=nums(np.r_[a, b]), rgba=nums(color), **attrs)


def child(parent, name, pos, **attrs):
    return ET.SubElement(parent, "body", name=name, pos=nums(pos), **attrs)


def site(parent, name, pos):
    return ET.SubElement(parent, "site", name=name, pos=nums(pos), size=".0008", rgba="0 0 0 0")


def quat(rotation):
    x, y, z, w = rotation.as_quat()
    return nums((w, x, y, z))


def iron_rotation(key):
    spec = IRONS[key]
    return Rotation.from_euler("zy", [spec["yaw"], spec["pitch"]])


def add_rest(world, key, origin, rotation):
    """Heat-resistant fork cradle and open guard for a removable hand tool."""
    rest = child(world, key + "_stand", origin, quat=quat(rotation))
    base = origin.copy()
    base[2] = TABLE + .004
    box(world, key + "_stand_base", (.123, .056, .008), base, (.18,.21,.25,1))
    for index, x in enumerate((-.044,.026) if key == "iron" else (-.031,.031)):
        stem = origin+rotation.apply((x,0.,-.021))
        depth = stem[2] - TABLE
        box(world, f"{key}_stand_leg_{index}", (.009, .038, depth),
            (*stem[:2], TABLE + depth/2), (.15,.17,.18,1))
        for sign in (-1,1):
            box(rest, f"{key}_rest_fork_{index}_{sign}", (.009,.0055,.0215),
                (x,sign*.0117,-.012), STEEL)
    if key == "iron":
        for sign in (-1,1):
            box(rest, f"iron_guard_{sign}", (.057,.0025,.024), (.088,sign*.027,-.001), STEEL)
        box(rest, "iron_guard_floor", (.057,.057,.003), (.088,0.,-.015), STEEL)


def add_tool(world, assets, key):
    spec = IRONS[key]
    origin = spec["origin"]
    rotation = iron_rotation(key)
    add_rest(world,key,origin,rotation)
    tool = child(world,key,origin,quat=quat(rotation))
    ET.SubElement(tool,"freejoint",name=f"{key}_free")
    if key=="iron":
        shape(tool,"iron_barrel","capsule",(.0108,.041),(0,0,0),(.17,.26,.45,1),
              mass=".036",friction="1.2 .014 .001",quat="0.70710678 0 0.70710678 0")
    else:
        angle = np.arange(8)*np.pi/4+np.pi/8
        vertices=[(x,r*np.cos(t),r*np.sin(t)) for x,r in
                  ((-.039,.008),(-.035,.0109),(.032,.0109),(.037,.008)) for t in angle]
        ET.SubElement(assets,"mesh",name="faceted_solder_pen",vertex=nums(np.ravel(vertices)))
        ET.SubElement(tool,"geom",name="solder_barrel",type="mesh",mesh="faceted_solder_pen",
                      rgba=".59 .24 .15 1",mass=".030",friction="1.25 .012 .001")
    box(tool,f"{key}_grip_top",(.047,.0185,.0135),(0,0,0),
        (.18,.26,.46,1) if key=="iron" else (.64,.25,.16,1),
        mass="0",contype="0",conaffinity="0")
    site(tool,key+"_grip",(0,0,0))
    if key == "iron":
        line(tool,"iron_shaft",(.040,0,0),(.087,0,0),.0048,STEEL,mass=".006")
        line(tool,"iron_chisel",(.087,0,0),(.109,0,0),.0028,COLD_SOLDER,mass=".003")
        shape(tool,"iron_tip","sphere",(.00105,),(.112,0,0),COLD_SOLDER,
              mass=".001",friction=".22 .006 .001",solref=".010 1")
        site(tool,"iron_tip_point",(.112,0,0))
    else:
        line(tool,"solder_cap",(.029,0,0),(.043,0,0),.0066,STEEL,mass=".003")
        feed = child(tool,"solder_stock",(.053,0,0))
        ET.SubElement(feed,"joint",name="solder_feed",type="slide",axis="-1 0 0",
                      damping=".31",range="0 .004",armature=".00034")
        line(feed,"solder_lead",(-.010,0,0),(.0136,0,0),.0004,COLD_SOLDER,
             mass=".00029",solref=".008 1")
        shape(feed,"solder_tip","sphere",(.0004,),(.014,0,0),COLD_SOLDER,mass=".00001",
              solref=".006 1",friction=".28 .006 .001")
        site(feed,"solder_tip_point",(.014,0,0))


def make_fixture(root, world):
    """A spring-clip pair with one friction-held swivelling support."""
    box(world,"steel_holder_foot",(.051,.138,.016),(0,-.108,TABLE+.008),(.22,.25,.31,1))
    stationary = child(world,"fixed_lead",(-.0005,-.063,.851))
    box(stationary,"fixed_post",(.012,.012,.075),(0,.012,-.036),STEEL)
    box(stationary,"fixed_clip",(.012,.021,.009),(0,.004,-.002),BLACK)
    box(stationary,"fixed_clip_top",(.009,.021,.003),(0,.004,.005),STEEL)
    lead_free = child(world,"extension_clip",HINGE)
    ET.SubElement(lead_free,"inertial",pos="0 .028 -.012",mass=".045",diaginertia=".00004 .000042 .000035")
    ET.SubElement(lead_free,"joint",name="alignment_joint",type="hinge",axis="0 0 1",
                  range="-.29 .012",damping=".074",frictionloss=".055",springref=str(CLIP_READY),
                  stiffness="0.04",ref=str(CLIP_READY))
    # Reference yaw is off-axis. The robot turns the clip's insulated lever toward the fixed lead.
    lead_free.set("euler",nums((0,0,CLIP_READY)))
    box(world,"pivot_post",(.017,.013,.075),(.001,-.166,.810),STEEL)
    line(lead_free,"swivel_arm",(0,0,-.010),(0,.030,-.010),.0054,STEEL,
         contype="0",conaffinity="0")
    box(lead_free,"movable_clip",(.012,.025,.009),(0,.038,-.002),BLACK)
    box(lead_free,"movable_clip_top",(.009,.025,.003),(0,.037,.005),STEEL)
    box(lead_free,"swivel_thumb",(.021,.017,.014),(.013,.026,.008),(.22,.36,.51,1),
        friction="1.0 .015 .002")
    site(lead_free,"thumb_grip",(.013,.026,.008))
    line(stationary,"motor_insulation",(0,-.002,0),(0,-.0330,0),.0011,RED)
    line(stationary,"copper_fixed",(0,-.0330,0),(0,-.0397,0),COPPER_RADIUS,COPPER,
         solref=".010 1",solimp=".985 .998 .00005",friction=".35 .003 .001")
    site(stationary,"fixed_center",(0,-.03635,0))
    line(lead_free,"extension_insulation",(0,.031,0),(0,.0604,0),.0011,(.9,.48,.11,1))
    line(lead_free,"copper_free",(0,.0604,0),(0,.0674,0),COPPER_RADIUS,COPPER,
         solref=".010 1",solimp=".985 .998 .00005",friction=".35 .003 .001")
    site(lead_free,"free_center",(0,.0639,0))
    shape(stationary,"joint_fillet","ellipsoid",(.00105,.00105,.0024),(.00049,-.03765,.00013),
          (.68,.72,.77,0),contype="0",conaffinity="0",euler="1.57079633 0 0")
    # Affixed PVC jacket uses measured endpoints, while the free tail is a visual drape.
    for parent, name, points, color in (
        (stationary,"motor_drape",[(0,0,0),(-.003,.020,-.014),(-.024,.073,-.060),(-.066,.127,-.064),
                                  (-.083,.171,-.063)],RED),
        (lead_free,"extension_drape",[(0,.037,0),(0,.014,-.008),(.010,.002,-.032),(.025,-.013,-.071),
                                     (.035,-.058,-.088),(.025,-.089,-.088),(.002,-.102,-.087)],(.9,.48,.11,1.)),
    ):
        for idx,(a,b) in enumerate(zip(points,points[1:])):
            line(parent,f"{name}_{idx}",a,b,.0011,color,contype="0",conaffinity="0")
    motor = child(world,"drive_motor",(-.082,.079,TABLE+.023))
    shape(motor,"motor_can","cylinder",(.0185,.027),(0,0,0),STEEL,
          quat="0.70710678 0.70710678 0 0")
    shape(motor,"gearhead","cylinder",(.0185,.010),(0,-.030,0),(.27,.31,.36,1),
          quat="0.70710678 0.70710678 0 0")
    shape(motor,"motor_shaft","cylinder",(.003,.010),(0,-.046,0),COLD_SOLDER,
          quat="0.70710678 0.70710678 0 0")
    box(motor,"insulated_mount",(.039,.032,.006),(0,0,-.02),(.15,.37,.48,1))
    lead_a = (-.079,.106,.786)
    lead_b = (-.052,.121,.769)
    line(world,"unpowered_return_tail",lead_a,lead_b,.00105,(.13,.13,.14,1))
    box(world,"unpowered_return_plug",(.012,.015,.008),(-.049,.127,.768),(.14,.15,.17,1))


def build_xml():
    root = ET.fromstring(yam_env.build_xml())
    root.set("model","Solder a robot motor lead")
    root.find("option").attrib.update(timestep=".001",iterations="56",cone="elliptic")
    world = root.find("worldbody")
    assets = root.find("asset")
    for name in ("apple","banana","orange","water_bottle","bowl","plate"):
        world.remove(world.find(f"body[@name='{name}']"))
    for asset in list(assets):
        if asset.get("name", "").startswith(("apple_", "banana_", "bowl_", "brown_glaze", "blue_glaze")):
            assets.remove(asset)
    for lamp in world.findall("light"):
        lamp.set("castshadow", "false")
    world.remove(world.find("camera[@name='top']"))
    top = yam_env.calibrated_camera(world, "top_cam", yam_env.CALIBRATION["cameras"]["top"], world=True)
    for side in ("left", "right"):
        world.find(f".//camera[@name='{side}']").set("name", f"{side}_cam")
    top_pos = np.fromstring(top.get("pos"), sep=" ")
    stand = world.find("body[@name='center_camera_stand']")
    foot, mast, beam = stand.findall("geom")
    mast_y = top_pos[1] + .07
    foot.set("pos", nums((top_pos[0], mast_y, TABLE + .008)))
    mast.set("fromto", nums((top_pos[0], mast_y, TABLE + .016, top_pos[0], mast_y, top_pos[2])))
    beam.set("fromto", nums((top_pos[0], mast_y, top_pos[2], *(top_pos + (0., .02, .02)))))
    stand.find("body").attrib.update(pos=top.get("pos"), quat=top.get("quat"))
    center = np.array((-.010, -.061, .829))
    eye_offset = np.asarray((-.527, -.374, .373))
    yam_env.camera(world, "judging_front", center + eye_offset, center)
    fovy = np.degrees(2. * np.arctan(20.955 * 480 / (2. * 34. * 640)))
    world.find("camera[@name='judging_front']").set("fovy", str(fovy))
    yam_env.camera(world, "joint_inspection", (.018, -.261, .967), JOINT)
    world.find("camera[@name='joint_inspection']").set("fovy", "12")
    for name in ("iron","solder"):
        add_tool(world,assets,name)
    make_fixture(root,world)
    return ET.tostring(root,encoding="unicode")


class Splice:
    """Thermal and cohesive response driven by native contact, feed and displacement."""

    def __init__(self,model,data,mode):
        self.model,self.data,self.mode = model,data,mode
        self.geoms={name:model.geom(name).id for name in
                    ("copper_fixed","copper_free","iron_tip","iron_chisel","iron_shaft","solder_tip","solder_lead",
                     "motor_insulation","extension_insulation","joint_fillet")}
        self.clip=model.joint("alignment_joint")
        self.feed=model.joint("solder_feed")
        self.temperatures=np.full(2,ROOM_C)
        self.force=np.zeros(6)
        self.stable_alignment=0.
        self.alignment_hold=0.
        self.joint_kept=False
        self.overlap_m=0.
        self.fed_m=0.
        self.freeze=None
        self.latched=False
        self.violations=set()
        self.contact_loads=np.zeros(4)
        self.peak_loads=np.zeros(4)
        self.peak_temperature=ROOM_C
        self.peak_iron_grip=0.
        self.peak_solder_grip=0.
        self.peak_bond_nm=0.
        self.alignment_q=None
        self.hot_contact_insulation_s=0.
        self.coarse_contact_seconds=0.
        self.recent_force=np.zeros(4)
        self.broken_joint=False
        self.safe_rest_time=0.
        self.events=[]
        self.tool_parts = {}
        self.pad_names={}
        for key in ("iron","solder"):
            self.tool_parts[key]=set(np.flatnonzero(model.geom_bodyid==model.body(key).id))
        for side in ("left","right"):
            self.pad_names[side]=[{i for i in range(model.ngeom) if f"{side}_tip_{finger}_contact_" in model.geom(i).name}
                                  for finger in ("left","right")]
        self.clip_geom = model.geom("swivel_thumb").id
        self.iron_stand_parts = {gid for gid in range(model.ngeom)
                                 if model.geom(gid).name.startswith(("iron_rest_fork_", "iron_guard_"))}
        self.iron_stand_load_n=0.
        self.tcp_ids = [model.site(f"{side}_tcp").id for side in ("left","right")]
        self.arm_by_geom = np.array([
            {"left": 1, "right": 2}.get(model.body(body).name.split("_", 1)[0], 0)
            for body in model.geom_bodyid
        ])
        self.arm_contact_s = 0.

    def xpos(self,name):
        return self.data.site(name).xpos.copy()

    def loads(self):
        copper=self.geoms["copper_fixed"],self.geoms["copper_free"]
        hot={self.geoms["iron_tip"],self.geoms["iron_chisel"]}
        heated_metal = hot | {self.geoms["iron_shaft"]}
        solder=self.geoms["solder_tip"],self.geoms["solder_lead"]
        state=np.zeros(4)
        grips=np.zeros((3,2))
        bad=0.
        support=0.
        cross_arm=False
        for index,contact in enumerate(self.data.contact[:self.data.ncon]):
            pair={contact.geom1,contact.geom2}
            g1,g2=contact.geom1,contact.geom2
            if contact.efc_address<0:
                continue
            arm1, arm2 = self.arm_by_geom[[g1, g2]]
            cross_arm |= bool(arm1 and arm2 and arm1 != arm2)
            if (heated_metal & pair) or any(g in pair for g in copper) or any(g in pair for g in solder):
                mj.mj_contactForce(self.model,self.data,index,self.force)
                force=max(0.,self.force[0])
                if pair==set(copper):
                    state[0]+=force
                if hot & pair:
                    for j,geom in enumerate(copper):
                        if geom in pair:
                            state[j+1]+=force
                if heated_metal & pair:
                    outside_workpiece=pair.difference(heated_metal).difference(copper).difference(self.iron_stand_parts)
                    if outside_workpiece:
                        bad+=force
                if any(g in pair for g in solder) and any(g in pair for g in copper):
                    state[3]+=force
            if pair & self.tool_parts["iron"] and pair & self.iron_stand_parts:
                mj.mj_contactForce(self.model,self.data,index,self.force)
                support += max(0.,self.force[0])
            for i,(name,side) in enumerate((("iron","right"),("solder","left"),("clip","left"))):
                targets={self.clip_geom} if name=="clip" else self.tool_parts[name]
                if not (pair & targets):
                    continue
                for j,pads in enumerate(self.pad_names[side]):
                    if pair & pads:
                        mj.mj_contactForce(self.model,self.data,index,self.force)
                        grips[i,j]+=max(0.,self.force[0])
        return state,grips,bad,support,cross_arm

    def apply(self):
        data=self.data
        data.qfrc_applied[:]=0
        q=float(data.qpos[self.feed.qposadr[0]])
        # A spring holds the solid wire at its remaining protrusion. The
        # irreversible reference length changes only while native hot contact
        # drives feed; the body remains in the normal contact solver.
        data.qfrc_applied[self.feed.dofadr[0]]=np.clip(
            850.*(self.fed_m-q),-1.9,1.9)
        if self.latched and not self.broken_joint:
            axis=self.clip.dofadr[0]
            angle=float(data.qpos[self.clip.qposadr[0]])
            applied=5.*(self.freeze-angle)-.09*float(data.qvel[axis])
            self.peak_bond_nm=max(self.peak_bond_nm,abs(applied))
            if abs(applied)>.15:
                self.broken_joint=True
                self.events.append(["bond broke",data.time])
            else:
                data.qfrc_applied[axis]+=applied

    def observe(self):
        data=self.data
        dt=self.model.opt.timestep
        a,b=self.xpos("fixed_center"),self.xpos("free_center")
        clip_q=float(data.qpos[self.clip.qposadr[0]])
        self.overlap_m=max(0.,.00685-abs(float(a[1]-b[1])))
        separation=float(np.linalg.norm((a-b)[[0,2]]))
        aligned=(.00306<=self.overlap_m<=.00612 and separation<=.00110
                 and abs(clip_q)<=.025)
        state,grip,bad,self.iron_stand_load_n,cross_arm=self.loads()
        self.contact_loads=state
        self.recent_force+= min(dt/.020,1.)*(state-self.recent_force)
        self.peak_loads=np.maximum(self.peak_loads,state)
        if aligned and state[0]>.015:
            self.stable_alignment+=dt
            self.alignment_hold+=dt
            self.alignment_q=clip_q
        else:
            self.alignment_hold=0.
        if cross_arm:
            self.arm_contact_s+=dt
            self.violations.add("contact between the two arms")
        if min(grip[0])>.25:
            self.peak_iron_grip=max(self.peak_iron_grip,float(min(grip[0])))
        if min(grip[1])>.25:
            self.peak_solder_grip=max(self.peak_solder_grip,float(min(grip[1])))
        if bad>.025 and self.mode!="cold":
            self.hot_contact_insulation_s+=dt
        tip_c=ROOM_C if self.mode=="cold" else TIP_SET_C
        # Lumped energy balance. Iron and copper coupling depend on real force.
        convection= -.85*(self.temperatures-ROOM_C)
        contact_transfer= 5.1*np.clip(state[1:3] / .16,0.,1.) * (tip_c-self.temperatures)
        cross = (self.temperatures[0]-self.temperatures[1])*min(6.,80.*state[0])
        net=convection + contact_transfer + (-cross,cross)
        self.temperatures=np.clip(self.temperatures+net*dt,ROOM_C,TIP_SET_C)
        self.peak_temperature=max(self.peak_temperature,float(min(self.temperatures)))
        self.coarse_contact_seconds+=dt*float(state[1]+state[2]>.05)
        hot_contact=self.alignment_hold>=.12 and min(self.temperatures)>=LIQUIDUS_C
        feed_contact=(state[3]>.014 and min(grip[1])>.25
                      and min(grip[0])>.25 and state[1]+state[2]>.04)
        if hot_contact and feed_contact and self.fed_m<.0031:
            self.fed_m+=dt*min(.0033, .045*state[3])
            if self.fed_m>=NEEDED_MM/1000 and self.freeze is None:
                self.freeze=clip_q
                self.events.append(["wet lap",data.time])
                self.model.geom_rgba[self.geoms["joint_fillet"],3]=1.
        if self.freeze is not None and max(self.temperatures)<LIQUIDUS_C-12 and not self.latched:
            self.latched=True
            self.events.append(["joint cooled",data.time])
        if self.freeze is not None and not self.latched and abs(clip_q-self.freeze)>.012:
            self.broken_joint=True
            self.events.append(["moved before cooling",data.time])
        if self.hot_contact_insulation_s>.28:
            self.violations.add("sustained hot metal contact outside the joint or guard")
        # Return to the holder and preserve the intact joint after releasing.
        body=self.data.body("iron")
        cradle=IRONS["iron"]["origin"]
        if body.xpos[2]<TABLE+.015 or abs(body.xpos[0])>.653 or abs(body.xpos[1]+.12)>.55:
            self.violations.add("iron fell out of the working height or table envelope")
        release_objects=np.stack((a,b,body.xpos,self.data.body("solder").xpos))
        distance=np.linalg.norm(data.site_xpos[self.tcp_ids,None]-release_objects,axis=-1)
        grips_released=np.max(grip[:2])<.04 and float(np.min(distance))>.075
        local_tip=iron_rotation("iron").inv().apply(self.xpos("iron_tip_point")-cradle)
        guarded=(.063<local_tip[0]<.123 and abs(local_tip[1])<.028 and -.015<local_tip[2]<.031)
        iron_at_rest=(np.linalg.norm(body.xpos-cradle)<.019 and np.linalg.norm(body.cvel)<.10
                      and self.iron_stand_load_n>.07 and guarded)
        self.joint_kept=(self.latched and not self.broken_joint and self.alignment_hold>=.12
                         and abs(clip_q-self.freeze)<.016)
        stable=grips_released and iron_at_rest and self.joint_kept
        self.safe_rest_time=self.safe_rest_time+dt if stable else 0.
        return grip

    def measure(self):
        score=0.
        if not self.violations and not self.broken_joint:
            if self.alignment_hold>=.12:
                score=.25
            if self.fed_m>=NEEDED_MM/1000 and self.joint_kept:
                score=.75
                if self.safe_rest_time>=1.:
                    score=1.
        return dict(score=score,time_s=float(self.data.time),
                    overlap_mm=1000*self.overlap_m,clip_angle_deg=np.rad2deg(self.data.qpos[self.clip.qposadr[0]]).item(),
                    native_contacts_n=self.contact_loads.tolist(),peak_contacts_n=self.peak_loads.tolist(),
                    conductor_temperatures_c=self.temperatures.tolist(),peak_min_temperature_c=self.peak_temperature,
                    stable_alignment_s=self.stable_alignment,final_alignment_hold_s=self.alignment_hold,
                    solder_feed_mm=1000*self.fed_m,
                    native_heat_contact_s=self.coarse_contact_seconds,iron_grip_n=self.peak_iron_grip,
                    solder_grip_n=self.peak_solder_grip,bond_solidified=self.latched,bond_retained=self.joint_kept,
                    bond_broken=self.broken_joint,arm_contact_s=self.arm_contact_s,
                    hot_contact_outside_s=self.hot_contact_insulation_s,
                    peak_bond_nm=self.peak_bond_nm,iron_stand_load_n=self.iron_stand_load_n,
                    iron_returned_hold_s=self.safe_rest_time,violations=sorted(self.violations),events=self.events)
