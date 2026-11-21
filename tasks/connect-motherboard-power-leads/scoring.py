"""Observe the fitted housings, captured clips and independent physical pull."""

from copy import copy

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from capture import read_latches
from scene import CONNECTORS, HEAD_HEIGHT, latch_centers

RELEASE_HOLD = 2.0


class Grade:
    """Observe seat faces and their loaded retaining shoulders at every substep."""
    def __init__(self,model,data,arms):
        self.model,self.data,self.arms=model,data,arms
        self.header_ids=[model.body(f"{spec.name}_header").id for spec in CONNECTORS]
        self.plug_ids=[model.body(f"{spec.name}_plug").id for spec in CONNECTORS]
        self.plug_geoms=[]
        self.latch={}
        self.hold=np.zeros(len(CONNECTORS))
        self.partial=np.zeros(len(CONNECTORS))
        self.peak_axial=np.zeros(len(CONNECTORS))
        self.peak_case_load=0.
        self.lock_load=[np.zeros(len(latch_centers(spec))) for spec in CONNECTORS]
        self.peak_clip_angle=[np.zeros(len(latch_centers(spec))) for spec in CONNECTORS]
        self.correction=0.
        self.last=[]
        self.arm_geom=np.fromiter((g.name.startswith(("left_","right_")) for g in
                                   (model.geom(i) for i in range(model.ngeom))),dtype=bool)
        self.case_geom=np.fromiter((g.name.startswith(("case_","rear_","front_","rail_","upper_",
                                   "lower_","PSU","dimm","cpu_","pcie","polymer_","board_")) for g in
                                   (model.geom(i) for i in range(model.ngeom))),dtype=bool)
        for index,spec in enumerate(CONNECTORS):
            self.plug_geoms.append(set(j for j in range(model.ngeom)
                 if model.geom(j).name.startswith((spec.name+"_grip",spec.name+"_tooth",spec.name+"_clip_nose",
                                                    spec.name+"_flex_beam",spec.name+"_clip_root",
                                                    spec.name+"_lead_contact_"))))
            for lane,_ in enumerate(latch_centers(spec)):
                self.latch[frozenset((model.geom(f"{spec.name}_clip_nose_{lane}").id,
                                      model.geom(f"{spec.name}_ledge_{lane}").id))]=(index,lane)
        self.wrench=np.empty(6)

    def update(self):
        d,m=self.data,self.model
        for row in self.lock_load:
            row[:]=0.
        axial=np.zeros(len(CONNECTORS))
        case_load=0.
        pad_touch=[False]*len(CONNECTORS)
        axial_direction=np.array([0.,0.,1.])
        for slot,(a,b) in enumerate(d.contact.geom):
            if min(a,b)<0:continue
            pair=frozenset((int(a),int(b)))
            latch=self.latch.get(pair)
            if latch is not None:
                mj.mj_contactForce(m,d,slot,self.wrench)
                normal=d.contact.frame[slot,:3]
                index,lane=latch
                plug_second=m.geom(b).name.endswith(f"_clip_nose_{lane}")
                load_on_nose=float(np.dot(normal,axial_direction)*self.wrench[0])*(1 if plug_second else -1)
                self.lock_load[index][lane]+=max(0.,-load_on_nose)
            if self.arm_geom[a] and self.case_geom[b] or self.arm_geom[b] and self.case_geom[a]:
                mj.mj_contactForce(m,d,slot,self.wrench)
                case_load+=self.wrench[0]
            for index,geoms in enumerate(self.plug_geoms):
                if (a in geoms)==(b in geoms):continue
                other=int(b) if a in geoms else int(a)
                plug_second=b in geoms
                if self.arm_geom[other]:
                    pad_touch[index]=True
                elif "_beam_pad_" in m.geom(other).name:
                    mj.mj_contactForce(m,d,slot,self.wrench)
                    force=self.wrench[0]*float(d.contact.frame[slot,2])*(1 if plug_second else -1)
                    axial[index]+=max(0.,force)
        self.peak_case_load=max(self.peak_case_load,case_load)
        np.maximum(self.peak_axial,axial,out=self.peak_axial)
        self.last=[]
        for index,(spec,head,plug) in enumerate(zip(CONNECTORS,self.header_ids,self.plug_ids)):
            matrix=d.xmat[head].reshape(3,3)
            p=matrix.T@(d.xpos[plug]-d.xpos[head])
            orientation=matrix.T@d.xmat[plug].reshape(3,3)
            corners=np.array([[x,y,0.] for x in (-spec.width/2,spec.width/2) for y in (-.0048,.0048)])
            gap=(corners@orientation[2])+p[2]-HEAD_HEIGHT
            angle=float(np.linalg.norm(Rotation.from_matrix(orientation).as_rotvec()))
            clips=np.array([float(d.joint(f"{spec.name}_clip_bend_{i}").qpos[0])
                             for i in range(len(latch_centers(spec)))])
            self.peak_clip_angle[index]=np.maximum(self.peak_clip_angle[index],clips)
            retained=bool(np.min(self.lock_load[index])>.18 and -0.0009 <= np.min(gap)
                          and np.max(gap)<.0015 and np.linalg.norm(p[:2])<.0013 and angle<.10)
            try_depth=bool(np.max(gap)<.0043 and np.min(gap)>-.0015
                           and np.linalg.norm(p[:2])<.0016 and angle<.16)
            hands_away = all(np.linalg.norm(d.site_xpos[arm.site]-d.xpos[plug])>.048 for arm in self.arms.values())
            self.partial[index]=self.partial[index]+m.opt.timestep if try_depth and hands_away and not pad_touch[index] else 0.
            self.hold[index]=self.hold[index]+m.opt.timestep if retained and hands_away and not pad_touch[index] else 0.
            self.last.append(dict(name=spec.name, plug_xyz_m=d.xpos[plug].tolist(),
                                  xy_error_mm=float(np.linalg.norm(p[:2])*1000),
                                  seam_gap_mm=(gap*1000).tolist(), angle_deg=float(angle*180/np.pi),
                                  axial_contact_n=float(axial[index]), peak_axial_contact_n=float(self.peak_axial[index]),
                                  clips_deg=np.rad2deg(clips).tolist(),
                                  max_clip_flex_deg=np.rad2deg(self.peak_clip_angle[index]).tolist(),
                                  retention_loads_n=self.lock_load[index].tolist(), released_hold_s=float(self.hold[index]),
                                  partial_hold_s=float(self.partial[index]), robot_contact=pad_touch[index]))

    def report(self):
        scores=np.where(self.hold>=RELEASE_HOLD,.5,np.where(self.partial>=RELEASE_HOLD,.25,0.))
        return dict(time_s=float(self.data.time),score=float(scores.sum()),
                    connector_scores=scores.tolist(),connectors=self.last,peak_case_contact_n=self.peak_case_load)


def captured(model, data, spec):
    return all(row["axial_clearance_mm"] is not None and -.20 <= row["axial_clearance_mm"] <= 1.50
               and row["x_overlap_mm"] >= .50 and row["y_overlap_mm"] >= .15
               for row in read_latches(model, data, spec))


def eligible_connector(row):
    return (-.90 <= min(row["seam_gap_mm"]) and max(row["seam_gap_mm"]) < 1.50
            and row["xy_error_mm"] < 1.30 and row["angle_deg"] < np.rad2deg(.10)
            and not row["robot_contact"])


class UnloadedLatchGrade(Grade):
    def update(self):
        previous = self.hold.copy()
        super().update()
        for i, spec in enumerate(CONNECTORS):
            near = self.data.xpos[self.plug_ids[i]]
            clear = all(np.linalg.norm(self.data.site_xpos[arm.site] - near) > .048 for arm in self.arms.values())
            ready = eligible_connector(self.last[i]) and clear and captured(self.model, self.data, spec)
            self.hold[i] = previous[i] + self.model.opt.timestep if ready else 0.
            self.last[i]["released_hold_s"] = float(self.hold[i])

    def proof(self):
        results = []
        for i, spec in enumerate(CONNECTORS):
            if self.hold[i] < RELEASE_HOLD:
                results.append(dict(name=spec.name, passed=False, reason="Released geometric hold missing"))
                continue
            pull_data = copy(self.data)
            observe = UnloadedLatchGrade(self.model, pull_data, self.arms)
            direction = pull_data.xmat[self.header_ids[i]].reshape(3, 3)[:, 2].copy()
            dt = self.model.opt.timestep
            for sample in range(round((.30 + RELEASE_HOLD) / dt)):
                progress = min(1., (sample + 1) * dt / .30)
                pull_data.xfrc_applied[self.plug_ids[i], :3] = 3. * progress * direction
                mj.mj_step(self.model, pull_data)
                observe.update()
            results.append(dict(name=spec.name, passed=bool(observe.hold[i] >= RELEASE_HOLD),
                                pull_n=3., final=observe.last[i],
                                latch_geometry=read_latches(self.model, pull_data, spec)))
        return results
