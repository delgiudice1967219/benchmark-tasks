"""Passive contact-triggered solder-transfer model and in-process inspection."""
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from scene import CAP_LEN, CAP_RADIUS, HOLE_RADIUS, PAD_XY, PCB_T, PCB_Z

AIR_C = 25.
IRON_C = 335.
LIQUIDUS_C = 217.
JOIN_VOLUME = .81
BRIDGE_VOLUME = 3.00
CAP_BELOW = PCB_Z-PCB_T


class Solder:
    def __init__(self, model, data, enabled, staged):
        self.model,self.data = model,data
        self.enabled = enabled
        self.staged = staged
        self.tool_ids = [model.body(name).id for name in ('iron','wire')]
        self.hand_sites = [model.site(f'{side}_tcp').id for side in ('left','right')]
        self.fingers = [set(model.body(f'{side}_tip_{finger}').id for finger in ('left','right'))
                        for side in ('left','right')]
        self.handles = [{model.geom(name).id for name in names} for names in (
            ('iron_grip','iron_ring','iron_tail'), ('wire_grip','wire_cap'),
        )]
        self.robot_side = {i: name.split('_')[0] for i in range(model.nbody)
                           if (name := model.body(i).name).startswith(('left_', 'right_'))}
        self.arm_collision_s = 0.
        self.held_s = 0.
        initial_axes=[]
        for body in self.tool_ids:
            matrix=np.empty(9)
            mj.mju_quat2Mat(matrix,model.body_quat[body])
            initial_axes.append(matrix.reshape(3,3)[:,2])
        self.tool_axes=np.asarray(initial_axes)
        self.cap=model.body('capacitor').id
        self.board=model.body('motherboard').id
        self.cap_geom=model.geom('cap_shell').id
        self.iron=model.geom('iron_hot_tip').id
        self.hot_shaft=model.geom('iron_chisel_shaft').id
        self.hot_geoms={model.geom(name).id for name in
                        ('iron_hot_tip','iron_chisel_shaft','iron_neck','iron_barrel')}
        self.tool_unsafe_contact_s=0.
        self.wire=model.geom('solder_wire').id
        self.joint_ids=[model.site(f'lead_tip_{i}').id for i in range(2)]
        self.lead_geoms=[model.geom(f'lead_{i}').id for i in range(2)]
        self.lands=[{model.geom(f'land_{i}_{piece}').id for piece in range(12)}
                    for i in range(2)]
        self.weld=model.equality('installed_capacitor').id
        self.temperature=np.full(2,AIR_C)
        self.peak=np.full(2,AIR_C)
        self.dose=np.zeros(2)
        self.wetting=np.zeros(2,dtype=bool)
        self.max_heating=np.zeros(2)
        self.max_feeding=np.zeros(2)
        self.heating_time=np.zeros(2)
        self.feeding_time=np.zeros(2)
        self.stable=0.
        self.seat_hold=0.
        self.ever_seated=False
        self.part_withdrew=False
        self.damage=False
        self.bridge=False
        self.fillet_ids=[[model.geom(f'{part}_{i}').id for part in ('fillet','fillet_taper')]
                         for i in range(2)]
        self.latest_heat=np.zeros(2)
        self.latest_wire=np.zeros(2)
        self.latest_miscontact=[]
        self.lead_lands=[f'lead_{i}' for i in range(2)]
        self.hot_contact_other=0.
        self.events=[]

    def pose(self):
        cap_position=self.data.xpos[self.cap]
        axes=self.data.xmat[self.cap].reshape(3,3)
        face=cap_position+CAP_LEN/2*axes[:,2]
        tips=self.data.site_xpos[self.joint_ids]
        radial=np.linalg.norm(tips[:,:2]-PAD_XY,axis=1)
        gap=CAP_BELOW-face[2]
        exposure=tips[:,2]-PCB_Z
        polarity=axes[0,0]>.985
        seated=(polarity and np.all(radial<.00062) and
                np.all((exposure>.0016)&(exposure<.006)) and -.00015<=gap<=.0011
                and axes[2,2]>.99)
        return seated, float(gap),radial,exposure,polarity

    def update(self):
        dt=self.model.opt.timestep
        seat, _, _, exposure, _=self.pose()
        self.seat_hold=self.seat_hold+dt if seat else 0.
        self.ever_seated |= self.seat_hold>.18
        self.part_withdrew |= self.staged and bool(np.any(exposure<=0.))
        heat=np.zeros(2)
        wire=np.zeros(2)
        wrongful=False
        unsafe=False
        crossing=False
        arm_collision=False
        grip=[[set(),set()],[set(),set()]]
        nearby=[]
        force=np.zeros(6)
        for ci, contact in enumerate(self.data.contact):
            a,b=(int(g) for g in contact.geom)
            body_a,body_b=(int(self.model.geom_bodyid[g]) for g in (a,b))
            across=(body_a in self.robot_side and body_b in self.robot_side
                    and self.robot_side[body_a] != self.robot_side[body_b])
            tool_handle=any(a in names or b in names for names in self.handles)
            if not (across or tool_handle or self.wire in (a,b) or a in self.hot_geoms or b in self.hot_geoms):
                continue
            mj.mj_contactForce(self.model,self.data,ci,force)
            normal=max(0.,float(force[0]))
            if normal<.001:
                continue
            arm_collision |= across and normal>.10
            if tool_handle and normal>.02:
                for index,names in enumerate(self.handles):
                    for handle,finger in ((a,body_b),(b,body_a)):
                        if handle not in names:
                            continue
                        for arm, fingers in enumerate(self.fingers):
                            if finger in fingers:
                                grip[index][arm].add(finger)
            pair={a,b}
            if normal>.06 and pair.intersection(self.hot_geoms):
                others=pair-self.hot_geoms
                if any(self.model.geom(index).name.startswith(('left_','right_','table'))
                       for index in others):
                    unsafe=True
            zone_found=False
            for i in range(2):
                on_land=bool(pair & self.lands[i])
                on_lead=self.lead_geoms[i] in pair
                if not (on_land or on_lead):
                    continue
                zone_found=True
                if self.iron in pair:
                    heat[i]+=normal
                if self.wire in pair:
                    wire[i]+=normal
            if not zone_found and self.iron in pair:
                other = b if a == self.iron else a
                if self.model.geom_bodyid[other] == self.board and normal>.15:
                    wrongful=True
                    nearby.append(self.model.geom(other).name)
            if self.iron in pair and self.wire in pair:
                # Melting solder against the iron alone cannot create a joint.
                crossing=True
        self.latest_heat=heat
        self.latest_wire=wire
        self.latest_miscontact=nearby
        self.max_heating=np.maximum(self.max_heating,heat)
        self.max_feeding=np.maximum(self.max_feeding,wire)
        self.hot_contact_other+=dt if wrongful else 0.
        self.tool_unsafe_contact_s+=dt if unsafe and self.enabled else 0.
        self.arm_collision_s+=dt if arm_collision else 0.
        opposite_grips=any(len(grip[0][arm])==2 and len(grip[1][1-arm])==2 for arm in (0,1))
        self.held_s+=dt if opposite_grips else 0.
        if self.hot_contact_other>.22 or self.tool_unsafe_contact_s>.06 or self.arm_collision_s>.05:
            self.damage=True
        target=np.where(heat>.035,IRON_C,AIR_C)
        tau=np.where(heat>.035,.68,1.05)
        if self.enabled:
            self.temperature+=(target-self.temperature)*dt/tau
        self.peak=np.maximum(self.peak,self.temperature)
        pre=self.wetting.copy()
        self.heating_time+=(heat>.035)*dt
        conditions=(self.temperature>=LIQUIDUS_C)&(wire>.018)&(heat>.035)&seat&opposite_grips
        self.feeding_time+=conditions*dt
        self.dose+=conditions*dt*1.42
        self.wetting|=self.dose>=JOIN_VOLUME
        for i in np.where(self.wetting&~pre)[0]:
            self.events.append({'time_s':float(self.data.time),'land':int(i),
                                'temperature_c':float(self.temperature[i]),'solder_mm3':float(self.dose[i])})
        self.bridge |= bool(np.all(wire>.018) and np.any(conditions))
        self.damage |= bool(np.any(self.heating_time>3.015) or np.any(self.dose>BRIDGE_VOLUME))
        self.bridge |= crossing and bool(np.all(self.temperature>=LIQUIDUS_C))
        if (np.all(self.wetting) and np.all(self.temperature<LIQUIDUS_C)
                and seat and not self.damage and not self.bridge and not self.data.eq_active[self.weld]):
            self.lock_current_pose()
        docked,clear=self.finished_layout()
        if (self.data.eq_active[self.weld] and seat and docked and clear
                and np.all(self.temperature<140.) and not np.any(heat>.035) and not np.any(wire>.018)):
            self.stable+=dt
        else:
            self.stable=0.

    def lock_current_pose(self):
        # Create the cooled-joint retention constraint at the native part pose,
        # preserving exactly the measured transform rather than snapping the part.
        board_rot=self.data.xmat[self.board].reshape(3,3)
        cap_rot=self.data.xmat[self.cap].reshape(3,3)
        xyz=board_rot.T@(self.data.xpos[self.cap]-self.data.xpos[self.board])
        xyzw=Rotation.from_matrix(board_rot.T@cap_rot).as_quat()
        self.model.eq_data[self.weld,3:10]=np.r_[xyz,xyzw[3],xyzw[:3]]
        self.data.eq_active[self.weld]=1
        self.events.append({'time_s':float(self.data.time),'mechanical_bond_at_native_pose':True})

    def finished_layout(self):
        centers=self.data.xpos[self.tool_ids]
        axes=self.data.xmat[self.tool_ids].reshape(2,3,3)[:,:,2]
        stored=np.all(np.linalg.norm(centers-self.model.body_pos[self.tool_ids],axis=1)<.010)
        angles=np.all(np.sum(axes*self.tool_axes,axis=1)>.96)
        velocities=self.data.cvel[self.tool_ids]
        still=np.all(np.linalg.norm(velocities[:,3:],axis=1)<.010)
        fingertips=self.data.site_xpos[self.hand_sites]
        clear=(np.all(np.linalg.norm(fingertips[:,None,:]-centers[None,:,:],axis=2)>.08)
               and np.all(np.linalg.norm(fingertips-self.data.xpos[self.cap],axis=1)>.08))
        return bool(stored and angles and still), bool(clear)

    def camera_material(self):
        for index,ids in enumerate(self.fillet_ids):
            self.model.geom_rgba[ids,3] = float(self.wetting[index])

    def sample(self):
        seated,gap,radial,exposure,polarity=self.pose()
        contaminated=self.damage or self.bridge or self.part_withdrew
        solder_actions=int(np.sum(self.wetting))
        score=0.
        if seated and not contaminated:
            score=(0. if self.staged else .25) if not solder_actions else (.5 if solder_actions==1 else .75)
            if solder_actions==2 and self.stable>=1.:
                score=1.
        docked,clear=self.finished_layout()
        return dict(score=score,time_s=float(self.data.time),seated=bool(seated),
                    tools_in_holders=docked, hands_clear=clear,
                    hot_tool_contact_with_robot_or_table_s=self.tool_unsafe_contact_s,
                    cross_arm_contact_s=self.arm_collision_s, opposite_tools_gripped_s=self.held_s,
                    seat_gap_mm=gap*1e3,lead_radial_error_mm=(radial*1e3).tolist(),
                    exposed_lead_mm=(exposure*1e3).tolist(),correct_polarity=bool(polarity),
                    lead_tips_m=self.data.site_xpos[self.joint_ids].tolist(),
                    pad_temperature_c=self.temperature.tolist(),peak_temperature_c=self.peak.tolist(),
                    solder_volume_mm3=self.dose.tolist(),wetted_lands=self.wetting.tolist(),
                    pad_iron_contact_n=self.latest_heat.tolist(),pad_solder_contact_n=self.latest_wire.tolist(),
                    max_iron_contact_n=self.max_heating.tolist(),max_solder_contact_n=self.max_feeding.tolist(),
                    heating_s=self.heating_time.tolist(),feeding_s=self.feeding_time.tolist(),
                    unwanted_hot_contact_s=self.hot_contact_other,latest_hot_contact_elsewhere=self.latest_miscontact,
                    bridge=bool(self.bridge),heat_damage=bool(self.damage),part_withdrew=self.part_withdrew,
                    cooled_and_undisturbed_s=self.stable,ever_seated=bool(self.ever_seated))
