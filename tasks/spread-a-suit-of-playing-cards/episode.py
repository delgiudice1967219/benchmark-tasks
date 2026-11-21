"""YAM action interface and method audit for the single-stroke card ribbon."""

from dataclasses import asdict

import mujoco as mj
import numpy as np

from scene import COUNT, FELT_BOUNDS, FELT_Z, build_model, reset
from scoring import RibbonGeometry, RibbonScore


CONTROL_HZ = 20
ACTION_LIMIT = 900
TASK_SLUG = 'spread-a-suit-of-playing-cards'
INSTRUCTION = ('Spread the thirteen face-up hearts across the felt in one continuous overlapping row, '
               'keeping Ace through King in order and every rank readable. Leave the row untouched '
               'for two seconds.')
SIDES = ('left', 'right')


class SweepAudit:
    """Use ordinary arm kinematics and physical contacts to detect one moving packet."""

    def __init__(self, model, data):
        names = [model.geom(i).name for i in range(model.ngeom)]
        self.rank_by_geom = np.full(model.ngeom, -1, dtype=int)
        self.side_by_geom = np.full(model.ngeom, -1, dtype=int)
        self.arm_by_geom = np.full(model.ngeom, -1, dtype=int)
        for geom, name in enumerate(names):
            if name.startswith('card_') and '_panel_' in name:
                self.rank_by_geom[geom] = int(name[5:7])
            for side, prefix in enumerate(SIDES):
                if name.startswith(f'{prefix}_') and '_contact_' in name:
                    self.arm_by_geom[geom] = side
                if name.startswith(f'{prefix}_tip_') and '_contact_' in name:
                    self.side_by_geom[geom] = side
        self.bodies = np.array([model.body(f'card_{i:02d}').id for i in range(COUNT)])
        self.sites = np.array([model.site(f'{side}_tcp').id for side in SIDES])
        self.start_cards = data.xpos[self.bodies].copy()
        self.previous_cards = self.start_cards.copy()
        self.active = None
        self.sweep_ended = False
        self.failed = False
        self.first_x = self.max_x = self.contact_time = 0.
        self.max_group = 0
        self.touched = set()
        self.arms_used = set()

    def observe_contact(self, data, touched, arms_in_contact):
        for contact in data.contact[:data.ncon]:
            first, second = contact.geom
            if min(first, second) < 0 or contact.dist > 0:
                continue
            for finger, card in ((first,second),(second,first)):
                side = int(self.side_by_geom[finger])
                rank = int(self.rank_by_geom[card])
                arm = int(self.arm_by_geom[finger])
                if rank >= 0 and arm >= 0:
                    arms_in_contact.add(arm)
                if side >= 0 and rank >= 0:
                    touched[side].add(rank)

    def update(self, data, touched, arms_in_contact):
        positions = data.xpos[self.bodies]
        motions = np.linalg.norm(positions[:, :2] - self.previous_cards[:, :2], axis=1)
        moving = int(np.count_nonzero(motions > .0005))
        points = data.site_xpos[self.sites]
        high = float(np.max(positions[:, 2]) - FELT_Z)
        if high > .025 or float(np.min(positions[:, 2])) < FELT_Z - .003:
            self.failed = True

        if self.active is None:
            if moving >= 2 and max(map(len,touched)) >= 2:
                self.active = int(np.argmax([len(hits) for hits in touched]))
                self.first_x = self.max_x = float(points[self.active, 0])
                self.contact_time = float(data.time)
            elif moving and float(np.max(np.abs(positions[:,:2]-self.start_cards[:,:2]))) > .009:
                # Deliberate spreading may start with a brief bevel. Pulling a card
                # away before a packet moves is a separate dealing action.
                self.failed = True

        self.arms_used.update(arms_in_contact)
        if self.active is not None:
            hits = touched[self.active]
            x = float(points[self.active, 0])
            idle_contact_time = float(data.time) - self.contact_time
            if self.arms_used - {self.active}:
                self.failed = True
            if idle_contact_time > .40:
                self.sweep_ended = True
            if self.sweep_ended and (moving or arms_in_contact):
                self.failed = True
            if self.active in arms_in_contact and not hits:
                # The fingers must do the spreading. A separate push with
                # the wrist or an arm link cannot extend the valid stroke.
                self.failed = True
            if hits:
                if x < self.max_x - .018:
                    self.failed = True
                self.max_x = max(self.max_x, x)
                self.max_group = max(self.max_group, len(hits))
                self.touched.update(hits)
                self.contact_time = float(data.time)

        self.previous_cards = positions.copy()
        released = not arms_in_contact and np.min(np.linalg.norm(points[:,None,:] - positions[None,:,:], axis=2)) >= .055
        return bool(released)

    @property
    def verified(self):
        return (not self.failed and self.active is not None and self.max_group >= 3
                and len(self.touched) >= 4 and self.max_x - self.first_x >= .035)


class RibbonTrial:
    def __init__(self, *, art=False):
        self.model = build_model(add_prints=art)
        self.data = reset(self.model)
        self.arms = [np.array([self.model.actuator(f'{side}_joint{i}').id for i in range(1,7)])
                     for side in SIDES]
        self.grippers = [np.array([self.model.actuator(f'{side}_finger_{sign}').id for sign in (-1,1)])
                         for side in SIDES]
        self.substeps = int(round(1 / CONTROL_HZ / self.model.opt.timestep))
        self.method = SweepAudit(self.model, self.data)
        x0,x1,y0,y1 = FELT_BOUNDS
        self.grade = RibbonScore(RibbonGeometry(self.model,FELT_Z,((x0,y0),(x1,y1))))
        self.value = 0.
        self.layout = self.grade.geometry.assess(self.data)
        self.actions = 0

    def step(self, action):
        ctrl = np.asarray(action,dtype=float).reshape(2,7)
        for i, targets in enumerate(ctrl):
            self.data.ctrl[self.arms[i]] = targets[:6]
            self.data.ctrl[self.grippers[i]] = targets[6] * .0475
        touched = (set(),set())
        arms_in_contact = set()
        for substep in range(self.substeps):
            mj.mj_step(self.model,self.data)
            self.method.observe_contact(self.data, touched, arms_in_contact)
        released = self.method.update(self.data, touched, arms_in_contact)
        self.value, self.layout = self.grade.update(self.data,1./CONTROL_HZ,
                                                     sweep_verified=self.method.verified,
                                                     released=released)
        self.actions += 1

    def summary(self):
        layout = asdict(self.layout)
        layout['ordered_links'] = [bool(link) for link in layout['ordered_links']]
        layout['overlap_fraction'] = [float(fraction) for fraction in layout['overlap_fraction']]
        return {'score': float(self.value), 'layout': layout,
                'actions': self.actions, 'clear_seconds': round(self.grade.clear_seconds,4),
                'continuous_sweep_verified': bool(self.method.verified),
                'method_invalidated': bool(self.method.failed),
                'hand_side': SIDES[self.method.active] if self.method.active is not None else None,
                'sweep_contact_span_mm': round(1000*(self.method.max_x-self.method.first_x),3),
                'peak_simultaneous_card_contacts': self.method.max_group,
                'distinct_cards_contacted': len(self.method.touched)}
