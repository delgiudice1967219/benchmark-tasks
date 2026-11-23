"""Observe physical installation, threading and dispensing state."""

import mujoco as mj
import numpy as np

from paper import PERFORATION, SEGMENTS
from scene import CASE_BASE, LID_HEIGHT, NOZZLE_INLET_RADIUS, NOZZLE_INNER, NOZZLE_TAPER_HEIGHT, TABLE


FINAL_HOLD_SECONDS = 2.0
CORE_CLEARANCE_M = .005
DELIVERED_CLEARANCE_M = .010
MIN_NEXT_TOWEL_M = .025
MAX_NEXT_TOWEL_M = .100
SCORE_TIERS = (.20, .45, .70, 1.00)


class Observer:
    def __init__(self, model):
        self.model = model
        names = [model.geom(index).name for index in range(model.ngeom)]
        self.core_geometry = self._geometry(names, "core_")
        self.roll_geometry = self._geometry(names, "roll_")
        self.case_geometry = self._geometry(names, "base", "case_", "cover_", "nozzle_", "lid_lip", "latch_")
        self.first_geometry = self._geometry(names, "first_wipe_patch_")
        next_geometry = self._geometry(names, "next_wipe_patch_")
        self.towel_geometry = np.r_[self.first_geometry, next_geometry]
        self.protected_geometry = np.r_[self.core_geometry, self.roll_geometry, self.case_geometry,
                                       self.towel_geometry]
        self.robot_geometry = self._geometry(names, "left_", "right_")
        self.table_id = model.geom("table").id
        self.first_path = self._path_ids("first_wipe")
        self.next_path = self._path_ids("next_wipe")
        self.equality_id = model.equality(PERFORATION).id
        self.preceding_ready = False
        self.perforation_intact = True
        self.failure = None
        self.held_since = dict.fromkeys(SCORE_TIERS)

    @staticmethod
    def _geometry(names, *prefixes):
        return np.array([index for index, name in enumerate(names) if name.startswith(prefixes)], dtype=int)

    def _path_ids(self, prefix):
        names = [f"{prefix}_node_{i:02}" for i in range(SEGMENTS)]
        names.append(f"{prefix}_leader")
        return [self.model.site(name).id for name in names]

    def _bounds(self, data, geometry, origin=None, axes=None):
        rotation = data.geom_xmat[geometry].reshape(-1, 3, 3)
        aabb = self.model.geom_aabb[geometry]
        center = data.geom_xpos[geometry] + np.einsum("nij,nj->ni", rotation, aabb[:, :3])
        if axes is not None:
            center = (center - origin) @ axes
            rotation = axes.T @ rotation
        extent = np.einsum("nij,nj->ni", np.abs(rotation), aabb[:, 3:])
        return np.vstack(((center - extent).min(axis=0), (center + extent).max(axis=0)))

    @staticmethod
    def _clearance(one, two):
        # A conservative lower bound using separation along each coordinate axis.
        return float(max(0., np.max(one[0] - two[1]), np.max(two[0] - one[1])))

    def _on_table(self, pairs, geometry):
        return bool(np.any((pairs == self.table_id).any(axis=1) & np.isin(pairs, geometry).any(axis=1)))

    def _core_clearance(self, data, roll, case_bounds):
        axes = roll.xmat.reshape(3, 3)
        core_roll_frame = self._bounds(data, self.core_geometry, roll.xpos, axes)
        roll_bounds = self._bounds(data, self.roll_geometry, roll.xpos, axes)
        core_world = self._bounds(data, self.core_geometry)
        return min(self._clearance(core_roll_frame, roll_bounds), self._clearance(core_world, case_bounds))

    def _roll_seated(self, dispenser, roll):
        r = dispenser.xmat.reshape(3,3)
        difference = r.T @ (roll.xpos-dispenser.xpos)
        alignment = float(r[:,2] @ roll.xmat.reshape(3,3)[:,2])
        return bool(np.linalg.norm(difference[:2]) <= .0045 and abs(difference[2]-CASE_BASE) <= .005
                    and alignment >= np.cos(np.deg2rad(2.5)))

    @staticmethod
    def _nozzle_crossings(path, lid):
        r = lid.xmat.reshape(3,3)
        local = (path-lid.xpos) @ r - [0,-.084,0]
        crossings = []
        bore = ((0., NOZZLE_INLET_RADIUS), (NOZZLE_TAPER_HEIGHT, NOZZLE_INNER), (LID_HEIGHT, NOZZLE_INNER))
        for plane, radius in bore:
            last = local[:-1]
            nxt = local[1:]
            # The paper is traced in material order from the supply roll toward the leader.
            ascending = (last[:,2] <= plane) & (nxt[:,2] > plane)
            descending = (nxt[:,2] <= plane) & (last[:,2] > plane)
            candidates = np.flatnonzero(ascending | descending)
            a = last[candidates]
            b = nxt[candidates]
            fractions = (plane - a[:, 2]) / (b[:, 2] - a[:, 2])
            points = a + fractions[:, None] * (b - a)
            # Ignore plane crossings away from the aperture when the free tail droops.
            within_aperture = np.flatnonzero(np.linalg.norm(points[:, :2], axis=1) <= radius - .001)
            if len(within_aperture) != 1:
                return None
            hit = within_aperture[0]
            index = candidates[hit]
            if not ascending[index]:
                return None
            crossings.append((index, fractions[hit]))
        return crossings if all(a < b for a, b in zip(crossings, crossings[1:])) else None

    @staticmethod
    def _remaining_length(path, crossing):
        last_index, fraction = crossing
        lengths = np.linalg.norm(np.diff(path[last_index:],axis=0),axis=1)
        return float(lengths.sum()-fraction*lengths[0])

    @staticmethod
    def _closed(data):
        return bool(abs(data.joint("lid_hinge").qpos[0]) <= .035
                    and abs(data.joint("latch_deflection").qpos[0]) <= .035)

    def audit_substep(self, data):
        """Call at the physics rate after checking the tearable perforation."""
        if self.failure:
            return
        if self._bounds(data, self.towel_geometry)[0, 2] <= .005:
            self.failure = "paper touched the floor"
            return
        intact = bool(data.eq_active[self.equality_id])
        if self.perforation_intact and not intact:
            roll, dispenser = (data.body(name) for name in ("roll", "dispenser"))
            if not (self.preceding_ready and self._closed(data) and self._roll_seated(dispenser, roll)):
                self.failure = "paper separated before the refill was seated, threaded and latched"
        self.perforation_intact = intact

    def _held_grade(self, candidate, time_s):
        holds = {}
        for tier, start in self.held_since.items():
            if candidate < tier:
                self.held_since[tier] = None
            else:
                if start is None:
                    start = self.held_since[tier] = time_s
                holds[tier] = max(0., float(time_s - start))
        score = max((tier for tier, duration in holds.items()
                     if duration >= FINAL_HOLD_SECONDS - 1e-7), default=0.)
        return score, holds.get(score or candidate, 0.)

    def observe(self, data):
        self.audit_substep(data)
        roll, dispenser, lid = (data.body(name) for name in ("roll","dispenser","lid"))
        active_pairs = data.contact.geom[data.contact.efc_address >= 0]
        case_bounds = self._bounds(data, self.case_geometry)
        core_clearance = self._core_clearance(data, roll, case_bounds)
        core_out = self._on_table(active_pairs, self.core_geometry) and core_clearance >= CORE_CLEARANCE_M
        seated = self._roll_seated(dispenser,roll)
        closed = self._closed(data)
        intact = bool(data.eq_active[self.equality_id])
        first = data.site_xpos[self.first_path]
        remaining = data.site_xpos[self.next_path]
        connected_path = np.vstack((remaining[:-1],first))
        threaded = self._nozzle_crossings(connected_path,lid) if intact else None
        next_crossings = self._nozzle_crossings(remaining,lid) if not intact else None
        clearance = (self._remaining_length(remaining,next_crossings[-1]) if next_crossings else 0.)

        if intact:
            self.preceding_ready = bool(core_out and seated and closed and threaded)
        first_bounds = self._bounds(data, self.first_geometry)
        on_table = (self._on_table(active_pairs, self.first_geometry)
                    and first_bounds[0, 2] >= TABLE - .005 and first_bounds[1, 2] <= TABLE + .040)
        sheet_clearance = self._clearance(first_bounds, case_bounds)
        ready = seated and closed and bool(threaded or next_crossings)
        delivered = bool(not intact and MIN_NEXT_TOWEL_M <= clearance <= MAX_NEXT_TOWEL_M and on_table
                         and sheet_clearance >= DELIVERED_CLEARANCE_M)
        credit_core = core_out and not self.failure
        credit_roll = credit_core and seated
        credit_ready = credit_roll and ready
        credit_towel = credit_ready and delivered
        eligible = round(.20*bool(credit_core)+.25*bool(credit_roll)+.25*bool(credit_ready)+.30*bool(credit_towel), 2)
        involved = np.isin(active_pairs, self.protected_geometry).any(axis=1)
        hands_clear = not np.any(involved & np.isin(active_pairs, self.robot_geometry).any(axis=1))
        candidate = eligible if hands_clear else 0.
        score, hold_seconds = self._held_grade(candidate, data.time)
        return {
            "score": score,
            "eligible_score": eligible,
            "released_hold_s": hold_seconds,
            "hands_clear": hands_clear,
            "core_set_aside": core_out,
            "core_clearance_m": core_clearance,
            "refill_seated": seated,
            "lid_latched": closed,
            "threaded_nozzle": bool(threaded or next_crossings),
            "next_towel_outside_m": clearance,
            "first_towel_delivered": delivered,
            "delivered_towel_clearance_m": sheet_clearance,
            "failure": self.failure,
        }
