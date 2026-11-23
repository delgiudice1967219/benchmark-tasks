"""Native articulated model of the first two folded centre-feed towels."""

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


SHEET_LENGTH = .380
PERFORATION = "sheet_perforation"
SEGMENTS = 19
# Parameters for the folded dry-paper approximation. They do not represent
# measurements of the commercial wipe or of the nozzle in the reference unit.
TEAR_TENSION_N = 3.0
TEAR_HOLD_S = .022
CREASE_YIELD_RAD = .195
CREASE_RESIDUAL_RAD = .025


def _coil_points(count):
    radius = .0155
    axial_pitch = .0115
    segment = SHEET_LENGTH / count
    # Pick a helix increment whose actual polyline chords have the nominal length.
    low, high = 0., np.pi
    for _ in range(40):
        theta = (low + high) / 2
        chord = np.hypot(2 * radius * np.sin(theta / 2), axial_pitch * theta / (2*np.pi))
        if chord < segment:
            low = theta
        else:
            high = theta
    straight_leader_sections = 5
    arc = np.arange(2*count+1-straight_leader_sections) * (low + high) / 2
    angle = np.pi - arc[-1] + arc
    coil = np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                            .012 + axial_pitch * arc / (2*np.pi)))
    leader = coil[-1] + np.column_stack((np.zeros((straight_leader_sections,2)),
                                         segment*np.arange(1,straight_leader_sections+1)))
    return np.vstack((coil, leader))


def _axis_frame(start, end):
    vertical = end - start
    vertical /= np.linalg.norm(vertical)
    broad = np.cross([0., 0., 1.], vertical)
    width = np.linalg.norm(broad)
    broad = broad / width if width > .001 else np.array((1.,0.,0.))
    thin = np.cross(vertical, broad)
    return Rotation.from_matrix(np.column_stack((broad, thin, vertical)))


def _chain(node, parent, points, name, world_rotation, world_origin, *, free=False):
    parent_origin = np.zeros(3)
    parent_rotation = Rotation.identity()
    for index, (start, end) in enumerate(zip(points, points[1:])):
        bunched_tip = free and index >= len(points)-6
        orientation = _axis_frame(start, end)
        offset = parent_rotation.inv().apply(start - parent_origin)
        local_orientation = parent_rotation.inv() * orientation
        length = np.linalg.norm(end - start)
        if free and index == 0:
            offset = world_rotation.apply(start) + world_origin
            local_orientation = world_rotation * orientation
        body = node(parent, "body", name=f"{name}_{index:02}", pos=offset,
                    quat=local_orientation.as_quat(scalar_first=True))
        if free and index == 0:
            node(body, "freejoint", name=f"{name}_free")
        else:
            node(body, "joint", name=f"{name}_flex_{index:02}", type="ball",
                 damping=.000055, stiffness=.037 if bunched_tip else .003, frictionloss=.000035,
                 armature=.00000002)
        for fold, (side, depth, width, tint) in enumerate((
            (-.0041,-.0006,.0042,(.43,.63,.81,1.)),
            ( .0000, .0011,.0050,(.38,.59,.79,1.)),
            ( .0040,-.0004,.0034,(.42,.62,.80,1.)),
        )):
            node(body, "geom", name=f"{name}_patch_{index:02}_{fold}", type="box",
                 pos=(side, depth, length/2), size=(width/2,.00245 if bunched_tip else .0016,length/2-.00012),
                 rgba=(*tint[:3], 0), mass=(.0017 * length / (SHEET_LENGTH*3)), margin=.0001,
                 friction=(.48, .004, .0002), solref=(.006, 1.))
        node(body, "site", name=f"{name}_node_{index:02}", pos=(0,0,0), size=.001,
             rgba=(0,0,0,0))
        if index == len(points)-2:
            node(body, "site", name=f"{name}_leader", pos=(0,0,length), size=.001,
                 rgba=(0,0,0,0))
        parent, parent_origin, parent_rotation = body, start, orientation
    return parent, length


def _skin(node, root, points, name, origin, rotation):
    """Render the folded cross-section continuously across the contact links."""
    sections = len(points) - 1
    circumference = 12
    contour = np.linspace(0, 2 * np.pi, circumference, endpoint=False)
    across = .0062 * np.cos(contour)
    depth = .00145 * np.sin(contour)
    orientations = [_axis_frame(start, end) for start, end in zip(points, points[1:])]
    positions = rotation.apply(points) + origin
    rings = []
    for index in range(sections + 1):
        frame = orientations[min(index, sections - 1)]
        # A folded tail retains several shallow ridges, with a fuller graspable end.
        fullness = 1 + .28 * (name == 'first_wipe' and index >= sections - 4)
        ridge = .00045 * np.sin(3 * np.pi * across / .0065)
        cross_section = np.column_stack((across, fullness * depth + ridge, np.zeros(circumference)))
        rings.append(positions[index] + (rotation * frame).apply(cross_section))
    faces = []
    for index in range(sections):
        base = index * circumference
        for vertex in range(circumference):
            line = base + vertex
            adjacent = base + (vertex + 1) % circumference
            faces.extend(((line, adjacent, adjacent + circumference),
                          (line, adjacent + circumference, line + circumference)))
    for vertex in range(1, circumference - 1):
        end = sections * circumference
        faces.extend(((0, vertex + 1, vertex), (end, end + vertex, end + vertex + 1)))
    skin = node(root.find('asset'), 'skin', name=f'{name}_surface',
                vertex=np.asarray(rings).ravel(), face=' '.join(str(i) for triangle in faces for i in triangle),
                rgba=(.44, .66, .83, 1.), group=0)
    for index, frame in enumerate(orientations):
        vertices = np.arange(index * circumference, (index + 2) * circumference)
        weights = np.r_[np.full(circumference, 1. if index == 0 else .5),
                        np.full(circumference, 1. if index == sections - 1 else .5)]
        node(skin, 'bone', body=f'{name}_{index:02d}', bindpos=positions[index],
             bindquat=(rotation * frame).as_quat(scalar_first=True), vertid=' '.join(str(i) for i in vertices), vertweight=weights)


def add(node, world, roll, root, origin, rotation):
    """Connect two physical sheets at a force-releasable native perforation."""
    points = _coil_points(SEGMENTS)
    next_root, next_length = _chain(node, roll, points[:SEGMENTS+1], "next_wipe", rotation, origin)
    _chain(node, world, points[SEGMENTS:], "first_wipe", rotation, origin, free=True)
    _skin(node, root, points[:SEGMENTS+1], "next_wipe", origin, rotation)
    _skin(node, root, points[SEGMENTS:], "first_wipe", origin, rotation)
    equality = root.find("equality")
    if equality is None:
        equality = node(root, "equality")
    node(equality, "connect", name=PERFORATION, body1=next_root.get("name"), body2="first_wipe_00",
         anchor=(0,0,next_length), solref=(.007,1.), solimp=(.95,.99,.001))
    node(root.find("contact"), "exclude", body1=next_root.get("name"), body2="first_wipe_00")


class TearablePerforation:
    """Separate the connect equality only under sustained native loading."""

    def __init__(self, model):
        self.equality_id = model.equality(PERFORATION).id
        self.body_id = model.body("first_wipe_00").id
        self.model = model
        self.overload_s = 0.
        self.tension_n = 0.
        self.break_time_s = None

    def observe_substep(self, data):
        if not data.eq_active[self.equality_id]:
            return
        active_rows = (data.efc_type == mj.mjtConstraint.mjCNSTR_EQUALITY) & (
            data.efc_id == self.equality_id
        )
        load = data.efc_force[active_rows]
        if len(load) != 3:
            self.overload_s = 0.
            self.tension_n = 0.
            return
        # Constraint force uses the endpoint displacement of body1 minus body2.
        # Positive projection along the detached sheet here resists tension.
        sheet_axis = data.xmat[self.body_id].reshape(3, 3)[:,2]
        self.tension_n = max(0., float(load @ sheet_axis))
        if self.tension_n < TEAR_TENSION_N:
            self.overload_s = 0.
            return
        self.overload_s += self.model.opt.timestep
        if self.overload_s >= TEAR_HOLD_S:
            data.eq_active[self.equality_id] = 0
            self.break_time_s = float(data.time)

class CreaseSet:
    """Leave a residual bend when loading exceeds the folded-paper yield angle."""

    def __init__(self, model):
        self.model = model

        self.joints = np.array([
            joint
            for joint in range(model.njnt)
            if "_wipe_flex_" in model.joint(joint).name
            and np.isclose(model.jnt_stiffness[joint], .003)
        ])
        self.indices = model.jnt_qposadr[self.joints, None] + np.arange(4)
        self.rest = model.qpos_spring[self.indices].copy()
        self.use = self.rest.copy()
        self.stiffness = model.jnt_stiffness[self.joints].copy()
        self.preview_stiffness = self.stiffness.copy()

    def reset_preview(self):
        self.use[:] = self.rest
        self.preview_stiffness[:] = self.stiffness

    def preview(self):
        self.model.qpos_spring[self.indices] = self.use
        self.model.jnt_stiffness[self.joints] = self.preview_stiffness

    def finish_preview(self, data):
        yielded = self._update(data, self.use)
        self.preview_stiffness[yielded] = .00008
        self.model.qpos_spring[self.indices] = self.rest
        self.model.jnt_stiffness[self.joints] = self.stiffness

    def observe(self, data):
        yielded = self._update(data, self.rest)
        self.stiffness[yielded] = .00008
        self.model.qpos_spring[self.indices] = self.rest
        self.model.jnt_stiffness[self.joints] = self.stiffness

    def _update(self, data, rest):
        current = data.qpos[self.indices]
        dot = np.einsum("ij,ij->i", current, rest)
        bend = 2 * np.arccos(np.clip(np.abs(dot), 0., 1.))
        plastic = bend > CREASE_YIELD_RAD
        actual = current[plastic] * np.sign(dot[plastic, None])
        previous = rest[plastic]
        remaining = CREASE_RESIDUAL_RAD / bend[plastic, None]
        updated = (1. - remaining) * actual + remaining * previous
        rest[plastic] = updated / np.linalg.norm(updated, axis=1, keepdims=True)
        return plastic
