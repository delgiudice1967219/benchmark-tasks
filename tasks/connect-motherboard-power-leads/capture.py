"""Read the native snap-latch's upper nose face against its fixed undercut."""

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from scene import latch_centers


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def intersect_convex(subject, outline):
    polygon = list(subject)
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        if not polygon:
            break
        edge = b - a
        previous = polygon[-1]
        past = cross2(edge, previous - a)
        output = []
        for current in polygon:
            now = cross2(edge, current - a)
            if (past >= 0.) != (now >= 0.):
                output.append(previous + (current - previous) * past / (past - now))
            if now >= 0.:
                output.append(current)
            previous, past = current, now
        polygon = output
    return np.asarray(polygon)


def face(model, data, geom, upper):
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start : start + count]
    body = int(model.geom_bodyid[geom])
    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    local_z = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)[:, 2]
    extrema = np.max(local_z) if upper else np.min(local_z)
    return world[np.abs(local_z - extrema) < 1e-7]


def read_latches(model, data, spec):
    header = data.body(spec.name + "_header")
    frame = header.xmat.reshape(3, 3)
    results = []
    for lane, _ in enumerate(latch_centers(spec)):
        catch = model.geom(f"{spec.name}_ledge_{lane}").id
        nose = model.geom(f"{spec.name}_clip_nose_{lane}").id
        lower = (face(model, data, catch, upper=False) - header.xpos) @ frame
        upper = (face(model, data, nose, upper=True) - header.xpos) @ frame
        try:
            cut = lower[ConvexHull(lower[:, :2]).vertices, :2]
            hook = upper[ConvexHull(upper[:, :2]).vertices, :2]
        except QhullError:
            results.append(dict(area_mm2=0., y_overlap_mm=0., x_overlap_mm=0., axial_clearance_mm=None))
            continue
        overlap = intersect_convex(hook, cut)
        if len(overlap) < 3:
            results.append(dict(area_mm2=0., y_overlap_mm=0., x_overlap_mm=0., axial_clearance_mm=None))
            continue
        coefficients = np.linalg.lstsq(np.c_[upper[:, :2], np.ones(len(upper))], upper[:, 2], rcond=None)[0]
        maximum_upper_z = float(np.max(np.c_[overlap, np.ones(len(overlap))] @ coefficients))
        area = abs(sum(cross2(a, b) for a, b in zip(overlap, np.roll(overlap, -1, axis=0)))) / 2
        width = np.ptp(overlap, axis=0)
        results.append(dict(area_mm2=float(area * 1e6), x_overlap_mm=float(width[0] * 1e3),
                            y_overlap_mm=float(width[1] * 1e3),
                            axial_clearance_mm=float((np.mean(lower[:, 2]) - maximum_upper_z) * 1e3)))
    return results
