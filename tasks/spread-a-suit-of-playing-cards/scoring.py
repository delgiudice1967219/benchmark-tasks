"""Score a released card ribbon using physical panel footprints and rank visibility."""

from dataclasses import dataclass
from itertools import product

import mujoco as mj
import numpy as np


CARD_COUNT = 13
CARD_AREA_M2 = .063 * .088
RELEASE_HOLD_S = 2.
CORNER_SIGNS = np.array(list(product((-1, 1), repeat=3)), dtype=float)
DOWN = np.array((0., 0., -1.))


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def convex_hull(points):
    points = sorted(set(map(tuple, points)))

    def half(sequence):
        hull = []
        for point in sequence:
            point = np.asarray(point)
            while len(hull) > 1 and cross(hull[-1] - hull[-2], point - hull[-1]) <= 1e-12:
                hull.pop()
            hull.append(point)
        return hull

    return half(points)[:-1] + half(reversed(points))[:-1]


def overlap_area(first, second):
    """Clip an ordered convex footprint against a second counterclockwise footprint."""
    overlap = list(first)
    for start, end in zip(second, second[1:] + second[:1]):
        edge = end - start
        previous = overlap.pop()
        previous_side = cross(edge, previous - start)
        clipped = []
        for current in overlap + [previous]:
            current_side = cross(edge, current - start)
            if (previous_side >= 0) != (current_side >= 0):
                ratio = previous_side / (previous_side - current_side)
                clipped.append(previous + ratio * (current - previous))
            if current_side >= 0:
                clipped.append(current)
            previous = current
            previous_side = current_side
        overlap = clipped
        if len(overlap) < 3:
            return 0.
    return abs(sum(cross(a, b) for a, b in zip(overlap, overlap[1:] + overlap[:1]))) / 2


@dataclass(frozen=True)
class Ribbon:
    cards_on_felt: bool
    rank_readable: tuple[bool, ...]
    ordered_links: tuple[bool, ...]
    overlap_fraction: tuple[float, ...]
    longest_readable_run: int


class RibbonGeometry:
    """Inspect face-up rank patches and overlapping cards from above."""

    def __init__(self, model, felt_z, felt_bounds, spread_axis=(1., 0.)):
        self.model = model
        self.felt_z = felt_z
        self.felt_low, self.felt_high = np.asarray(felt_bounds, dtype=float)
        self.axis = np.asarray(spread_axis, dtype=float)
        self.axis /= np.linalg.norm(self.axis)
        names = [model.geom(geom).name for geom in range(model.ngeom)]
        self.panels = [
            [geom for geom, name in enumerate(names) if name.startswith(f"card_{rank:02d}_panel_")]
            for rank in range(CARD_COUNT)
        ]
        if not all(self.panels):
            raise ValueError("Every card needs a named physical face panel")
        self.card_by_geom = np.full(model.ngeom, -1, dtype=int)
        for rank, selected in enumerate(self.panels):
            ink = [geom for geom, name in enumerate(names)
                   if name.startswith((f"card_{rank:02d}_rank_", f"rank_{rank:02d}_", f"ink_{rank:02d}_"))]
            self.card_by_geom[selected + ink] = rank
        self.corners = [[model.site(f"rank_{rank:02d}_{corner}").id for corner in range(4)]
                        for rank in range(CARD_COUNT)]
        self.normals = [model.site(f"rank_{rank:02d}_normal").id for rank in range(CARD_COUNT)]
        self.hit = np.empty(1, dtype=np.int32)
        self.clear_rise = np.array((0., 0., .20))

    def physical_corners(self, data, selected):
        return np.vstack([
            data.geom_xpos[geom] + (CORNER_SIGNS * self.model.geom_size[geom])
            @ data.geom_xmat[geom].reshape(3, 3).T
            for geom in selected
        ])

    def readable_rank(self, data, rank, corners):
        face = data.site_xmat[self.normals[rank]].reshape(3, 3)
        if face[2, 2] < .90:
            return False
        center = corners.mean(axis=0)
        # Test the middle and four inset corners. Edge contacts may hit the
        # neighboring panel seam without covering any of the printed rank.
        points = np.vstack((center, center + .92 * (corners - center)))
        for sample in points:
            depth = mj.mj_ray(self.model, data, sample + self.clear_rise, DOWN, None, True, -1, self.hit)
            hit = int(self.hit[0])
            if depth < 0 or hit < 0 or int(self.card_by_geom[hit]) != rank:
                return False
        return True

    def assess(self, data):
        shells = [self.physical_corners(data, selected) for selected in self.panels]
        table_margin = .002
        cards_on_felt = all(
            np.all(shell[:, :2] >= self.felt_low - table_margin)
            and np.all(shell[:, :2] <= self.felt_high + table_margin)
            and shell[:, 2].min() >= self.felt_z - table_margin
            and shell[:, 2].max() <= self.felt_z + .026
            for shell in shells
        )
        rank_patches = data.site_xpos[self.corners]
        projected = rank_patches.mean(axis=1)[:, :2] @ self.axis
        visible = tuple(self.readable_rank(data, rank, corners)
                        for rank, corners in enumerate(rank_patches))
        hulls = [convex_hull(shell[:, :2]) for shell in shells]
        overlap = tuple(overlap_area(left, right) / CARD_AREA_M2
                        for left, right in zip(hulls, hulls[1:]))
        links = tuple(visible[index] and visible[index + 1] and overlap[index] >= .10
                      and projected[index + 1] - projected[index] >= .004
                      for index in range(CARD_COUNT - 1))
        longest, running = int(any(visible)), int(visible[0])
        for index, linked in enumerate(links):
            running = running + 1 if linked else int(visible[index + 1])
            longest = max(longest, running)
        return Ribbon(bool(cards_on_felt), visible, links, overlap, longest)


class RibbonScore:
    """Combine the result with continuous-sweep video and released dwell checks."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.clear_seconds = 0.
        self.damage_seen = False

    def update(self, data, seconds, *, sweep_verified, released, damaged=False):
        if seconds <= 0:
            raise ValueError("Use a positive elapsed duration")
        result = self.geometry.assess(data)
        self.damage_seen |= damaged
        if self.damage_seen or not result.cards_on_felt or not sweep_verified:
            self.clear_seconds = 0.
            return 0., result
        full_layout = result.longest_readable_run == CARD_COUNT
        self.clear_seconds = self.clear_seconds + seconds if released and full_layout else 0.
        if self.clear_seconds + 1e-9 >= RELEASE_HOLD_S:
            return 1., result
        if result.longest_readable_run >= 8:
            return .50, result
        if result.longest_readable_run >= 3:
            return .25, result
        return 0., result
