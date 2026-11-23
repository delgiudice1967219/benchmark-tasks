"""Native stock-table geometry for a portable centre-pull towel dispenser."""

from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from dream_examples.yam_mujoco import yam_env
from paper import add as add_paper


TABLE = .760
ROLL = np.array((-.136, -.140, TABLE + .001))
CASE = np.array((.104, -.144, TABLE + .001))
ROLL_RADIUS = .075
ROLL_HEIGHT = .183
CORE_RADIUS = .030
CORE_THICKNESS = .00115
CASE_INNER = .0788
CASE_OUTER = .087
CASE_BASE = .004
CASE_HEIGHT = .195
LID_HEIGHT = .018
NOZZLE_INNER = .015
NOZZLE_INLET_RADIUS = NOZZLE_INNER + .006
NOZZLE_TAPER_HEIGHT = .014
SIDE_LID_OPEN = 2.76
CASE_RESET_Y = -.593
ROLL_BLUE = (.36, .59, .80, 1.)
CORE_BROWN = (.66, .47, .32, 1.)
POLYMER = (.15, .19, .23, 1.)
RUBBER = (.64, .18, .15, 1.)
CAMERAS = {name: name for name in ("top_cam", "left_cam", "right_cam", "judging_front")}


def node(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {
        key: yam_env.numbers(val) if isinstance(val, (list, tuple, np.ndarray)) else str(val)
        for key, val in attrs.items()
    })


def box(parent, name, size, position, color, **attrs):
    return node(parent, "geom", name=name, type="box", size=np.asarray(size)/2,
                pos=position, rgba=color, **attrs)


def ring_slice(parent, assets, prefix, inner, outer, height, z, color, *,
               count=24, start=0, span=360, center=(0., 0.), inner_high=None, **attrs):
    """Convex annular sectors preserve a real open aperture for contacts."""
    angles = np.linspace(np.deg2rad(start), np.deg2rad(start+span), count+1)
    for index, (one, two) in enumerate(zip(angles, angles[1:])):
        points = []
        for radius in (inner,outer):
            for theta in (one,two):
                for dz in (-height/2,height/2):
                    r = inner_high if radius == inner and dz > 0 and inner_high is not None else radius
                    points.append((center[0]+r*np.cos(theta),center[1]+r*np.sin(theta),z+dz))
        name = f"{prefix}_{index:02d}"
        node(assets, "mesh", name=name, vertex=np.asarray(points).ravel())
        node(parent, "geom", name=name, type="mesh", mesh=name, rgba=color, **attrs)


def build(*, on_side=False, with_paper=False, case_on_side=False, case_x=.028, case_y=-.499):
    root = ET.fromstring(yam_env.build_xml())
    root.set("model", "Reload a portable centre-pull towel dispenser")
    root.find("option").attrib.update(timestep=".001", iterations="60", cone="elliptic")
    world = root.find("worldbody")
    assets = root.find("asset")
    for child in tuple(world):
        if child.get("name") in ("apple", "banana", "orange", "water_bottle", "bowl", "plate"):
            world.remove(child)
    for asset in tuple(assets):
        if asset.get("name", "").startswith(("apple_", "banana_", "bowl_", "brown_glaze", "blue_glaze")):
            assets.remove(asset)
    for light in world.findall("light"):
        light.set("castshadow", "false")

    world.remove(world.find("camera[@name='top']"))
    top = yam_env.calibrated_camera(world, "top_cam", yam_env.CALIBRATION["cameras"]["top"], world=True)
    for side in ("left", "right"):
        world.find(f".//camera[@name='{side}']").set("name", f"{side}_cam")
    stand = world.find("body[@name='center_camera_stand']")
    top_pos = np.fromstring(top.get("pos"), sep=" ")
    foot, mast, beam = stand.findall("geom")
    mast_y = top_pos[1] + .07
    foot.set("pos", yam_env.numbers((top_pos[0], mast_y, TABLE+.008)))
    mast.set("fromto", yam_env.numbers((top_pos[0], mast_y, TABLE+.016,
                                       top_pos[0], mast_y, top_pos[2])))
    beam.set("fromto", yam_env.numbers((top_pos[0], mast_y, top_pos[2], *top_pos + [0, .02, .02])))
    stand.find("body").attrib.update(pos=top.get("pos"), quat=top.get("quat"))
    if case_on_side:
        # Show the open mouth, the lid nozzle, and the area where the core is set down.
        yam_env.camera(world, "judging_front", (.63, -.04, 1.47), (-.07, -.28, .80))
        world.find("camera[@name='judging_front']").set("fovy", "38")
    else:
        yam_env.camera(world, "judging_front", (.535, -.685, 1.350), (-.025, -.255, .903))
        world.find("camera[@name='judging_front']").set("fovy", "37")

    roll_origin = np.array((-.160, -.400, TABLE+ROLL_RADIUS+.001)) if on_side else ROLL
    half_sqrt = np.sqrt(.5)
    roll_rotation = (half_sqrt, -half_sqrt, 0, 0) if on_side else (1, 0, 0, 0)
    roll = node(world, "body", name="roll", pos=roll_origin, quat=roll_rotation)
    node(roll, "freejoint", name="roll_free")
    ring_slice(roll, assets, "roll", ROLL_RADIUS*.416, ROLL_RADIUS,
               ROLL_HEIGHT, ROLL_HEIGHT/2, ROLL_BLUE, count=24, mass=.72/24,
               friction=(.48, .008, .0004), solref=(.008, 1.))
    if with_paper:
        add_paper(node, world, roll, root, roll_origin,
                  Rotation.from_quat(roll_rotation, scalar_first=True))
    # The open lap and red perforation identify the inside edge of the removable sleeve.
    core_offset = [0, .005, 0] if on_side else [0, 0, .005]
    core = node(world, "body", name="core", pos=roll_origin+core_offset, quat=roll_rotation)
    node(core, "freejoint", name="core_free")
    parent, origin = core, np.zeros(2)
    for i in range(5):
        start, span = 19+i*67.4, 67.4
        if i:
            theta = np.deg2rad(start)
            join = (CORE_RADIUS-CORE_THICKNESS/2)*np.array((np.cos(theta), np.sin(theta)))
            parent = node(parent, "body", name=f"core_panel_{i}", pos=(*tuple(join-origin),0))
            node(parent, "joint", name=f"core_fold_{i}", type="hinge", axis=(0,0,1),
                 range=(-.16,.60), stiffness=.015, damping=.0018, frictionloss=.0005,
                 armature=.000001)
            origin = join
        ring_slice(parent, assets, f"core_panel_{i}", CORE_RADIUS-CORE_THICKNESS, CORE_RADIUS,
                   .177, .177/2, CORE_BROWN, count=5, start=start, span=span,
                   center=-origin, mass=.021/25, friction=(.30,.005,.0004), solref=(.008,1.))
    box(core, "core_peel_edge", (.025, .0018, .052), (.0175, .002, .154),
        CORE_BROWN, mass=.001, friction=(.54, .004, .0004))
    box(core, "red_core_mark", (.0018, .0019, .030), (.021, .002, .158),
        (.71, .13, .08, 1.), mass=0., contype=0, conaffinity=0)
    node(core, "site", name="core_grip", pos=(.020, .002, .168), size=.001, rgba=(0,0,0,0))

    case_origin = (case_x,case_y,TABLE+.089) if case_on_side else CASE
    case_quat = (-.5,.5,.5,.5) if case_on_side else (1,0,0,0)
    dispenser = node(world, "body", name="dispenser", pos=case_origin, quat=case_quat)
    node(dispenser, "freejoint", name="dispenser_free")
    node(dispenser, "geom", name="base", type="cylinder", size=(CASE_OUTER, CASE_BASE/2),
         pos=(0,0,CASE_BASE/2), rgba=POLYMER, mass=.33, friction=(.78,.012,.001))
    ring_slice(dispenser, assets, "case", CASE_INNER, CASE_OUTER, CASE_HEIGHT-CASE_BASE,
               (CASE_HEIGHT+CASE_BASE)/2, POLYMER, count=32, mass=.38/32,
               friction=(.45,.005,.0004))
    if case_on_side:
        # Flat exterior faces of the housing support the portable case on a bench.
        for index, y in enumerate((-.037, .037)):
            box(dispenser, f"case_bench_contact_{index}", (.009,.019,.156),
                (-.083,y,.088), POLYMER, mass=.024, friction=(.67,.009,.0006))
    lid = node(dispenser, "body", name="lid", pos=(0,.084,CASE_HEIGHT+.001))
    node(lid, "joint", name="lid_hinge", type="hinge", axis=(-1,0,0), range=(-.005,2.83),
         damping=.055, frictionloss=.18 if case_on_side else .05, armature=.002)
    ring_slice(lid, assets, "cover", NOZZLE_INNER+.010, CASE_OUTER, LID_HEIGHT, LID_HEIGHT/2,
               POLYMER, center=(0.,-.084), count=24, mass=.10/24)
    ring_slice(lid, assets, "nozzle_inlet", NOZZLE_INLET_RADIUS, NOZZLE_INNER+.010, NOZZLE_TAPER_HEIGHT,
               NOZZLE_TAPER_HEIGHT/2, RUBBER, center=(0.,-.084), inner_high=NOZZLE_INNER,
               count=20, mass=.015/20, friction=(.58,.005,.0007))
    ring_slice(lid, assets, "nozzle_outlet", NOZZLE_INNER, NOZZLE_INNER+.010, LID_HEIGHT-NOZZLE_TAPER_HEIGHT,
               (LID_HEIGHT+NOZZLE_TAPER_HEIGHT)/2, RUBBER, center=(0.,-.084), count=20, mass=.005/20,
               friction=(.58,.005,.0007))
    box(lid, "lid_lip", (.042,.018,.012), (0,-.177,.011),
        POLYMER, mass=.010)
    node(lid, "site", name="lip_grip", size=.001, pos=(0,-.173,.011), rgba=(0,0,0,0))
    node(lid, "site", name="nozzle_inside", size=.001, pos=(0,-.084,-.003), rgba=(0,0,0,0))
    node(lid, "site", name="nozzle_outside", size=.001, pos=(0,-.084,LID_HEIGHT+.003), rgba=(0,0,0,0))
    box(dispenser, "case_latch_mount", (.031,.032,.017),(0,-.097,.162),POLYMER,mass=.010)
    latch = node(dispenser, "body", name="latch", pos=(0,-.110,.169))
    node(latch, "joint", name="latch_deflection", type="hinge", axis=(1,0,0), range=(-.010,.32),
         stiffness=.25, damping=.018, frictionloss=.003, armature=.000073)
    box(latch, "latch_lever", (.027,.004,.047), (0,-.002,.0235),
        POLYMER, mass=.006, friction=(.31,.005,.0003))
    box(latch, "latch_crown", (.021,.008,.004), (0,.001,.047),
        POLYMER, mass=.002, friction=(.09,.003,.0002))
    node(latch, "geom", name="latch_pawl", type="capsule", size=(.0065,),
         fromto=(-.009,.003,.051,.009,.003,.051), rgba=POLYMER,
         mass=.002, friction=(.09,.003,.0002), solref=(.007,1.))
    node(root.find("contact"), "exclude", body1="dispenser", body2="lid")
    return ET.tostring(root, encoding="unicode")
