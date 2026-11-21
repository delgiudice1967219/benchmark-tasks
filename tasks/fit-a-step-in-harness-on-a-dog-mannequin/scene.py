"""Calibrated tabletop station and physically articulated small-dog props."""

from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from dream_examples.yam_mujoco import yam_env


TABLE_Z = .760
DT = .0005
HOME = np.array([0., 1., 1., -.6, 0., 0.])
DOG_ORIGIN = np.array([0., -.045, .913])
HARNESS_ORIGIN = np.array([0., -.234, TABLE_Z + .004])
CAMERAS = dict(top_cam="top", left_cam="left", right_cam="right", judging_front="harness_detail")
SIDES = ("left", "right")
CHOCOLATE = ".43 .25 .14 1"
TAN = ".68 .43 .22 1"
BLACK = ".12 .12 .11 1"
TEAL = ".12 .45 .59 1"
RED = ".72 .22 .17 1"


def vec(values):
    return " ".join(f"{x:.10g}" for x in values)


def geom(parent, name, shape, size, *, at=(0., 0., 0.), mass=.0,
         color=TAN, contact=True, **kwargs):
    params = dict(name=name, type=shape, pos=vec(at), size=vec(size), mass=str(mass),
                  rgba=color, friction=".66 .002 .001", solref=".004 1",
                  solimp=".95 .995 .0006")
    if not contact:
        params.update(contype="0", conaffinity="0")
    params.update(kwargs)
    return ET.SubElement(parent, "geom", params)


def add_dog(world):
    dog = ET.SubElement(world, "body", name="mannequin", pos=vec(DOG_ORIGIN))
    ET.SubElement(dog, "freejoint", name="dog_free")
    geom(dog, "chest", "ellipsoid", (.040, .086, .049), at=(0, -.026, 0), mass=.28)
    geom(dog, "haunches", "ellipsoid", (.035, .062, .043), at=(0, .070, -.003),
         mass=.16, color=CHOCOLATE)
    geom(dog, "neck", "capsule", (.028,), mass=.045,
         fromto=vec((0, -.080, .018, 0, -.117, .047)))
    geom(dog, "head", "ellipsoid", (.038, .042, .035), at=(0, -.144, .054), mass=.055)
    geom(dog, "muzzle", "ellipsoid", (.024, .032, .016), at=(0, -.181, .040),
         mass=.019, color=".86 .58 .32 1")
    geom(dog, "nose", "ellipsoid", (.017, .009, .011), at=(0, -.206, .043), color=BLACK, contact=False)
    for hand, sign in zip(SIDES, (1, -1)):
        geom(dog, f"{hand}_ear", "ellipsoid", (.014, .022, .033),
             at=(sign * .032, -.133, .053), mass=.005, color=CHOCOLATE,
             euler=vec((0., sign * .45, 0.)))
        geom(dog, f"{hand}_eye", "sphere", (.0042,), at=(sign * .028, -.169, .066),
             color=BLACK, contact=False)
        add_leg(dog, hand, sign, forward=True)
        add_leg(dog, hand, sign, forward=False)
    geom(dog, "tail", "capsule", (.012,), mass=.012,
         fromto=vec((0, .117, .007, 0, .186, .051)), color=CHOCOLATE)


def add_leg(dog, side, sign, *, forward):
    label = f"{side}_{'fore' if forward else 'hind'}"
    y = -.085 if forward else .096
    x = .032 if forward else .029
    thigh = ET.SubElement(dog, "body", name=f"{label}_thigh", pos=vec((sign * x, y, -.034)))
    ET.SubElement(thigh, "joint", name=f"{label}_shoulder", type="hinge", axis="1 0 0",
                  range="-.42 .68", stiffness=".95", damping=".038", armature=".00003")
    geom(thigh, f"{label}_pad_upper", "capsule", (.013,), mass=.022,
         fromto="0 0 .001 0 0 -.047")
    lower = ET.SubElement(thigh, "body", name=f"{label}_lower", pos="0 0 -.048")
    ET.SubElement(lower, "joint", name=f"{label}_ankle", type="hinge", axis="1 0 0",
                  range="-.41 .56", stiffness=".65", damping=".031", armature=".00002")
    geom(lower, f"{label}_pad_lower", "capsule", (.010,), mass=.016,
         fromto="0 0 .001 0 0 -.057")
    geom(lower, f"{label}_paw", "ellipsoid", (.015, .020, .010),
         at=(0, -.004, -.061), mass=.011, color=".83 .57 .36 1")
    ET.SubElement(lower, "site", name=f"{label}_sole", pos="0 -.004 -.071", size=".002", rgba="0 0 0 0")


def buckle(world, side, point, *, with_latch=False, inward=False, dual_adjust=False):
    body = ET.SubElement(world, "body", name=f"{side}_buckle", pos=vec(point))
    ET.SubElement(body, "freejoint", name=f"{side}_buckle_free")
    if not with_latch:
        geom(body, f"{side}_buckle_shell", "box", (.014, .012, .004), mass=.014, color=BLACK)
        ET.SubElement(body, "site", name=f"{side}_buckle_end", size=".002", rgba="0 0 0 0")
        return

    if (side == "left") == inward:
        body.set("quat", "0 0 0 1")
    if side == "left":
        female_buckle(body, side)
    else:
        male_buckle(body, side)
    if dual_adjust:
        clip_geom(body, side, "outer_bar", (-.0189, 0, 0), (.0016, .0125, .004), .002)
        for sign in (-1, 1):
            clip_geom(body, side, f"outer_rail_{sign}", (-.0138, sign * .011, 0), (.0067, .0015, .004), .001)
    ET.SubElement(body, "site", name=f"{side}_buckle_end", pos=vec((-.019 if dual_adjust else -.014, 0, 0)),
                  size=".002", rgba="0 0 0 0")
    ET.SubElement(body, "site", name=f"{side}_buckle_grasp", pos=vec((-.013 if dual_adjust else -.008, 0, 0)),
                  size=".002", rgba="0 0 0 0")


def clip_geom(parent, side, label, at, size, mass=.001):
    geom(parent, f"{side}_buckle_{label}_shell", "box", size, at=at, mass=mass,
         color=BLACK, priority="1", friction=".25 .002 .001", solref=".003 1")


def female_buckle(body, side):
    for sign in (-1, 1):
        clip_geom(body, side, f"face_{sign}", (0, 0, sign * .0038), (.015, .0118, .00085))
        clip_geom(body, side, f"rear_{sign}", (-.008, sign * .0104, 0), (.007, .0014, .0034))
        clip_geom(body, side, f"catch_{sign}", (.0122, sign * .0102, 0), (.0028, .0017, .0034))
    clip_geom(body, side, "strap_end", (-.013, 0, 0), (.002, .0118, .0043), .003)


def male_buckle(body, side):
    clip_geom(body, side, "base", (-.007, 0, 0), (.008, .0118, .0045), .008)
    # A central guide extends between the two release tabs. It holds the clip
    # faces parallel while it is loaded, as in a moulded side-release buckle.
    geom(body, f"{side}_buckle_guide_shell", "capsule", (.00235,), mass=.0006, color=BLACK,
         priority="1", friction=".25 .002 .001", solref=".003 1", fromto=".001 0 0 .024 0 0")
    for sign in (-1, 1):
        name = f"{side}_buckle_tine_{sign}"
        tine = ET.SubElement(body, "body", name=name, pos=vec((.0005, sign * .0055, 0)))
        ET.SubElement(tine, "joint", name=f"{name}_flex", type="hinge", axis="0 0 1",
                      stiffness=".085", damping=".00032", armature=".00000006",
                      range=vec((-.36, .045) if sign > 0 else (-.045, .36)))
        clip_geom(tine, side, f"tine_{sign}", (.0098, 0, 0), (.0103, .00115, .0019), .0006)
        geom(tine, f"{name}_tooth_shell", "mesh", (), mass=.0002, color=BLACK, priority="1",
             friction=".25 .002 .001", solref=".002 1", mesh=f"buckle_tooth_{sign}")
        ET.SubElement(tine, "site", name=f"{name}_catch", pos=vec((.0123, sign * .0046, 0)),
                      size=".001", rgba="0 0 0 0")


def add_tooth_assets(asset):
    for sign in (-1, 1):
        shape = ((.0120, sign * .0009), (.0120, sign * .0053),
                 (.0200, sign * .0012), (.0200, sign * .0009))
        vertices = [item for height in (-.0018, .0018) for x, y in shape for item in (x, y, height)]
        ET.SubElement(asset, "mesh", name=f"buckle_tooth_{sign}", vertex=vec(vertices))


def add_harness(world, equality, contacts, *, with_latch=False, inward=False, dual_adjust=False):
    origin = HARNESS_ORIGIN
    bridge = ET.SubElement(world, "body", name="harness_bridge", pos=vec(origin))
    ET.SubElement(bridge, "freejoint", name="bridge_free")
    geom(bridge, "bridge_web", "box", (.0065, .040, .0017), mass=.006, color=TEAL,
         priority="1", solref=".002 1", solimp=".96 .995 .00025")
    for offset in (-1, 1):
        ET.SubElement(bridge, "site", name=f"bridge_fork_{offset}", pos=vec((0, offset * .039, 0)),
                      size=".001", rgba="0 0 0 0")
    for side, sign in zip(SIDES, (1, -1)):
        span = .145 if inward else .130 if with_latch else .113
        end = origin + (sign * span, 0, 0)
        strap_offset = .019 if dual_adjust else .014
        buckle_offset = (sign * (-strap_offset if inward else strap_offset), 0, .001) if with_latch else (0, 0, 0)
        buckle(world, side, end + buckle_offset, with_latch=with_latch, inward=inward, dual_adjust=dual_adjust)
        for offset in (-1, 1):
            start = origin + (0, offset * .039, 0)
            # A deliberately loose strap bows gently away from the centerline.
            # This is ordinary lay-flat slack rather than a rigid positioning aid.
            bow = np.array((sign * .035 if inward else 0, offset * .013, 0.))
            shape = np.array([start + weight * (end - start) + bow * np.sin(weight * np.pi)
                              for weight in np.linspace(0, 1, 6)])
            label = f"{side}_{'front' if offset < 0 else 'rear'}"
            name = add_web_chain(bridge, contacts, label, shape, origin)
            ET.SubElement(equality, "connect", name=f"{label}_stitched_seam",
                          site1=name, site2=f"{side}_buckle_end", solref=".003 1",
                          solimp=".95 .995 .001")
            ET.SubElement(contacts, "exclude", body1=f"{label}_seg_4", body2=f"{side}_buckle")
        ET.SubElement(contacts, "exclude", body1=f"{side}_front_seg_4", body2=f"{side}_rear_seg_4")
    for edge in ("front", "rear"):
        ET.SubElement(contacts, "exclude", body1=f"left_{edge}_seg_0", body2=f"right_{edge}_seg_0")


def add_web_chain(bridge, contacts, label, points, origin):
    parent, translated = bridge, origin
    for index, (first, second) in enumerate(zip(points, points[1:])):
        direction = second - first
        length = float(np.linalg.norm(direction))
        local_center = direction / 2
        angle = float(np.arctan2(direction[1], direction[0]))
        name = f"{label}_seg_{index}"
        joint = ET.SubElement(parent, "body", name=name, pos=vec(first - translated))
        ET.SubElement(joint, "joint", name=f"{name}_bend", type="ball",
                      stiffness=".00027", damping=".00006", armature=".000001")
        geom(joint, f"{name}_web", "box", (length / 2 + .0003, .0063, .0016),
             at=local_center, mass=.0018, color=RED, priority="1",
             solref=".002 1", solimp=".96 .995 .00025", euler=vec((0, 0, angle)))
        if index > 0:
            ET.SubElement(contacts, "exclude", body1=name, body2=parent.get("name"))
        ET.SubElement(joint, "site", name=f"{name}_origin", size=".0011", rgba="0 0 0 0")
        translated, parent = first, joint
    end_name = f"{label}_seam"
    ET.SubElement(parent, "site", name=end_name, pos=vec(points[-1] - translated),
                  size=".0011", rgba="0 0 0 0")
    return end_name


def model_for_trial(*, with_harness=True, with_latch=False, inward=False, dual_adjust=False):
    scene = ET.fromstring(yam_env.build_xml())
    scene.find("size").set("memory", "320M")
    scene.find("option").attrib.update(timestep=str(DT), noslip_iterations="0",
                                       iterations="65", impratio="3")
    world = scene.find("worldbody")
    for part in tuple(world.findall("body")):
        if part.get("name") in {"apple", "banana", "orange", "water_bottle", "bowl", "plate"}:
            world.remove(part)
    for light in world.findall("light"):
        light.set("castshadow", "false")
    yam_env.camera(world, "harness_detail", (.27, -.62, 1.35), (0., -.16, .87))
    world.find('camera[@name="harness_detail"]').set("fovy", "39")
    add_dog(world)
    if with_harness:
        equality = ET.SubElement(scene, "equality")
        if with_latch:
            add_tooth_assets(scene.find("asset"))
        add_harness(world, equality, scene.find("contact"), with_latch=with_latch, inward=inward,
                    dual_adjust=dual_adjust)
    for actuator in scene.find("actuator"):
        name = actuator.get("name", "")
        if "_joint" in name:
            limit = 7 if name[-1] in "456" else 27
            actuator.set("forcerange", f"{-limit} {limit}")
    return mj.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))


def reset(model):
    data = mj.MjData(model)
    for side in SIDES:
        for index, value in enumerate(HOME, 1):
            data.joint(f"{side}_joint{index}").qpos[:] = value
        for sign in (-1, 1):
            data.joint(f"{side}_finger_{sign}").qpos[:] = .0475
    data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    mj.mj_forward(model, data)
    return data


def stance(side, *, angle=0.):
    sign = 1. if side == "left" else -1.
    axis = np.array((sign, 0., 0.))
    toward = Rotation.from_euler("x", angle, degrees=True).apply((0., 0., -1.))
    return np.column_stack((np.cross(axis, toward), axis, toward))
