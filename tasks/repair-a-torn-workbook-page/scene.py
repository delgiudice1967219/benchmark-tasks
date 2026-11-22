"""A partly torn A5 workbook page and supported repair tissue in a stock YAM scene."""

from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

from dream_examples.yam_mujoco import yam_env


HZ = 20
DT = .0005
HOME = np.array((0., 1., 1., -.6, 0., 0.))
ARMS = ("left", "right")
TABLE_Z = .760
BOOK_RISE = .016
PAPER_Z = .797 + BOOK_RISE
BOOK_X = .025
SPINE_Y = -.218
BOOK_WIDTH = .210
PAGE_DEPTH = .148
FREE_EDGE_OVERHANG = .004
TEAR_START = .082
FLAP_LENGTH = PAGE_DEPTH - TEAR_START
PAPER_HALF = .000069
TAPE_X = -.212
TAPE_Y = -.134
TAPE_Z = .79718
TAPE_HALF = .010
TAPE_STEP = .0185
TAPE_COUNT = 4
TAPE_THICKNESS = .000055
LINER_THICKNESS = .000054
TAPE_TIP = TAPE_Y + TAPE_STEP * TAPE_COUNT
CAMERAS = {"top_cam": "top", "left_cam": "left", "right_cam": "right",
           "judging_front": "repair_detail"}


def num(values):
    return " ".join(f"{value:.10g}" for value in values)


def child(parent, tag, name=None, **kwargs):
    if name is not None:
        kwargs.update(name=name)
    return ET.SubElement(parent, tag, **kwargs)


def box(parent, name, centre, half, rgb, **kwargs):
    return child(parent, "geom", name, type="box", pos=num(centre), size=num(half),
                 rgba=rgb, **kwargs)


def paper_box(parent, name, centre, half, mass, color=".87 .86 .79 1"):
    return box(parent, name, centre, half, color, mass=str(mass), friction=".37 .001 .0001",
               solref=".0014 1", solimp=".92 .99 .00018", condim="3")


def ink(parent, x, y, width, height, color=".18 .26 .37 1", reverse=False):
    z = (PAPER_HALF + .000012) * (-1 if reverse else 1)
    box(parent, None, (x, y, z), (width / 2, height / 2, .000004), color,
        contype="0", conaffinity="0", mass="0")


def add_workbook(world, exclusions):
    depth = PAGE_DEPTH - FREE_EDGE_OVERHANG
    for direction, side in ((-1, "back"), (1, "front")):
        y = SPINE_Y + direction * depth / 2
        box(world, "workbook_cover_" + side, (BOOK_X, y, .779 + BOOK_RISE),
            (.107, depth / 2, .003), ".13 .24 .36 1")
        box(world, "workbook_stack_" + side, (BOOK_X, y, .7894 + BOOK_RISE),
            (.104, depth / 2, .0074), ".90 .90 .83 1", friction=".51 .001 .0001")
    box(world, "book_support", (BOOK_X, SPINE_Y, .7675 + BOOK_RISE / 2),
        (.111, .142, .0075 + BOOK_RISE / 2), ".33 .34 .31 1")
    # Add the intact bound portion. The two loose flaps can move relative to it at the tear origin.
    page = child(world, "body", "bound_page", pos=num((BOOK_X, SPINE_Y, PAPER_Z)))
    child(page, "joint", "turn_page", type="hinge", axis="1 0 0", limited="true",
          range=num((-.035, np.pi + .035)), damping=".0005", armature=".000003")
    paper_box(page, "bound_page_paper", (0, TEAR_START / 2, 0),
              (BOOK_WIDTH / 2, TEAR_START / 2, PAPER_HALF), .0019)
    ink(page, -.051, .067, .066, .0068, ".20 .38 .56 1")
    for y in (.050, .045, .040, .031, .026):
        for x, width in ((-.059, .050), (.021, .090), (.076, .012)):
            ink(page, x, y, width, .00065, ".33 .41 .44 1")
    ink(page, 0, .014, .179, .0010, ".32 .47 .52 1")
    ink(page, 0, .014, .179, .0010, ".32 .47 .52 1", reverse=True)
    tear_offsets = (0, .00055, -.001, .00028, .00145, -.0012, .0007, .00020)
    spacing = FLAP_LENGTH / len(tear_offsets)
    for sign, side in ((1, "left"), (-1, "right")):
        flap = child(page, "body", f"flap_{side}", pos=num((0, TEAR_START, 0)))
        child(flap, "joint", f"flap_hinge_{side}", type="hinge", axis="1 0 0", limited="true",
              range="-.48 .61", damping=".00028", armature=".0000017",
              stiffness=(".009" if side == "left" else ".012"),
              springref=(".160" if side == "left" else "0"))
        for i, notch in enumerate(tear_offsets):
            seam = notch + sign * .00026
            edge = sign * BOOK_WIDTH / 2
            paper_box(flap, f"flap_{side}_row_{i}", ((seam + edge) / 2, (i + .5) * spacing, 0),
                      (abs(edge - seam) / 2, spacing / 2, PAPER_HALF), .00010)
            child(flap, "site", f"tear_{side}_{i}", pos=num((seam, (i + .5) * spacing, PAPER_HALF)),
                  size=".0004", rgba=".70 .23 .15 0")
            if i % 2 == 0:
                for offset, width in ((.013, .013), (.029, .014), (.065, .045)):
                    ink(flap, sign * offset, (i + .50) * spacing, width, .00076)
                ink(flap, sign * .051, (i + .05) * spacing, .101, .00047, ".50 .60 .65 1")
                ink(flap, sign * .051, (i + .05) * spacing, .101, .00047, ".50 .60 .65 1", reverse=True)
        child(exclusions, "exclude", f"no_spine_{side}_collision", body1="bound_page", body2=f"flap_{side}")


def add_tissue(world, exclusions, equality):
    liner_z = TAPE_Z - (TAPE_THICKNESS + LINER_THICKNESS) / 2
    liner = child(world, "body", "release_liner", pos=num((TAPE_X, TAPE_Y, liner_z)))
    box(liner, "silicone_liner", (.0, .026, 0), (.0145, .036, LINER_THICKNESS / 2), ".72 .83 .85 1",
        friction=".30 .001 .0001", solref=".0013 1", solimp=".90 .99 .00012")
    box(world, "clipboard_base", (TAPE_X, TAPE_Y + .02, .779),
        (.036, .046, .018), ".23 .31 .36 1", friction=".47 .001 .0001")
    box(world, "clipboard_spring", (TAPE_X, TAPE_Y - .006, TAPE_Z + .003),
        (.024, .005, .0031), ".32 .39 .42 1")
    parent = world
    for i in range(TAPE_COUNT):
        origin = (TAPE_X, TAPE_Y, TAPE_Z) if i == 0 else (0., TAPE_STEP, 0.)
        tissue = child(parent, "body", f"tissue_{i}", pos=num(origin))
        if i == 0:
            child(tissue, "freejoint", "lift_tissue")
        else:
            child(tissue, "joint", f"tissue_bend_{i}", type="hinge", axis="1 0 0",
                  limited="true", range="-2.8 2.8", stiffness=".000055",
                  damping=".00017", armature=".00000055")
            child(exclusions, "exclude", f"tissue_neighbours_{i}", body1=f"tissue_{i - 1}", body2=f"tissue_{i}")
        for sign, side in ((1, "left"), (-1, "right")):
            name = f"adhesive_{i}_{side}"
            box(tissue, name, (sign * TAPE_HALF / 2, TAPE_STEP / 2, 0),
                (TAPE_HALF / 2, TAPE_STEP / 2, TAPE_THICKNESS / 2), ".83 .78 .57 .46", mass=".000020",
                friction=".52 .001 .0001", solref=".0013 1", solimp=".92 .99 .00012", condim="3",
                margin=".00012")
            # The glassine and adhesive sit flush while bonded. The contact
            # margin used for paper must not preload the liner in tension.
            child(exclusions, "pair", geom1=name, geom2="silicone_liner", condim="3",
                  margin="0", gap="0", solref=".0013 1", solimp=".92 .99 .00012",
                  friction=".30 .30 .001 .0001 .0001")
            # Anchors are set from the coincident observed contact geometry on first bonding.
            paper = "bound_page" if i == 0 else f"flap_{side}"
            child(equality, "connect", f"patch_{i}_{side}", body1=f"tissue_{i}", body2=paper,
                  anchor=num((sign * .0045, TAPE_STEP / 2, 0.)), active="false",
                  solref=".0017 1", solimp=".95 .995 .00015")
        if i < TAPE_COUNT - 1:
            child(equality, "connect", f"release_bond_{i}", body1=f"tissue_{i}", body2="release_liner",
                  anchor=num((0., TAPE_STEP / 2, 0.)), solref=".004 1", solimp=".93 .99 .00012")
        parent = tissue


def build_model():
    root = ET.fromstring(yam_env.build_xml())
    root.find("option").attrib.update(timestep=str(DT), noslip_iterations="0", iterations="55", impratio="3")
    ET.SubElement(root.find("option"), "flag", warmstart="disable")
    world = root.find("worldbody")
    for node in tuple(world):
        if node.get("name") in {"apple", "banana", "orange", "water_bottle", "bowl", "plate"}:
            world.remove(node)
    yam_env.camera(world, "repair_detail", (-.080, -.515, 1.325), (-.060, -.213, .815), up=(0, 0, 1))
    world.find('camera[@name="repair_detail"]').set("fovy", "35")
    for node in world.findall("light"):
        node.set("castshadow", "false")
    for geom in world.iter("geom"):
        if "_tip_" in geom.get("name", "") and "_contact_" in geom.get("name", ""):
            geom.attrib.update(friction=".95 .008 .0004", priority="1",
                               solref=".002 1", solimp=".93 .99 .00022", condim="3")
    equality = child(root, "equality")
    add_workbook(world, root.find("contact"))
    add_tissue(world, root.find("contact"), equality)
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model


def reset(model):
    data = mj.MjData(model)
    for side in ARMS:
        for i, value in enumerate(HOME, 1):
            data.joint(f"{side}_joint{i}").qpos[:] = value
        for sign in (-1, 1):
            data.joint(f"{side}_finger_{sign}").qpos[:] = .0475
    data.joint("flap_hinge_left").qpos[:] = .142
    data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    mj.mj_forward(model, data)
    return data
