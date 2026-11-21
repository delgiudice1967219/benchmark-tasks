"""Compare an internal terminal friction model with the original axial packs."""

from xml.etree import ElementTree as ET

from scene import BRASS, CONNECTORS, cell_layout, mesh_prism, v


TIN_NORMAL_PER_CIRCUIT_N = 1.47
TIN_AXIAL_DRAG_NOMINAL_PER_CIRCUIT_N = .50
TIN_FRICTION = TIN_AXIAL_DRAG_NOMINAL_PER_CIRCUIT_N / TIN_NORMAL_PER_CIRCUIT_N
PIN_HALF_WIDTH_M = .00043
TAB_HALF_OPENING_M = .00016
TAB_NOMINAL_FLEX_M = PIN_HALF_WIDTH_M - TAB_HALF_OPENING_M
TAB_STIFFNESS_N_M = TIN_NORMAL_PER_CIRCUIT_N / (2 * TAB_NOMINAL_FLEX_M)


def use_internal_tin_terminals(root):
    """Retain the stock keys, catches and loom. Only terminal-force generation differs."""
    world = root.find("worldbody")
    contact = root.find("contact")
    assets = root.find("asset")
    for spec in CONNECTORS:
        header = world.find(f'body[@name="{spec.name}_header"]')
        plug = world.find(f'body[@name="{spec.name}_plug"]')
        for part in list(header):
            if part.get("name", "").startswith(spec.name + "_beam_pack_"):
                header.remove(part)
        for pair in list(contact):
            if pair.get("geom1", "").startswith(spec.name + "_beam_pad_"):
                contact.remove(pair)
        for col, row, x, y, _ in cell_layout(spec):
            pin = f"{spec.name}_pin_{col}_{row}"
            for sign in (-1, 1):
                name = f"{spec.name}_terminal_tab_{col}_{row}_{sign}"
                flex = ET.SubElement(plug, "body", name=name, pos=v((x, y, -.00315)))
                ET.SubElement(flex, "joint", name=name, type="slide", axis=v((sign, 0, 0)),
                              range="-.000015 .0010", stiffness=str(TAB_STIFFNESS_N_M),
                              damping="1.35", armature=".0015", solreflimit=".003 1")
                # The beveled entrance guides the header pin; the longer straight face supplies drag.
                profile = [(sign*.00070, -.00113), (sign*.00070, .00145),
                           (sign*TAB_HALF_OPENING_M, .00145),
                           (sign*TAB_HALF_OPENING_M, -.00043), (sign*.00046, -.00113)]
                mesh_prism(flex, assets, name, profile, 1, .00049, (0., 0., 0.), BRASS,
                           mass=".00011", contype="0", conaffinity="0", group="3",
                           solref=".003 1", solimp=".95 .998 .00022")
                ET.SubElement(contact, "pair", geom1=pin, geom2=name, condim="3", solref=".003 1",
                              solimp=".95 .998 .00022", friction=v((TIN_FRICTION, TIN_FRICTION, .001, .0002, .0002)))
