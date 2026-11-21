"""Measure the two buckle catches on a single-axis bench fixture."""

import json
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

from scene import add_tooth_assets, female_buckle, male_buckle


def fixture():
    root = ET.fromstring("""<mujoco>
      <compiler angle="radian"/>
      <option gravity="0 0 0" timestep=".0002" iterations="80"/>
      <asset/><worldbody/><actuator/>
    </mujoco>""")
    world = root.find("worldbody")
    add_tooth_assets(root.find("asset"))
    female = ET.SubElement(world, "body", name="female", pos=".045 0 0", quat="0 0 0 1")
    male = ET.SubElement(world, "body", name="male", pos="0 0 0")
    ET.SubElement(male, "joint", type="slide", name="slide", axis="1 0 0", damping=".4")
    female_buckle(female, "left")
    male_buckle(male, "right")
    ET.SubElement(root.find("actuator"), "position", name="driver", joint="slide", kp="3700",
                  forcerange="-18 18")
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def bench_test(*, insert):
    model = fixture()
    data = mj.MjData(model)
    slide = model.joint("slide")
    travel = .0245 if insert else .012
    tines = [model.joint(f"right_buckle_tine_{sign}_flex").qposadr[0] for sign in (-1, 1)]
    peak_flex = 0.
    peak_push = 0.
    for index in range(4750):
        data.ctrl[0] = min(travel, travel * index / 3500)
        mj.mj_step(model, data)
        peak_flex = max(peak_flex, float(np.max(np.abs(data.qpos[tines]))))
        peak_push = max(peak_push, float(data.actuator_force[0]))

    model.actuator_gainprm[0, 0] = 0
    model.actuator_biasprm[0, :] = 0
    data.qfrc_applied[slide.dofadr[0]] = -5
    for _ in range(5000):
        mj.mj_step(model, data)
        if data.qpos[slide.qposadr[0]] < -.025:
            break
    contacts = {tuple(model.geom(int(geom_id)).name for geom_id in collision)
                for collision in data.contact.geom}
    catches = {pair for pair in contacts if "catch" in pair[0] and "tooth" in pair[1]}
    return dict(two_catches_held=len(catches) == 2, peak_push_n=round(peak_push, 3),
                peak_tine_flex_deg=round(np.rad2deg(peak_flex), 3),
                gap_after_pull_mm=round((.045 - data.qpos[slide.qposadr[0]]) * 1000, 3),
                catches=sorted(catches))


if __name__ == "__main__":
    positive, control = bench_test(insert=True), bench_test(insert=False)
    print(json.dumps(dict(fully_inserted=positive, shallow_insertion=control), indent=2))
    assert positive["two_catches_held"] and not control["two_catches_held"]
