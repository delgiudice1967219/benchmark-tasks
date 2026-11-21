"""Real-size PC chassis and contact-only keyed power harness model."""
from dataclasses import dataclass, replace
import os
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import numpy as np

example_path = os.environ.get("ROBOCURVE_YAM_EXAMPLE_DIR")
if example_path:
    sys.path.insert(0, str(Path(example_path).resolve()))
    import yam_env
else:
    from dream_examples.yam_mujoco import yam_env

TABLE_Z = .760
CASE = np.array((.014, -.075, TABLE_Z))
WIDTH, LENGTH, HEIGHT = .387, .381, .230
BOARD_Z = TABLE_Z + .044
BENCH_DECK_Z = .875
BENCH_LIFT = BENCH_DECK_Z + .0266 - BOARD_Z
ROOF = TABLE_Z + HEIGHT
PITCH = .0042
PIN_LENGTH = .0056
HEAD_HEIGHT = .0104
PLUG_HEIGHT = .014
CATCH_Z = .0141
LAYOUT_SHIFT = np.array((0., -.065, 0.))
NYLON = (.145, .159, .172, 1)
SOCKET = (.244, .261, .278, 1)
SHELL = (.118, .13, .155, 1)
LEAD = (.21, .213, .235, 1)
BRASS = (.81, .65, .33, 1)
PCB = (.12, .227, .186, 1)

@dataclass(frozen=True)
class Connector:
    name: str
    columns: int
    socket: tuple
    angle: float
    cable_dock: tuple
    hole: tuple
    nominal_force: float
    camera: str
    dock_z: float

    @property
    def width(self):
        return .0516 if self.columns == 12 else .018

    @property
    def floor(self):
        return self.socket[2] + HEAD_HEIGHT - PIN_LENGTH

    @property
    def left_right(self):
        return np.array((np.cos(self.angle), np.sin(self.angle), 0.))

    @property
    def near_far(self):
        return np.array((-np.sin(self.angle), np.cos(self.angle), 0.))


ATX = Connector("ATX", 12, (.073, -.021, BOARD_Z), np.pi / 2,
                (.133, -.086), (.098, -.022, BOARD_Z -.012), 30., "atx_detail", TABLE_Z+.035)
EPS = Connector("EPS", 4, (-.102, .078, BOARD_Z), np.pi,
                (-.092, -.032), (-.097, .102, BOARD_Z -.012), 15., "eps_detail", BOARD_Z)
CONNECTORS = (ATX, EPS)


def v(values):
    return yam_env.numbers(values)


def body(parent, name, position, angle=0):
    return ET.SubElement(parent, "body", name=name, pos=v(position), euler=v((0, 0, angle)))


def box(parent, name, position, dimensions, color, **attributes):
    return ET.SubElement(parent, "geom", name=name, type="box", pos=v(position),
                         size=v(np.asarray(dimensions) / 2), rgba=v(color), **attributes)


def capsule(parent, name, a, b, radius, color, **attrs):
    return ET.SubElement(parent, "geom", name=name, type="capsule", fromto=v((*a, *b)),
                         size=str(radius), rgba=v(color), **attrs)


def tube(parent, name, position, radius, height, color, **attrs):
    return ET.SubElement(parent, "geom", name=name, type="cylinder", pos=v(position),
                         size=v((radius,height/2)), rgba=v(color), **attrs)


def mesh_prism(parent, assets, name, profile, long_axis, half_length, position, color,
               taper=False, **attrs):
    """Extrude an ordered convex profile in the two transverse coordinates."""
    vertex = []
    plane = [dimension for dimension in range(3) if dimension != long_axis]
    levels = [(-half_length, .67), (-half_length+.0011, 1.), (half_length, 1.)] if taper else [
        (-half_length, 1.), (half_length, 1.)]
    for long_pos, scale in levels:
        for point in profile:
            xyz = np.zeros(3)
            xyz[long_axis] = long_pos
            xyz[plane] = np.array(point)*scale
            vertex.extend(xyz)
    ET.SubElement(assets, "mesh", name=name, vertex=v(vertex))
    return ET.SubElement(parent, "geom", name=name, type="mesh", mesh=name, pos=v(position),
                         rgba=v(color), **attrs)


def build_chassis(world):
    chassis = body(world, "Q300L", CASE)
    # On its motherboard-tray side, the stock side panel is removed.
    box(chassis,"case_side_panel",(0,0,.004),(WIDTH,LENGTH,.005),SHELL)
    box(chassis,"rear_io_wall",(-.191,0,.114),(.004,.377,.226),SHELL)
    box(chassis,"front_mesh_wall",(.192,0,.114),(.004,.377,.226),SHELL)
    box(chassis,"upper_frame_wall",(0,.189,.115),(.385,.004,.229),SHELL)
    box(chassis,"lower_frame_wall",(0,-.189,.115),(.385,.004,.229),SHELL)
    for y in (-.185, .185):
        box(chassis,"rail_high_"+str(y),(0,y,.224),(.384,.012,.014),(.22,.23,.26,1))
    for x in (-.187, .187):
        box(chassis,"rail_end_"+str(x),(x,0,.224),(.012,.378,.014),(.22,.23,.26,1))
    tray_z = .033
    # The existing outside routing space stays open beside and above the board.
    box(chassis, "motherboard_tray", (-.0565,.035,tray_z),(.263,.263,.004),(.18,.195,.212,1))
    box(chassis, "front_tray", (.14,0,tray_z),(.082,.37,.004),(.17,.185,.209,1))
    box(chassis, "cable_routing_lip",(.093,-.016,tray_z-.002),(.016,.18,.003),(.243,.258,.29,1))
    box(chassis, "PSU",(-.122,-.137,.077),(.125,.086,.150),(.22,.225,.24,1))
    for i, y in enumerate((-.160,-.145,-.130,-.115)):
        box(chassis,f"PSU_ridge_{i}",(-.122,y,.153),(.099,.003,.003),(.10,.115,.13,1))


def build_test_bench(world):
    """The BC1 V2's 360 × 270 mm deck and PSU mounted to a foot under it."""
    deck = BENCH_DECK_Z - BENCH_LIFT
    base = TABLE_Z - BENCH_LIFT
    bench = body(world,"BC1V2",(-.024,-.026,base))
    aluminum=(.58,.61,.635,1)
    box(bench,"case_bench_deck",(0,0,deck-base-.004),(.360,.270,.008),aluminum)
    for x in (-.154,.154):
        for y in (-.109,.109):
            capsule(bench,f"case_bench_leg_{x}_{y}",(x,y,.009),(x,y*.80,.109),.0069,aluminum)
            box(bench,f"case_bench_rubber_{x}_{y}",(x,y,.004),(.027,.029,.008),SHELL)
        box(bench,f"case_bench_foot_{x}",(x,0,.015),(.012,.221,.016),aluminum)
    box(bench,"case_bench_CX550",(-.044,.033,.055),(.125,.150,.086),SHELL)
    for i in range(11):
        box(bench,f"case_bench_vent_{i}",(-.099+i*.010,.033,.099),(.003,.089,.002),NYLON)
    for x in (-.168,.168):
        for y in np.linspace(-.095,.095,7):
            tube(bench,f"case_bench_screw_{x}_{y}",(x,y,.116),.002,.002,BRASS,
                 contype="0",conaffinity="0")


def build_board(world, standoff_height):
    board = body(world, "B450M", (-.044,-.033,BOARD_Z-.001))
    box(board, "board_fr4", (0,0,0),(.240,.244,.0016),PCB)
    for i, (x,y) in enumerate(((-.11,-.11),(.10,-.11),(-.11,.11),(.10,.11),(.007,.004))):
        tube(board,f"standoff_{i}",(x,y,-.001-standoff_height/2),.0025,standoff_height,BRASS)
        tube(board,f"screw_{i}",(x,y,.0011),.0024,.0013,(.65,.70,.69,1))
    for i, (y,length,height) in enumerate(((.058,.030,.04),(.015,.039,.035),(-.029,.043,.028),(.103,.031,.022))):
        box(board,f"rear_port_{i}",(-.105,y,height/2),(.025,length,height),(.62,.65,.69,1))
    box(board,"cpu_socket",(-.013,.046,.006),(.046,.045,.012),(.26,.28,.29,1))
    box(board,"heat_spreader",(-.013,.046,.013),(.039,.039,.004),(.63,.68,.72,1))
    for i in range(4):
        box(board,f"dimm_{i}",(.029+.009*i,.042,.005),(.005,.119,.010),(.206,.226,.24,1))
    for i in range(3):
        box(board,f"pcie_{i}",(-.047,-.039-.021*i,.004),(.085-.019*i,.009,.008),(.195,.214,.229,1))
    for i in range(6):
        x = -.054+.014*i
        tube(board,f"polymer_cap_{i}",(x,.076,.004),.0031,.008,(.62,.675,.694,1))
    box(board,"chipset_heatsink",(.061,-.067,.008),(.039,.032,.016),(.18,.20,.218,1))
    for i in range(5):
        box(board,f"chipset_fin_{i}",(.046+.007*i,-.067,.017),(.002,.030,.003),(.29,.309,.321,1))
    for i in range(9):
        box(board,f"sata_{i}",(.083+.006*(i//3),-.085+.008*(i%3),.004),(.004,.006,.008),NYLON)
    for i in range(14):
        x = -.071+.011*i
        box(board,f"trace_{i}",(x,-.106+i*.0002,.0011),(.00036,.032,.0004),(.42,.51,.33,1),
            contype="0",conaffinity="0")
    for i in range(6):
        x = -.061+.019*i
        box(board,f"logic_{i}",(x,-.106,.0026),(.008,.006,.0036),(.12,.13,.15,1))


def cell_layout(spec):
    """Alternating mapped square/chamfered Mini-Fit Jr-style cells."""
    square = {"ATX": ({0, 1, 4, 5, 8, 10}, {2, 3, 6, 7, 9, 11}),
              "EPS": ({0, 3}, {1, 2})}
    return [(col, row, (col - (spec.columns - 1)/2)*PITCH, (row-.5)*PITCH,
              col in square[spec.name][row])
            for col in range(spec.columns) for row in range(2)]


def build_header(root, world, spec):
    header = body(world, spec.name + "_header", spec.socket, spec.angle)
    w = spec.width
    rail_depth=.0090
    box(header,spec.name+"_socket_floor",(0,0,.0024),(w,.0116,.0048),SOCKET)
    for sign in (-1,1):
        box(header,f"{spec.name}_header_side_{sign}",(0,sign*.00530,.00695),
            (w,.0013,.0069),SOCKET, friction=".16 .001 .0002")
        box(header,f"{spec.name}_header_end_{sign}",(sign*(w/2-.00041),0,.00695),
            (.00082,rail_depth,.0069),SOCKET, friction=".16 .001 .0002")
    # The 5566-family drawing shows a continuous web between the two rows
    # and walls separating the individual contact cavities.
    web=(.00046,.0056)
    for col in range(spec.columns-1):
        x=(col-(spec.columns-2)/2)*PITCH
        box(header,f"{spec.name}_cavity_web_{col}",(x,0,.0076),
            (web[0],rail_depth,web[1]),SOCKET,friction=".13 .001 .0002")
    box(header,f"{spec.name}_cavity_divider",(0,0,.0076),
        (w-.00164,web[0],web[1]),SOCKET,friction=".13 .001 .0002")
    for col,row,x,y,square in cell_layout(spec):
        # Full pin grid is drawn. The terminal beam pack below supplies the
        # lumped axial reaction in the native contact solver.
        box(header,f"{spec.name}_pin_{col}_{row}",(x,y,.0070),(.00086,.00086,.0044),BRASS,
            contype="0", conaffinity="0")
        if not square:
            sign = 1 if row else -1
            profile = [(-.00182,sign*.00182),(-.00098,sign*.00182),(-.00182,sign*.00098)]
            mesh_prism(header,root.find("asset"),f"{spec.name}_key_corner_{col}_{row}",
                       profile,2,.00155,(x,y,.0084),SOCKET,
                       friction=".12 .001 .0002")
    # Catch is narrow so it is accessible to a real thumb from above.
    # Positive y is outward from the header. Its top is an entry cam and
    # its bottom is a horizontal retention shoulder.
    cam_profile = [(-.0004,-.0018),(.0025,-.0018),(.0025,-.00115),(.00003,.0015),(-.0004,.0015)]
    for lane,x in enumerate(latch_centers(spec)):
        mesh_prism(header,root.find("asset"),f"{spec.name}_ledge_{lane}",cam_profile,0,.00326,
                   (x,.0058,CATCH_Z),SOCKET,friction=".19 .002 .0002")
    # Plunger beam packs stand in for spring temples inside the actual pins.
    # They shorten under the keyed plug and rebound unless its catch holds.
    for lane,x in enumerate(np.linspace(-w*.32,w*.32,2)):
        # Top starts just below the contact grid. A keyed nose drives each pack.
        push = body(header,f"{spec.name}_beam_pack_{lane}", (x,0,.0082))
        ET.SubElement(push,"joint",name=f"{spec.name}_beam_{lane}",type="slide",axis="0 0 1",
                      range="-.0048 0.0001",stiffness=str(spec.nominal_force/2/.0043),
                      damping=".65",armature=".0004",solreflimit=".003 1")
        box(push,f"{spec.name}_beam_pad_{lane}",(0,0,0),(w*.34,.0084,.00045),BRASS,mass=".003",
            contype="4",conaffinity="0",solref=".003 1",friction=".12 .001 .0002")
        for col,row,px,_,_ in cell_layout(spec):
            if abs(px-x) <= w*.17:
                ET.SubElement(root.find("contact"),"pair",geom1=f"{spec.name}_beam_pad_{lane}",
                              geom2=f"{spec.name}_tooth_{col}_{row}",solref=".003 1",
                              friction=".13 .13 .001 .0002 .0002")


def latch_centers(spec):
    return (0.,) if spec.columns==12 else (-.0042,.0042)


def chamfer_profile(square,row):
    half=.00172
    if square:
        return [(-half,-half),(half,-half),(half,half),(-half,half)]
    cut=.00076
    square_points=[(-half,-half+cut),(-half+cut,-half),(half,-half),(half,half),(-half,half)]
    return square_points if row==0 else [(x,-y) for x,y in square_points]


def build_plug(root,world,spec):
    root_z=spec.dock_z+PIN_LENGTH+.0004
    plug=body(world,spec.name+"_plug",(*spec.cable_dock,root_z),spec.angle)
    ET.SubElement(plug,"freejoint",name=spec.name+"_free")
    # EPS uses the real pre-paired pair of moulded four-position halves.
    divides= (0.,) if spec.columns==12 else (-.0044,.0044)
    for lane,x in enumerate(divides):
        w=spec.width if spec.columns==12 else .00855
        box(plug,f"{spec.name}_grip_{lane}",(x,0.,PLUG_HEIGHT/2),(w,.0096,PLUG_HEIGHT),NYLON,
            mass=".024" if spec.columns==12 else ".008", friction="1.04 .012 .001",priority="2")
    for col,row,x,y,square in cell_layout(spec):
        mesh_prism(plug,root.find("asset"),f"{spec.name}_tooth_{col}_{row}",chamfer_profile(square,row),
                   2,PIN_LENGTH/2,(x,y,-PIN_LENGTH/2),NYLON,mass=".0003",
                   solref=".003 1",friction=".13 .001 .0002",taper=True)
    for lane,x in enumerate(latch_centers(spec)):
        box(plug,f"{spec.name}_clip_root_{lane}",(x,.00675,.0121),(.0065,.0045,.0016),
            NYLON,mass=".001",friction=".45 .003 .001")
        clip=body(plug,f"{spec.name}_clip_{lane}",(x,.00915,.0120))
        ET.SubElement(clip,"joint",name=f"{spec.name}_clip_bend_{lane}",type="hinge",axis="1 0 0",
                      range="-.10 .57",stiffness=".88" if spec is ATX else ".52",
                      damping=".009",armature=".000004",
                      solreflimit=".003 1")
        box(clip,f"{spec.name}_flex_beam_{lane}",(0,.0003,-.0076),
            (.0065,.0009,.0146),(.29,.309,.325,1),mass=".0012",friction=".19 .001 .0002")
        # Nose slides outside the fixed catch, then snaps inward underneath.
        cam_profile = [(-.00265,.00015),(-.0008,-.0018),(.0006,-.0018),(.0006,.00105),(-.00265,.00105)]
        mesh_prism(clip,root.find("asset"),f"{spec.name}_clip_nose_{lane}",cam_profile,0,.0029,
                   (0.,0.,-.0114),(.37,.39,.408,1),mass=".001",solref=".003 1",friction=".19 .001 .0002")
        ET.SubElement(root.find("contact"),"exclude",body1=f"{spec.name}_plug",body2=f"{spec.name}_clip_{lane}")
    ET.SubElement(plug,"site",name=spec.name+"_strain",pos=v((0,0,.0143)),size=".0006",rgba="0 0 0 0")
    return plug


def _segment_range(start, end, count):
    return [np.array(start)+(np.array(end)-start)*i/count for i in range(count)]


def cable_points(spec):
    """Sample a rest arc from the existing case loom opening to the plug."""
    end=np.array((*spec.cable_dock,spec.dock_z+PIN_LENGTH+.0147))
    first=np.array(spec.hole)
    mid=(first+end)/2+np.array((.029 if spec.name=="ATX" else -.017, .010, .039))
    top=first+np.array((0,0,.028))
    boot=end+np.array((0.,0.,.018))
    return [first,top,*_segment_range(top,mid,3)[1:],*_segment_range(mid,boot,4),boot,end]


def build_cable(root,world,spec,open_bench):
    points=cable_points(spec)
    # Behind-tray visual section locates the real PSU and routing entry.
    if open_bench:
        deck=BENCH_DECK_Z-BENCH_LIFT
        turn=(spec.hole[0],spec.hole[1],deck-.031)
        for i,(a,b) in enumerate(zip(((-.003,.060,deck-.040),turn),(turn,spec.hole))):
            capsule(world,f"{spec.name}_under_bench_{i}",a,b,.0042,LEAD,
                    contype="0",conaffinity="0")
    else:
        capsule(world,spec.name+"_behind_tray",(-.043,-.211,TABLE_Z+.035),spec.hole,.0042,LEAD,
                contype="0",conaffinity="0")
    link=body(world,f"{spec.name}_bundle_anchor",points[0])
    for index,(a,b) in enumerate(zip(points[:-1],points[1:])):
        if index:
            link=body(link,f"{spec.name}_bundle_{index}",a-points[index-1])
            ET.SubElement(link,"joint",name=f"{spec.name}_bend_{index}",type="ball",
                          stiffness=".027" if spec.name=="EPS" else ".041",damping=".0015",armature=".000002")
        d=b-a
        radius=.0025 if spec.name=="EPS" else .00365
        capsule(link,f"{spec.name}_lead_contact_{index}",(0,0,0),d,radius,LEAD,
                mass=".0031" if spec.name=="ATX" else ".0014",
                contype="8",conaffinity="1",friction=".46 .003 .001")
        normal=np.cross(d,[0,0,1])
        if np.linalg.norm(normal)<.001:
            normal=np.array([1.,0.,0.])
        else:
            normal/=np.linalg.norm(normal)
        for wire in (-1,0,1):
            offset=normal*radius*wire*.69
            capsule(link,f"{spec.name}_wire_{index}_{wire}",offset,d+offset,radius*.52,
                    ((.09,.112,.14,1) if wire else (.29,.296,.31,1)),mass="0",contype="0",conaffinity="0")
    ET.SubElement(root.find("equality"),"weld",name=spec.name+"_strain_relief",body1=f"{spec.name}_bundle_{len(points)-2}",
                  body2=spec.name+"_plug",torquescale=".023",solref=".010 1",solimp=".92 .995 .0011")


def scene_xml(reset_skew=False, layout_shift=LAYOUT_SHIFT, layout_yaw=0., open_bench=False):
    root=ET.fromstring(yam_env.build_xml())
    root.set("model","Desktop motherboard harness")
    root.find("option").attrib.update(timestep=".002",iterations="60",cone="elliptic")
    root.find("visual/global").set("offwidth","800")
    world=root.find("worldbody")
    for light in world.findall("light"):
        light.set("castshadow","false")
    for name in ("apple","banana","orange","water_bottle","bowl","plate"):
        world.remove(world.find(f"body[@name='{name}']"))
    world.remove(world.find("camera[@name='top']"))
    yam_env.calibrated_camera(world,"top",yam_env.CALIBRATION["cameras"]["top"],world=True)
    turn = np.array(((np.cos(layout_yaw),-np.sin(layout_yaw),0),
                     (np.sin(layout_yaw),np.cos(layout_yaw),0),(0,0,1)))
    layout_shift = layout_shift + (0,0,BENCH_LIFT if open_bench else 0)

    def place(point):
        return CASE+turn@(np.asarray(point)-CASE)+layout_shift

    for name,eye,target,fov in (
        ("case_detail",(.020,-.480,1.455),(.015,-.056,.884),43),
        (ATX.camera,(.222,-.319,1.267),(.064,-.028,.851),27),
        (EPS.camera,(-.324,-.153,1.188),(-.093,.052,.850),27)):
        yam_env.camera(world,name,place(eye),place(target))
        world.find(f"camera[@name='{name}']").set("fovy",str(fov))
    ET.SubElement(root,"equality")
    torque_caps=(27,27,27,7,7,7)
    for actuator in root.find("actuator"):
        name=actuator.get("name","")
        if "_joint" in name:
            index=int(name[-1])-1
            torque=torque_caps[index]
            actuator.attrib.update(forcerange=v((-torque,torque)))
        elif "finger" in name:
            actuator.attrib.update(forcerange="-40 40",kp="4000",kv="65")
    scene_objects_before = len(world)
    specs=CONNECTORS
    if open_bench:
        build_test_bench(world)
        deck=BENCH_DECK_Z-BENCH_LIFT
        specs=(replace(ATX,dock_z=deck,hole=(.167,-.046,deck-.006)),
               replace(EPS,hole=(-.125,.122,deck-.006)))
    else:
        build_chassis(world)
    build_board(world,.025 if open_bench else .008)
    for spec in specs:
        build_header(root,world,spec)
        build_plug(root,world,spec)
        build_cable(root,world,spec,open_bench)
    if reset_skew:
        plug=world.find("body[@name='ATX_plug']")
        plug.set("euler",v((0.,0.,ATX.angle+np.pi)))
        # Keep its strain relief at the same central exit.
    for item in list(world)[scene_objects_before:]:
        if "pos" in item.attrib:
            item.set("pos",v(place(np.fromstring(item.get("pos"),sep=" "))))
            orientation = np.fromstring(item.get("euler","0 0 0"),sep=" ")
            orientation[2]+=layout_yaw
            item.set("euler",v(orientation))
        elif "fromto" in item.attrib:
            item.set("fromto",v(np.concatenate([place(endpoint) for endpoint in
                                      np.fromstring(item.get("fromto"),sep=" ").reshape(2,3)])))
    for geom in world.find("body[@name='center_camera_stand']").iter("geom"):
        geom.set("group","3")
    for geom in world.iter("geom"):
        if geom.get("rgba") == "0 0 0 0":
            geom.set("group", "3")
    return ET.tostring(root,encoding="unicode")
