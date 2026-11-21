"""Real-scale motherboard rework scene in the calibrated dual-YAM station."""
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

if station_source := os.environ.get('ROBOCURVE_DREAM_YAM'):
    yam_env = SimpleNamespace(**runpy.run_path(str(Path(station_source) / 'yam_env.py')))
else:
    from dream_examples.yam_mujoco import yam_env

TOP = .760
CAMERAS = {name: name for name in ('top_cam', 'left_cam', 'right_cam', 'judging_front')}
PCB_Z = .866
PCB_X = .020
PCB_FRONT = -.058
PCB_T = .0016
PAD_Y = PCB_FRONT - .011
PITCH = .005
HOLE_RADIUS = .00054
LEAD_RADIUS = .0003
PAD_RADIUS = .00118
CAP_LEN = .0125
CAP_RADIUS = .005
CAP_PICK = np.array((.095, -.006, TOP+.034 + CAP_LEN / 2 + .0001))
CAP_AXIS = np.array((0., 0., 1.))
LEAD_LEN = .0064
PAD_XY = np.array(((PCB_X-PITCH/2, PAD_Y), (PCB_X+PITCH/2, PAD_Y)))
PCB_BASE = np.array((PCB_X, PCB_FRONT-.109, PCB_Z-PCB_T/2))
IRON_PICK = np.array((.213,.059,.832))
WIRE_PICK = np.array((-.184,.050,.830))
IRON_REACH = .087
WIRE_REACH = .070
PCB_RGB = (.07, .29, .20, 1)
PAD_RGB = (.75, .45, .20, 1)
MASKED = dict(contype='0', conaffinity='0')


def elem(parent, tag, **attributes):
    values = {name: yam_env.numbers(value) if isinstance(value, (tuple, list, np.ndarray)) else str(value)
              for name, value in attributes.items()}
    return ET.SubElement(parent, tag, values)


def box(parent, name, position, size, color, **attributes):
    return elem(parent, 'geom', name=name, type='box', pos=position,
                size=np.array(size)*.5, rgba=color, **attributes)


def cylinder(parent, name, position, radius, length, color, **attributes):
    return elem(parent, 'geom', name=name, type='cylinder', pos=position,
                size=(radius, length*.5), rgba=color, **attributes)


def capsule(parent, name, a, b, radius, color, **attributes):
    return elem(parent, 'geom', name=name, type='capsule', fromto=(*a,*b),
                size=radius, rgba=color, **attributes)


def sector_ring(parent, assets, label, x, y, low, high, inner, outer, color, n=12):
    """Annular convex sectors leave an open passage for each native lead collider."""
    step = 2*np.pi/n
    coordinates = [(radius*np.cos(angle), radius*np.sin(angle), z)
                   for z in (low, high) for angle in (-step/2, step/2)
                   for radius in (inner, outer)]
    elem(assets, 'mesh', name=label, vertex=np.ravel(coordinates))
    for index in range(n):
        elem(parent, 'geom', name=f'{label}_{index}', type='mesh', mesh=label,
             pos=(x,y,0), euler=(0,0,index*step), rgba=color,
             friction=(.49,.002,.0002), solref=(.004,1), solimp=(.96,.995,.0002))


def board_geometry(world, assets):
    body = elem(world, 'body', name='motherboard', pos=PCB_BASE)
    width, depth = .305, .218
    trace_y = .109-.011
    low, high = -.0008, .0008
    side = .002
    # Five rectangles plus two plated sectors fill a .004 m service strip.
    box(body,'laminate_main',(0,(trace_y-side-.109)/2,0),
        (width,trace_y-side+.109,PCB_T),PCB_RGB)
    box(body,'laminate_front',(0,(trace_y+side+.109)/2,0),
        (width,.109-trace_y-side,PCB_T),PCB_RGB)
    edges = (-width/2, -PITCH/2-side, -PITCH/2+side,
             PITCH/2-side, PITCH/2+side, width/2)
    for index in (0,2,4):
        a,b = edges[index:index+2]
        box(body,f'laminate_strip_{index}',((a+b)/2,trace_y,0),
            (b-a,side*2,PCB_T),PCB_RGB)
    # Exact passage is the inscribed polygon of the ring. The surround fills
    # its square boundary; its visible land is kept as a separate collider.
    for index,x in enumerate((-PITCH/2,PITCH/2)):
        label = f'hole_{index}'
        sector_ring(body,assets,label,x,trace_y,low,high,HOLE_RADIUS,side,
                    PCB_RGB)
        shell_width = .00063
        for angle in np.arange(4)*np.pi/2:
            local = np.array((np.cos(angle), np.sin(angle)))
            corner = (x,trace_y)+local*(side-shell_width/2)
            geom = box(body,f'corner_{index}_{int(round(angle*100))}',(*corner,0),
                       (shell_width,side*2,PCB_T),PCB_RGB)
            geom.set('euler',yam_env.numbers((0,0,angle)))
        sector_ring(body,assets,f'land_{index}',x,trace_y,high,high+.0001,
                    HOLE_RADIUS,PAD_RADIUS,PAD_RGB)
        sector_ring(body,assets,f'front_land_{index}',x,trace_y,low-.0001,low,
                    HOLE_RADIUS,PAD_RADIUS,PAD_RGB)
        # Fillet is drawn only in the camera after the corresponding passive
        # material response. It has no collision or role in grading.
        cylinder(body,f'fillet_{index}',(x,trace_y,high+.00042),.00108,.00066,
                 (.75,.80,.82,0),**MASKED)
        cylinder(body,f'fillet_taper_{index}',(x,trace_y,high+.00090),.00065,.00042,
                 (.71,.77,.83,0),**MASKED)

    # Readable no-logo procedural motherboard. Silkscreen, plated traces,
    # connector housings and real-scale clash bodies around the service area.
    inks = (.70,.76,.59,1)
    trace = (.20,.48,.32,1)
    back = high+.00005
    for i, (x, length) in enumerate(zip(np.linspace(-.131,.131,29),
                                        .033+.026*np.sin(np.arange(29)*.73)**2)):
        y = .075-(i%6)*.025
        box(body,f'back_trace_{i}',(x,y-length/2,back),(.00068,length,.00005),trace,**MASKED)
        x_to = x + .009*np.cos(i*2)
        box(body,f'back_elbow_{i}',((x+x_to)/2,y-length,back),
            (abs(x_to-x)+.0007,.00072,.00005),trace,**MASKED)
        for offset in (0,.008):
            cylinder(body,f'via_{i}_{int(offset*1e3)}',(x_to,y-length-offset,back),.00058,.00008,
                     (.48,.55,.40,1),**MASKED)
    for i in range(86):
        x = (i*29%151-75)*.00175
        y = (i*83%157-78)*.00113
        if abs(x) < .015 and y > .078:
            continue
        box(body,f'smd_{i}',(x,y,high+.00035),(.0018,.0031,.0007),(.17,.18,.18,1))
        for dx in (-.00093,.00093):
            box(body,f'contact_{i}_{dx:+.4f}',(x+dx,y,high+.00037),(.00045,.0031,.00065),
                (.61,.63,.58,1),**MASKED)
    for x,y in ((-.143,-.098),(-.143,.099),(.144,.099),(.144,-.098),(.025,-.090)):
        cylinder(body,f'screw_{x}_{y}',(x,y,high+.00011),.0021,.00022,(.50,.56,.53,1),**MASKED)
    for index in range(5):
        y = .049-index*.019
        box(body,f'expansion_slot_{index}',(-.039,y,-.0058),(.109,.0076,.010),(.86,.83,.74,1))
        box(body,f'expansion_groove_{index}',(-.039,y,-.0109),(.102,.0024,.0006),(.23,.24,.22,1),**MASKED)
    for index in range(4):
        x = .048+index*.014
        box(body,f'memory_slot_{index}',(x,-.041,-.0071),(.0071,.130,.013),(.085,.10,.14,1))
        for y in (-.102,.020):
            box(body,f'memory_latch_{index}_{y}',(x,y,-.013),(.0085,.006,.005),(.88,.86,.77,1))
    box(body,'cpu_socket',(-.012,.052,-.0056),(.047,.047,.009),(.84,.83,.75,1))
    box(body,'socket_shield',(-.012,.052,-.0105),(.041,.040,.0015),(.58,.60,.56,1))
    box(body,'power_header',(.114,-.023,-.0067),(.012,.049,.012),(.90,.89,.83,1))
    for i,y in enumerate((-.081,-.052,-.020,.015,.038)):
        box(body,f'rear_port_{i}',(-.140,y,-.009),(.021,.018,.017),(.59,.62,.62,1))
    for i,(x,y) in enumerate(((-.034,.093),(-.022,.093),(.029,.093),(.04,.093),(.059,.086))):
        cylinder(body,f'neighbor_cap_{i}',(x,y,-.0071),.0049,.0125,(.085,.17,.13,1))
        cylinder(body,f'neighbor_cap_lid_{i}',(x,y,-.0135),.0048,.0006,(.67,.69,.65,1),**MASKED)
    # Pads have visible opposite polarity in the setup close-up.
    for x in (-PITCH/2-.0033,PITCH/2+.0033):
        box(body,f'pmark_{x}',(x,trace_y,back),(.0014,.00035,.00006),inks,**MASKED)
    box(body,'pmark_plus',(PITCH/2+.0033,trace_y,back),(.00035,.0014,.00006),inks,**MASKED)

    for side in (-1,1):
        x = PCB_X+side*.157
        box(world,f'holder_base_{side}',(x,PCB_BASE[1],TOP+.006),
            (.061,.102,.012),(.17,.19,.19,1))
        cylinder(world,f'holder_column_{side}',(x,PCB_BASE[1],(PCB_Z+TOP)/2),.009,PCB_Z-TOP,
                 (.42,.46,.46,1))
        box(world,f'holder_clamp_{side}',(x,PCB_BASE[1],PCB_Z-.004),(.019,.042,.018),
            (.16,.19,.18,1))
        cylinder(world,f'holder_knob_{side}',(x,PCB_BASE[1],PCB_Z+.009),.009,.009,(.22,.24,.23,1))


def free_body(world, name, position, quaternion=None):
    obj = elem(world,'body',name=name,pos=position)
    if quaternion is not None:
        obj.set('quat',yam_env.numbers(quaternion))
    elem(obj,'freejoint',name=name+'_free')
    return obj


def capacitor(world, reversed_polarity=False, staged=False):
    quat = (0,0,0,1) if reversed_polarity else (1,0,0,0)
    if staged:
        below = PCB_Z-PCB_T-CAP_LEN-.00052
        block_xy = (PCB_X,PAD_Y)
        cylinder(world,'support_base',(*block_xy,TOP+.006),.016,.012,(.28,.31,.32,1))
        cylinder(world,'part_retainer_column',(*block_xy,(below+TOP+.012)/2),
                 .0046,below-TOP-.012,(.36,.39,.40,1))
        cylinder(world,'part_retainer_head',(*block_xy,below-.001),
                 .0046,.002,(.31,.38,.44,1),friction=(.87,.003,.0003),
                 solref=(.002,1),solimp=(.985,.999,.00010),priority=4)
        for ir in range(12):
            angle=ir*np.pi/6
            radial=np.array((np.cos(angle),np.sin(angle)))
            c=np.array(block_xy)+radial*.00543
            pocket=box(world,f'retainer_side_{ir}',(*c,below+.0017),
                       (.0005,.0029,.0034),(.32,.39,.45,1),friction=(.66,.003,.0003),
                       solref=(.002,1),solimp=(.985,.999,.00010),priority=4)
            pocket.set('euler',yam_env.numbers((0,0,angle)))
        start = (PCB_X,PAD_Y,PCB_Z-PCB_T-CAP_LEN/2-.00040)
    else:
        cylinder(world,'esd_parts_block',(CAP_PICK[0],CAP_PICK[1],TOP+.017),
            .00375,.034,(.30,.37,.44,1),friction=(.68,.003,.0003))
        start = CAP_PICK
    obj = free_body(world,'capacitor',start,quat)
    cap_green = (.065,.19,.14,1)
    cap_aluminum = (.65,.69,.66,1)
    cylinder(obj,'cap_shell',(0,0,0),CAP_RADIUS,CAP_LEN,cap_green,
             mass=.0024, friction=(.83,.003,.0003), priority=2)
    cylinder(obj,'cap_lid',(0,0,-CAP_LEN/2+.0005),CAP_RADIUS*.97,.001,cap_aluminum,
             mass=0, **MASKED)
    cylinder(obj,'cap_plug',(0,0,CAP_LEN/2-.0006),CAP_RADIUS*.94,.0011,(.14,.13,.12,1),
             mass=0, **MASKED)
    for index,angle in enumerate(np.linspace(-.66,.66,8)):
        strip = box(obj,f'polar_stripe_{index}',(.00497*np.cos(angle),.00497*np.sin(angle),0),
                    (.00012,.0010,.0118),(.78,.82,.68,1),mass=0,**MASKED)
        strip.set('euler',yam_env.numbers((0,0,angle)))
    for index,side in enumerate((-1,1)):
        root = elem(obj,'body',name=f'lead_body_{index}',pos=(side*(PITCH/2+.00029),0,CAP_LEN/2))
        elem(root,'joint',name=f'lead_twist_{index}',type='hinge',axis=(0,1,0),
             range=(-.16,.16), stiffness=.032, damping=.00011, armature=8e-8)
        # The preformed spring leads narrow at the tip to aid insertion.
        dx = -side*.00012
        capsule(root,f'lead_{index}',(0,0,.0001),(dx,0,LEAD_LEN-.00008),LEAD_RADIUS,
                cap_aluminum,mass=.00006,friction=(.60,.004,.0003),
                solref=(.003,1), solimp=(.96,.995,.0002))
        elem(root,'site',name=f'lead_tip_{index}',pos=(dx,0,LEAD_LEN),
             size=.00045,rgba=(0,0,0,0))


def local_frame(axis):
    """Return body frame with local +z pointing down the tool."""
    from scipy.spatial.transform import Rotation
    z = np.asarray(axis,float)
    z /= np.linalg.norm(z)
    x = np.cross((0,0,1),z)
    x /= np.linalg.norm(x)
    y = np.cross(z,x)
    xyzw = Rotation.from_matrix(np.column_stack((x,y,z))).as_quat()
    return np.r_[xyzw[3],xyzw[:3]]


def add_rest(world,name,point,radius,axis):
    horizontal = np.array(axis)
    horizontal[2] = 0
    horizontal /= np.linalg.norm(horizontal)
    normal = np.array((-horizontal[1],horizontal[0],0))
    for index,distance in enumerate((-.023,.023)):
        ref = point+distance*horizontal
        for side in (-1,1):
            center = ref+normal*side*radius*.76
            capsule(world,f'{name}_holder_{index}_{side}',
                    center+horizontal*(-.009)+(0,0,-radius*.76),
                    center+horizontal*.009+(0,0,-radius*.76),
                    .0024,(.35,.39,.40,1),friction=(.55,.004,.0003))
        box(world,f'{name}_stand_{index}',(ref[0],ref[1],(TOP+ref[2]-radius)/2),
            (.020,.020,ref[2]-radius-TOP),(.18,.20,.20,1))


def tools(world, iron_axis, wire_axis):
    # Racks hold the cylinder axis horizontal, with positive red/blue
    # plastic handle ends distinguished from the working tips.
    for name,point,axis,radius in [
        ('iron',IRON_PICK,iron_axis,.009),
        ('wire',WIRE_PICK,wire_axis,.0072),
    ]:
        add_rest(world,name,point,radius,axis)
        tub = free_body(world,name,point,local_frame(axis))
        cylinder(tub,name+'_grip',(0,0,0),radius,.084 if name=='iron' else .11,
                 (.10,.29,.64,1) if name=='iron' else (.72,.76,.78,1),
                 mass=.042 if name=='iron' else .023,friction=(.95,.019,.0015),condim=6,priority=2)
        if name=='iron':
            cylinder(tub,'iron_tail',(0,0,-.056),.007,.028,(.09,.11,.13,1),mass=.009)
            cylinder(tub,'iron_ring',(0,0,.041),.010,.006,(.26,.29,.28,1),mass=.008)
            cylinder(tub,'iron_barrel',(0,0,.0585),.0035,.029,(.55,.58,.57,1),mass=.009)
            cylinder(tub,'iron_chisel_shaft',(0,0,.0781),.00145,.011,
                     (.42,.45,.42,1),mass=.002)
            capsule(tub,'iron_neck',(0,0,.0829),(0,0,.0856),.00053,
                    (.66,.57,.43,1),mass=.0008)
            elem(tub,'geom',name='iron_hot_tip',type='sphere',pos=(0,0,IRON_REACH-.00068),
                 size=.00069,rgba=(.76,.68,.52,1),mass=.0002,solref=(.003,1),solimp=(.95,.999,.0001))
            elem(tub,'site',name='iron_point',pos=(0,0,IRON_REACH),size=.0004,rgba=(0,0,0,0))
        else:
            cylinder(tub,'wire_cap',(0,0,.053),.008,.007,(.15,.36,.70,1),mass=.005)
            capsule(tub,'solder_wire',(0,0,.054),(0,0,WIRE_REACH-.0004),.000405,
                    (.67,.74,.74,1),mass=.00018,solref=(.003,1),solimp=(.96,.996,.0001))
            elem(tub,'site',name='solder_point',pos=(0,0,WIRE_REACH),size=.0004,rgba=(0,0,0,0))


def scene_xml(iron_axis,wire_axis,reversed_polarity=False,staged=False):
    root = ET.fromstring(yam_env.build_xml())
    world = root.find('worldbody')
    assets = root.find('asset')
    for name in ('apple','banana','orange','water_bottle','bowl','plate'):
        world.remove(world.find(f'body[@name="{name}"]'))
    for asset in list(assets):
        if asset.get('name', '').startswith(('apple_', 'banana_', 'bowl_', 'brown_glaze', 'blue_glaze')):
            assets.remove(asset)
    for lamp in world.findall('light'):
        lamp.set('castshadow', 'false')
    root.find('option').attrib.update(iterations='65',timestep='.001')
    root.find('size').set('memory','192M')
    root.find('visual/quality').attrib.update(shadowsize='1024',offsamples='4')
    root.find('visual/global').attrib.update(offwidth='720',offheight='480')
    world.remove(world.find('camera[@name="top"]'))
    top = yam_env.calibrated_camera(world, 'top_cam', yam_env.CALIBRATION['cameras']['top'], world=True)
    for side in ('left', 'right'):
        world.find(f'.//camera[@name="{side}"]').set('name', f'{side}_cam')
    top_pos = np.fromstring(top.get('pos'), sep=' ')
    stand = world.find('body[@name="center_camera_stand"]')
    foot, mast, beam = stand.findall('geom')
    mast_y = top_pos[1] + .07
    foot.set('pos', yam_env.numbers((top_pos[0], mast_y, TOP+.008)))
    mast.set('fromto', yam_env.numbers((top_pos[0], mast_y, TOP+.016, top_pos[0], mast_y, top_pos[2])))
    beam.set('fromto', yam_env.numbers((top_pos[0], mast_y, top_pos[2], *top_pos + [0., .02, .02])))
    stand.find('body').attrib.update(pos=top.get('pos'), quat=top.get('quat'))
    center = np.array((PCB_X, PAD_Y, PCB_Z+.006))
    eye = center + .65 * np.array((-.527, -.374, .373))
    yam_env.camera(world, 'judging_front', eye, center)
    fovy = np.degrees(2. * np.arctan(20.955 * 480 / (2. * 34. * 640)))
    world.find('camera[@name="judging_front"]').set('fovy', str(fovy))
    for side in ('left','right'):
        for index in range(1,7):
            act=root.find(f'actuator/position[@name="{side}_joint{index}"]')
            limit=27 if index<=3 else 7
            act.set('forcerange',f'-{limit} {limit}')
        for index in (-1,1):
            act=root.find(f'actuator/position[@name="{side}_finger_{index}"]')
            act.set('forcerange','-40 40')
    board_geometry(world,assets)
    capacitor(world,reversed_polarity,staged)
    tools(world,iron_axis,wire_axis)
    equalities=elem(root,'equality')
    elem(equalities,'weld',name='installed_capacitor',body1='motherboard',
         body2='capacitor',active='false',solref=(.01,1),solimp=(.90,.98,.001))
    return ET.tostring(root,encoding='unicode')
