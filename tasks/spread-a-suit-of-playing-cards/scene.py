"""Stock dual-YAM station with a full-size face-up suit packet on felt."""

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np
from dream_examples.yam_mujoco import yam_env


COUNT = 13
RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
TABLE_Z = .760
FELT_Z = TABLE_Z + .002
HALF_LENGTH = .044
HALF_WIDTH = .0315
THICKNESS = .00031
CARD_MASS = .00178
FELT_MU = .62
PAPER_MU = .22
DT = .00025
DECK_XY = np.array([-.215, -.098])
FELT_BOUNDS = (-.355, .155, -.222, .026)
READY = np.array([0., 1., 1., -.6, 0., 0.])
CAMERAS = {'top_cam':'top', 'left_cam':'left', 'right_cam':'right', 'judging_front':'ribbon_detail'}


def vector(values):
    return ' '.join(f'{x:.9g}' for x in values)


def add_cards(world):
    tz = FELT_Z
    for i in range(COUNT):
        body = ET.SubElement(world, 'body', name=f'card_{i:02d}',
                             pos=vector([*DECK_XY, tz + (i+.5)*THICKNESS]))
        ET.SubElement(body, 'freejoint', name=f'card_free_{i:02d}')
        ET.SubElement(body, 'geom', name=f'card_{i:02d}_panel_0', type='box',
                      size=vector([HALF_LENGTH, HALF_WIDTH, THICKNESS/2]), mass=str(CARD_MASS),
                      friction=f'{PAPER_MU} .002 .001', solref='.002 1', solimp='.95 .99 .00006',
                      condim='3', priority='0', rgba='.95 .94 .89 1', group='3')
        z = THICKNESS/2+.000015
        # The card has two like-numbered corners. The trailing one reads in this spread direction.
        for j, (x, y) in enumerate(((-.0418, .0201),(-.0356, .0201),(-.0356, .0288),(-.0418, .0288))):
            ET.SubElement(body, 'site', name=f'rank_{i:02d}_{j}', pos=vector([x,y,z]),
                          size='.00012', rgba='0 0 0 0', group='4')
        ET.SubElement(body, 'site', name=f'rank_{i:02d}_normal', pos=vector([-.039,.024,z]),
                      size='.0001', rgba='0 0 0 0', group='4')


def build_model(add_prints=False):
    root = ET.fromstring(yam_env.build_xml())
    option = root.find('option')
    option.attrib.update(timestep=str(DT), iterations='60', noslip_iterations='0', impratio='4')
    ET.SubElement(option, 'flag', warmstart='disable')
    root.find('visual/global').attrib.update(offwidth='1920', offheight='1080')
    world = root.find('worldbody')
    for part in list(world):
        if part.get('name') in {'apple', 'banana', 'orange', 'water_bottle', 'bowl', 'plate'}:
            world.remove(part)
    x0, x1, y0, y1 = FELT_BOUNDS
    ET.SubElement(world, 'geom', name='playing_felt', type='box',
                  pos=vector([(x0+x1)/2,(y0+y1)/2,TABLE_Z+.001]),
                  size=vector([(x1-x0)/2,(y1-y0)/2,.001]), mass='0',
                  friction=f'{FELT_MU} .002 .001', solref='.002 1', solimp='.95 .99 .00006',
                  condim='3', priority='1', rgba='.11 .31 .22 1')
    # Preserve the station cameras. This separate fixed review view is kept close enough to inspect ranks.
    yam_env.camera(world,'ribbon_detail',(-.137,-.117,1.438),(-.137,-.098,.762),up=(0,1,0))
    world.find('camera[@name="ribbon_detail"]').set('fovy','20')
    for light in world.findall('light'):
        light.set('castshadow','false')
    for g in world.iter('geom'):
        name = g.get('name', '')
        if '_tip_' in name and '_contact_' in name:
            # The stock CAD and joints stay untouched. A dry rubber-paper contact coefficient is an assumption.
            g.attrib.update(friction='.43 .004 .001', priority='3', solref='.002 1', solimp='.93 .99 .0004', condim='3')
    add_cards(world)
    if add_prints:
        add_art(root.find('asset'), world)
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def reset(model):
    data = mj.MjData(model)
    for side in ('left','right'):
        for i, angle in enumerate(READY,1):
            data.joint(f'{side}_joint{i}').qpos[:] = angle
        for f in (-1,1):
            data.joint(f'{side}_finger_{f}').qpos[:] = .0475
    data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:,0]]]
    mj.mj_forward(model, data)
    return data


def add_art(asset, world):
    """Original playing-card artwork with 1:1 dimensions and visible corner print."""
    from PIL import Image, ImageDraw, ImageFont

    directory = Path(__file__).resolve().parent / 'card_art'
    directory.mkdir(exist_ok=True)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    font = ImageFont.truetype(font_path, 115)
    side = 512
    outline = [(x, y) for x,y in [(-1,-1), (1,-1), (1,1),(-1,1)]]
    vertices = [value for x,y in outline for value in (HALF_LENGTH*x,HALF_WIDTH*y,THICKNESS/2+.000012)]
    ET.SubElement(asset, 'mesh', name='card_face', inertia='shell', vertex=vector(vertices),
                  texcoord='0 0 1 0 1 1 0 1', face='0 1 2 2 3 0')
    ink = '#ab1233'

    def glyph(surface, text, xy, bounds, color=ink):
        mark = Image.new('RGBA',(260,180))
        artist = ImageDraw.Draw(mark)
        artist.text((0,0),text,fill=color,font=font)
        mark = mark.crop(mark.getbbox())
        mark.thumbnail(bounds,Image.Resampling.LANCZOS)
        x,y=xy
        surface.paste(mark,(x+(bounds[0]-mark.width)//2,y+(bounds[1]-mark.height)//2),mark)

    def court(surface, rank):
        icon = Image.new('RGBA',(512,512))
        painter=ImageDraw.Draw(icon)
        robes={'J':'#28475a','Q':'#ac294d','K':'#7c2338'}
        painter.polygon(((144,253),(170,179),(215,155),(298,153),(342,179),(370,253)),
                        fill=robes[rank],outline='#3f3037',width=4)
        painter.ellipse((221,94,298,174),fill='#f0b777',outline='#573a31',width=4)
        if rank == 'J':
            hat=[(209,122),(234,70),(292,62),(313,119)]
        else:
            bump=11 if rank == 'Q' else 0
            hat=[(218,116),(211,74+bump),(234,91),(258,62+bump),(281,90),(308,69+bump),(295,116)]
        painter.polygon(hat,fill='#d3973d',outline='#61452b',width=4)
        painter.ellipse((238,127,244,134),fill='#402d2b')
        painter.ellipse((275,127,281,134),fill='#402d2b')
        painter.line((239,155,280,155),fill='#5d302b',width=4)
        painter.line((147,252,368,252),fill='#593342',width=8)
        surface.paste(icon,(0,0),icon)
        icon=icon.rotate(180)
        surface.paste(icon,(0,0),icon)

    pip_rows={
        1:[(256,255)],2:[(256,126),(256,386)],3:[(256,y) for y in (126,256,386)],
        4:[(x,y) for x in (177,335) for y in (126,386)],
        5:[(x,y) for x in (177,335) for y in (126,386)]+[(256,256)],
        6:[(x,y) for x in (177,335) for y in (126,256,386)],
        7:[(x,y) for x in (177,335) for y in (126,256,386)]+[(256,191)],
        8:[(x,y) for x in (177,335) for y in (126,256,386)]+[(256,191),(256,321)],
        9:[(x,y) for x in (177,335) for y in (126,213,300,386)]+[(256,256)],
        10:[(x,y) for x in (177,335) for y in (126,213,300,386)]+[(256,167),(256,343)]}
    for i, rank in enumerate(RANKS):
        img = Image.new('RGB',(side,side),'#fbf8ef')
        draw = ImageDraw.Draw(img)
        badge = Image.new('RGBA',(side,side))
        glyph(badge,rank,(13,25),(36,60))
        glyph(badge,'♥',(13,93),(36,41))
        img.paste(badge,(0,0),badge)
        badge = badge.rotate(180)
        img.paste(badge,(0,0),badge)
        draw.rounded_rectangle((2,3,509,508), radius=15,outline='#bba3a6',width=3)
        if i < 10:
            for x,y in pip_rows[i+1]:
                span = 108 if i == 0 else 46
                glyph(img,'♥',(x-span//2,y-span//2),(span,span))
        else:
            court(img,rank)
        path = directory / f'face_{i:02d}.png'
        # MJCF's first image row is the bottom of this mesh's UV surface.
        img.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path)
        ET.SubElement(asset, 'texture', name=f'card_art_{i:02d}', type='2d', file=str(path))
        ET.SubElement(asset, 'material', name=f'card_face_{i:02d}', texture=f'card_art_{i:02d}',
                      specular='.04', shininess='.04')
        body = world.find(f'body[@name="card_{i:02d}"]')
        ET.SubElement(body, 'geom', name=f'card_{i:02d}_rank_print', type='mesh', mesh='card_face',
                      material=f'card_face_{i:02d}', contype='0', conaffinity='0', mass='0')
