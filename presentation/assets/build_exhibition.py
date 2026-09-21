"""Original interpretive exhibition machinery. Blender 4.5: --background --python this-file.
Coordinates authored Y-up, converted to Blender Z-up before glTF export.
Not manufacturer CAD; dimensions are visual schematic units, not source measurements.
"""
import bpy
import math
from pathlib import Path
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'frontend/public/models'
OUT.mkdir(parents=True, exist_ok=True)

def mat(name, color, metal=0, rough=.4):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Metallic'].default_value = metal
    p.inputs['Roughness'].default_value = rough
    return m

steel = mat('Satin machined stainless', (.49,.55,.59), .85,.29)
roll = mat('Ground hardened steel', (.63,.67,.69), .92,.22)
frame = mat('Graphite powdercoat', (.065,.095,.11), .35,.38)
pearl = mat('Ceramic pearl service panels', (.74,.79,.78), .22,.35)
dark = mat('Gaskets and rubber', (.018,.025,.029), .05,.72)
teal = mat('Petrol enamel', (.018,.25,.27), .3,.34)
orange = mat('Safety amber', (.93,.32,.07), .15,.38)
web = mat('Coated electrode illustrative', (.065,.072,.081), .15,.69)

def finish(o, name, material, bevel=0):
    o.name = name
    o.data.materials.append(material)
    if bevel:
        b=o.modifiers.new('Machined edge radii','BEVEL'); b.width=bevel; b.segments=3
        b=o.modifiers.new('Weighted corner normals','WEIGHTED_NORMAL')
    return o

def box(name, pos, size, material=frame, bevel=.035):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    o=bpy.context.object; o.dimensions=size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(o,name,material,bevel)

def lathe(name, x,y,z, profile, material=steel, segments=64):
    # Axial/radius profile along world Z. Includes steps and chamfers, not a plain cylinder.
    vs=[]; faces=[]
    for a,r in profile:
        for i in range(segments):
            t=i*math.tau/segments; vs.append((x+r*math.cos(t),y+r*math.sin(t),z+a))
    for j in range(len(profile)-1):
        for i in range(segments):
            a=j*segments+i; b=j*segments+(i+1)%segments
            faces.append((a,b,b+segments,a+segments))
    faces += [tuple(reversed(range(segments))),tuple((len(profile)-1)*segments+i for i in range(segments))]
    mesh=bpy.data.meshes.new(name); mesh.from_pydata(vs,[],faces); mesh.update()
    o=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(o)
    finish(o,name,material)
    for p in mesh.polygons: p.use_smooth = len(p.vertices)==4
    return o

def shaft(name,x,y,r=.6,length=2.8):
    h=length/2
    return lathe(name,x,y,0,[(-h-.48,.17),(-h-.35,.17),(-h-.3,.23),(-h-.05,.23),(-h,r-.035),(-h+.05,r),(h-.05,r),(h,r-.035),(h+.05,.23),(h+.3,.23),(h+.35,.17),(h+.48,.17)],roll)

def bolt(x,y,z):
    lathe('M16 hex socket',x,y,z,[(-.025,.055),(.035,.055),(.045,.042)],steel,6)
    lathe('Socket recess',x,y,z+.046,[(0,.019),(.003,.019)],dark,6)

def bearing(x,y,z,r=.32):
    box('Split bearing chock',(x,y,z),(r*2.7,r*2.6,.32),pearl,.055)
    lathe('Bearing flange',x,y,z,[(-.19,r*.72),(-.18,r),(.18,r),(.2,r*.85)],steel)
    lathe('Shaft seal',x,y,z+.205,[(0,r*.66),(.026,r*.66)],dark)
    lathe('Shaft cap',x,y,z+.24,[(0,r*.5),(.025,r*.5)],steel)
    for dx in [-r*.95,r*.95]:
        for dy in [-r*.85,r*.85]: bolt(x+dx,y+dy,z+.18)

def pipe(name,points,r=.035,material=dark):
    c=bpy.data.curves.new(name,'CURVE'); c.dimensions='3D'; c.bevel_depth=r; c.bevel_resolution=3
    s=c.splines.new('BEZIER'); s.bezier_points.add(len(points)-1)
    for p,co in zip(s.bezier_points,points): p.co=co; p.handle_left_type='AUTO'; p.handle_right_type='AUTO'
    o=bpy.data.objects.new(name,c); bpy.context.collection.objects.link(o); o.data.materials.append(material)

def base(width=4.8):
    box('Welded machine bed',(0,.38,0),(width,.48,3.8),frame,.09)
    for x in [-width/2+.35,width/2-.35]:
        for z in [-1.55,1.55]:
            box('Leveling foot',(x,.08,z),(.42,.16,.42),dark,.05)
            box('Adjustable mount',(x,.2,z),(.17,.2,.17),steel,.025)
    for z in [-1.91,1.91]:
        box('Removable fascia',(0,.41,z),(width-.3,.26,.04),pearl,.015)
        for x in [-width/2+.3,width/2-.3]: bolt(x,.41,z+.025)

def console(x):
    box('HMI pedestal',(x,1.0,1.8),(.18,1.3,.2),frame)
    box('HMI enclosure',(x,1.85,1.8),(.82,.65,.24),pearl,.07)
    box('Inset display bezel',(x,1.89,1.94),(.65,.43,.025),dark,.025)
    box('Status display',(x,1.9,1.958),(.56,.32,.012),teal,.015)
    for dy in [0,.08,.16]: box('Interface trace',(x-.09,1.8+dy,1.968),(.28,.012,.004),pearl,0)
    lathe('Emergency stop',x+.25,1.6,1.97,[(0,.06),(.07,.06)],orange,24)

def calender():
    base()
    for z in [-1.62,1.62]:
        for x in [-.91,.91]:
            box('Portal column',(x,2.05,z),(.32,2.95,.4),frame,.055)
            box('Ground linear slide',(x,2.35,z+.22),(.095,2.1,.055),steel,.012)
        box('Loading bridge',(0,3.64,z),(2.25,.42,.55),frame,.065)
        box('Bearing guide bed',(0,.85,z),(2.2,.3,.52),frame)
        bearing(0,1.36,z,.39); bearing(0,2.65,z,.39)
        # Vertical hydraulic ram uses lathe rotated from Z-axis to Y-axis.
        ram=lathe('Hydraulic loading ram',0,0,0,[(-.36,.23),(-.3,.29),(.3,.29),(.34,.23)],teal)
        ram.rotation_euler.x=math.pi/2; ram.location=(0,4.12,z)
        rod=lathe('Chrome loading piston',0,0,0,[(-.35,.095),(.35,.095)],roll)
        rod.rotation_euler.x=math.pi/2; rod.location=(0,3.42,z)
        pipe('Hydraulic supply',[(-.18,4.25,z),(-.55,4.34,z),(-1.16,3.4,z),(-1.16,.7,z)],.034)
    shaft('Upper precision roll',0,2.65,.62)
    shaft('Lower precision roll',0,1.36,.62)
    for x in [-1.92,1.92]:
        shaft('Web guide roll',x,1.84,.16,2.55)
        for z in [-1.52,1.52]:
            box('Guide support',(x,1.16,z),(.2,1.14,.22),frame)
            bearing(x,1.84,z,.15)
    box('Electrode through nip',(0,2.0,0),(5.8,.025,2.35),web,.008)
    box('Drive cabinet',(0,1.38,-2.35),(2.0,1.55,.75),pearl,.09)
    for y in [1.0+i*.09 for i in range(8)]: box('Cabinet ventilation',(0,y,-2.738),(1.25,.022,.009),dark,.006)
    console(2.22)

def coater():
    base(5.4)
    shaft('Backing roll',0,1.35,.63)
    for z in [-1.62,1.62]:
        box('Bearing pedestal',(0,1.04,z),(.95,.9,.48),frame,.055)
        bearing(0,1.35,z,.34)
        box('Die positioning tower',(.76,2.0,z),(.3,2.65,.32),frame)
        box('Die cross slide',(.28,2.88,z),(1.45,.2,.32),steel)
        box('Micrometer carriage',(-.12,2.64,z),(.38,.4,.42),teal)
        p=lathe('Micrometer spindle',0,0,0,[(-.32,.065),(.23,.065),(.25,.14),(.43,.14)],steel,32)
        p.rotation_euler.x=math.pi/2; p.location=(-.12,3.1,z)
    # Die head with tapered wedge section extruded across the web.
    verts=[(-.48,2.58,-1.36),(.22,2.58,-1.36),(.15,2.25,-1.36),(-.04,2.025,-1.36),(-.15,2.025,-1.36),(-.48,2.58,1.36),(.22,2.58,1.36),(.15,2.25,1.36),(-.04,2.025,1.36),(-.15,2.025,1.36)]
    faces=[(4,3,2,1,0),(5,6,7,8,9)]+[(i,(i+1)%5,(i+1)%5+5,i+5) for i in range(5)]
    m=bpy.data.meshes.new('Precision die wedge'); m.from_pydata(verts,[],faces); m.update()
    o=bpy.data.objects.new('Slot die manifold',m); bpy.context.collection.objects.link(o); finish(o,o.name,steel,.014)
    box('Die lip seam',(-.098,2.033,0),(.014,.012,2.42),dark,.002)
    for z in [-1.1,-.7,-.3,.3,.7,1.1]:
        b=box('Die clamping screw',(-.49,2.43,z),(.06,.075,.075),steel,.012)
    for x in [-2.08,2.08]:
        shaft('Infeed outfeed guide',x,1.82,.16,2.55)
        for z in [-1.52,1.52]:
            box('Guide bracket',(x,1.15,z),(.22,1.18,.24),frame)
            bearing(x,1.82,z,.15)
    box('Foil entering coating bead',(-1.4,1.992,0),(2.8,.015,2.35),roll,.003)
    box('Coated web leaving die',(1.4,2.008,0),(2.8,.018,2.32),web,.003)
    pipe('Slurry feed hose',[(.02,2.61,-.75),(.15,3.18,-.85),(1.3,3.12,-1.72),(1.65,1.0,-1.9)],.06,teal)
    box('Metering pump cabinet',(1.6,1.13,-2.1),(1.1,1.15,.65),pearl,.07)
    console(2.45)

for name,build in [('calender',calender),('coater',coater)]:
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    build()
    # Group under coordinate-conversion root, apply conversion for correct glTF Y-up.
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.convert(target='MESH')
    for o in list(bpy.context.selected_objects):
        # Rotate world coordinates +90° about X: authored Y becomes Blender Z.
        from mathutils import Matrix
        o.matrix_world = Matrix.Rotation(math.pi/2,4,'X') @ o.matrix_world
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'assets'/f'{name}.blend'))
    bpy.ops.export_scene.gltf(filepath=str(OUT/f'{name}.glb'),export_format='GLB',export_apply=True)
    print('EXPORTED',name)
