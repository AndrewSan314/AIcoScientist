"""
High-Fidelity Industrial 3D Asset Generator for AIcoScientist Digital Twin.
Generates production-ready, highly detailed .glb models using trimesh:
1. calender.glb - Precision Roll Calendering Machine
2. coater.glb   - Slot-Die Coating Station & Master Unwinder
3. dryer.glb    - Continuous Convection/IR Modular Drying Tunnel
4. mixer.glb    - Planetary High-Shear Vacuum Mixer & Dosing Station
5. cycler.glb   - Industrial 19-inch Multi-Channel Cycler Rack
"""

import math
import numpy as np
from pathlib import Path
import trimesh

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODELS = ROOT / 'frontend/public/models'
DIST_MODELS = ROOT / 'frontend/dist/models'
PUBLIC_MODELS.mkdir(parents=True, exist_ok=True)
DIST_MODELS.mkdir(parents=True, exist_ok=True)

# ----------------- PBR Materials -----------------
def pbr_mat(name, color, metal=0.6, rough=0.3):
    return trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=name,
            baseColorFactor=[*color, 1.0] if len(color) == 3 else color,
            metallicFactor=metal,
            roughnessFactor=rough
        )
    )

MAT_CHROME   = pbr_mat('MirrorChrome', (0.95, 0.96, 0.98), metal=0.98, rough=0.06)
MAT_STEEL    = pbr_mat('MachinedSteel', (0.78, 0.82, 0.85), metal=0.88, rough=0.22)
MAT_FRAME    = pbr_mat('GraphitePowdercoat', (0.12, 0.16, 0.20), metal=0.40, rough=0.38)
MAT_PANEL    = pbr_mat('PearlCleanroomPanel', (0.90, 0.93, 0.94), metal=0.20, rough=0.28)
MAT_TEAL     = pbr_mat('IndustrialTeal', (0.05, 0.42, 0.46), metal=0.55, rough=0.30)
MAT_AMBER    = pbr_mat('SafetyAmber', (0.94, 0.48, 0.12), metal=0.30, rough=0.35)
MAT_RED      = pbr_mat('EmergencyRed', (0.85, 0.12, 0.10), metal=0.40, rough=0.30)
MAT_DARK     = pbr_mat('DarkTrimRubber', (0.06, 0.08, 0.10), metal=0.15, rough=0.65)
MAT_BRASS    = pbr_mat('PolishedBrass', (0.88, 0.72, 0.28), metal=0.85, rough=0.22)
MAT_GLASS    = pbr_mat('InspectionGlass', (0.22, 0.38, 0.45, 0.65), metal=0.10, rough=0.08)
MAT_IR_GLOW  = pbr_mat('GlowingQuartzIR', (1.00, 0.45, 0.08), metal=0.10, rough=0.15)
MAT_SCREEN   = pbr_mat('OledDisplayCyan', (0.03, 0.50, 0.55), metal=0.20, rough=0.20)
MAT_LED_GRN  = pbr_mat('StatusLedGreen', (0.10, 0.90, 0.35), metal=0.10, rough=0.20)
MAT_GOLD     = pbr_mat('GoldContact', (0.92, 0.76, 0.20), metal=0.95, rough=0.18)

# ----------------- Geometry Helpers -----------------
def make_box(extents, pos=(0, 0, 0), mat=MAT_FRAME):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(pos)
    m.visual = mat
    return m

def make_cyl(radius, height, pos=(0, 0, 0), axis='y', sections=32, mat=MAT_STEEL):
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    if axis == 'x':
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
    elif axis == 'z':
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(pos)
    m.visual = mat
    return m

def make_annulus(r_min, r_max, height, pos=(0, 0, 0), axis='y', sections=32, mat=MAT_STEEL):
    m = trimesh.creation.annulus(r_min=r_min, r_max=r_max, height=height, sections=sections)
    if axis == 'x':
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
    elif axis == 'z':
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(pos)
    m.visual = mat
    return m

def add_gauge(scene, name, pos, axis='x', radius=0.14):
    gauge_body = make_cyl(radius=radius, height=0.06, pos=pos, axis=axis, sections=24, mat=MAT_STEEL)
    scene.add_geometry(gauge_body, node_name=f'{name}_body', geom_name=f'{name}_body')
    offset = 0.035 if axis == 'x' else 0.035
    face_pos = list(pos)
    if axis == 'x': face_pos[0] += offset
    elif axis == 'z': face_pos[2] += offset
    gauge_face = make_cyl(radius=radius * 0.88, height=0.01, pos=face_pos, axis=axis, sections=24, mat=MAT_PANEL)
    scene.add_geometry(gauge_face, node_name=f'{name}_face', geom_name=f'{name}_face')
    # Needle
    needle = make_box((0.015, radius * 0.7, 0.005), pos=face_pos, mat=MAT_RED)
    scene.add_geometry(needle, node_name=f'{name}_needle', geom_name=f'{name}_needle')

def add_console(scene, name, pos, heading='calender'):
    # Pedestal & tilted screen console
    x, y, z = pos
    pedestal = make_box((0.2, 1.2, 0.2), pos=(x, y - 0.6, z), mat=MAT_FRAME)
    scene.add_geometry(pedestal, node_name=f'{name}_pedestal', geom_name=f'{name}_pedestal')
    enclosure = make_box((0.82, 0.62, 0.22), pos=(x, y, z), mat=MAT_PANEL)
    scene.add_geometry(enclosure, node_name=f'{name}_enclosure', geom_name=f'{name}_enclosure')
    bezel = make_box((0.68, 0.44, 0.03), pos=(x, y, z + 0.11), mat=MAT_DARK)
    scene.add_geometry(bezel, node_name=f'{name}_bezel', geom_name=f'{name}_bezel')
    display = make_box((0.62, 0.38, 0.02), pos=(x, y, z + 0.12), mat=MAT_SCREEN)
    scene.add_geometry(display, node_name=f'{name}_display', geom_name=f'{name}_display')
    # Emergency Stop
    estop_base = make_cyl(radius=0.06, height=0.03, pos=(x + 0.28, y - 0.18, z + 0.12), axis='z', sections=16, mat=MAT_AMBER)
    estop_btn = make_cyl(radius=0.05, height=0.04, pos=(x + 0.28, y - 0.18, z + 0.15), axis='z', sections=16, mat=MAT_RED)
    scene.add_geometry(estop_base, node_name=f'{name}_estop_base', geom_name=f'{name}_estop_base')
    scene.add_geometry(estop_btn, node_name=f'{name}_estop_btn', geom_name=f'{name}_estop_btn')
    # Stack Light
    stack_base = make_cyl(radius=0.03, height=0.15, pos=(x - 0.32, y + 0.38, z), axis='y', sections=12, mat=MAT_FRAME)
    light_g = make_cyl(radius=0.035, height=0.06, pos=(x - 0.32, y + 0.48, z), axis='y', sections=12, mat=MAT_LED_GRN)
    light_a = make_cyl(radius=0.035, height=0.06, pos=(x - 0.32, y + 0.55, z), axis='y', sections=12, mat=MAT_AMBER)
    light_b = make_cyl(radius=0.035, height=0.06, pos=(x - 0.32, y + 0.62, z), axis='y', sections=12, mat=MAT_TEAL)
    for g in [stack_base, light_g, light_a, light_b]:
        scene.add_geometry(g, node_name=f'{name}_stack_{g.visual.material.name}', geom_name=f'{name}_stack_{g.visual.material.name}')


# ==============================================================================
# 1. BUILD CALENDER GLB
# ==============================================================================
def build_calender_scene():
    scene = trimesh.Scene()
    # Bed & Foundation
    bed = make_box((3.8, 0.48, 3.6), pos=(0, 0.24, 0), mat=MAT_FRAME)
    scene.add_geometry(bed, node_name='Machine bed', geom_name='Machine bed')
    for x in [-1.6, 1.6]:
        for z in [-1.5, 1.5]:
            foot = make_box((0.42, 0.16, 0.42), pos=(x, 0.08, z), mat=MAT_DARK)
            stud = make_cyl(radius=0.08, height=0.2, pos=(x, 0.18, z), axis='y', sections=16, mat=MAT_STEEL)
            scene.add_geometry(foot, node_name=f'Foot_{x}_{z}', geom_name=f'Foot_{x}_{z}')
            scene.add_geometry(stud, node_name=f'Stud_{x}_{z}', geom_name=f'Stud_{x}_{z}')
    for z in [-1.81, 1.81]:
        fascia = make_box((3.5, 0.28, 0.04), pos=(0, 0.26, z), mat=MAT_PANEL)
        scene.add_geometry(fascia, node_name=f'Fascia_{z}', geom_name=f'Fascia_{z}')

    # Twin Portal Upright Columns
    for z in [-1.45, 1.45]:
        for x in [-0.92, 0.92]:
            col = make_box((0.36, 3.8, 0.52), pos=(x, 2.3, z), mat=MAT_FRAME)
            rail = make_box((0.10, 3.2, 0.06), pos=(x, 2.3, z + (0.27 if z < 0 else -0.27)), mat=MAT_STEEL)
            scene.add_geometry(col, node_name=f'Portal column {x} {z}', geom_name=f'Portal column {x} {z}')
            scene.add_geometry(rail, node_name=f'Ground linear slide {x} {z}', geom_name=f'Ground linear slide {x} {z}')
        # Top Crosshead Bridge
        bridge = make_box((2.35, 0.52, 0.62), pos=(0, 4.25, z), mat=MAT_FRAME)
        scene.add_geometry(bridge, node_name=f'Loading bridge {z}', geom_name=f'Loading bridge {z}')

        # Hydraulic Actuators
        ram = make_cyl(radius=0.22, height=0.68, pos=(0, 4.65, z), axis='y', sections=24, mat=MAT_TEAL)
        piston = make_cyl(radius=0.12, height=0.60, pos=(0, 4.05, z), axis='y', sections=24, mat=MAT_CHROME)
        scene.add_geometry(ram, node_name=f'Hydraulic loading ram {z}', geom_name=f'Hydraulic loading ram {z}')
        scene.add_geometry(piston, node_name=f'Chrome loading piston {z}', geom_name=f'Chrome loading piston {z}')

        # Bearing Chocks (Top & Bottom)
        for y, r_name in [(2.5, 'top'), (1.5, 'bottom')]:
            chock = make_box((0.84, 0.74, 0.36), pos=(0, y, z), mat=MAT_PANEL)
            flange = make_cyl(radius=0.32, height=0.08, pos=(0, y, z + (0.19 if z > 0 else -0.19)), axis='z', sections=24, mat=MAT_STEEL)
            seal = make_cyl(radius=0.22, height=0.05, pos=(0, y, z + (0.24 if z > 0 else -0.24)), axis='z', sections=24, mat=MAT_DARK)
            scene.add_geometry(chock, node_name=f'Split bearing chock {r_name} {z}', geom_name=f'Split bearing chock {r_name} {z}')
            scene.add_geometry(flange, node_name=f'Flange {r_name} {z}', geom_name=f'Flange {r_name} {z}')
            scene.add_geometry(seal, node_name=f'Seal {r_name} {z}', geom_name=f'Seal {r_name} {z}')
            # Chock fasteners
            for dx in [-0.32, 0.32]:
                for dy in [-0.26, 0.26]:
                    b = make_cyl(radius=0.025, height=0.06, pos=(dx, y + dy, z + (0.19 if z > 0 else -0.19)), axis='z', sections=12, mat=MAT_STEEL)
                    scene.add_geometry(b, node_name=f'Bolt_{r_name}_{z}_{dx}_{dy}', geom_name=f'Bolt_{r_name}_{z}_{dx}_{dy}')

    # Signature Mirror-Polished Precision Rolls (Aligned along Z axis, centered at X=0)
    top_roll = make_cyl(radius=0.48, height=2.35, pos=(0, 2.5, 0), axis='z', sections=48, mat=MAT_CHROME)
    bot_roll = make_cyl(radius=0.48, height=2.35, pos=(0, 1.5, 0), axis='z', sections=48, mat=MAT_CHROME)
    scene.add_geometry(top_roll, node_name='Upper precision roll', geom_name='Upper precision roll')
    scene.add_geometry(bot_roll, node_name='Lower precision roll', geom_name='Lower precision roll')

    # Web Guide Rollers
    for x in [-1.75, 1.75]:
        guide = make_cyl(radius=0.14, height=2.35, pos=(x, 2.0, 0), axis='z', sections=32, mat=MAT_CHROME)
        scene.add_geometry(guide, node_name=f'Web guide roll {x}', geom_name=f'Web guide roll {x}')
        for z in [-1.45, 1.45]:
            bracket = make_box((0.24, 1.25, 0.24), pos=(x, 1.15, z), mat=MAT_FRAME)
            scene.add_geometry(bracket, node_name=f'Guide support {x} {z}', geom_name=f'Guide support {x} {z}')

    # Analog Pressure Gauges & Gap Micrometers
    add_gauge(scene, 'HydraulicPressureGauge1', pos=(-1.0, 3.2, 1.72), axis='z', radius=0.16)
    add_gauge(scene, 'HydraulicPressureGauge2', pos=(1.0, 3.2, 1.72), axis='z', radius=0.16)

    # Electrical Cabinet with Louver Vents
    cabinet = make_box((2.0, 2.2, 0.8), pos=(0, 1.5, -2.4), mat=MAT_PANEL)
    scene.add_geometry(cabinet, node_name='Drive cabinet', geom_name='Drive cabinet')
    for y_v in np.linspace(0.8, 2.2, 10):
        louver = make_box((1.4, 0.03, 0.02), pos=(0, y_v, -2.81), mat=MAT_DARK)
        scene.add_geometry(louver, node_name=f'Louver_{y_v}', geom_name=f'Louver_{y_v}')

    # Operator Console
    add_console(scene, 'CalenderHMI', pos=(2.2, 1.8, 1.85))

    return scene


# ==============================================================================
# 2. BUILD COATER GLB
# ==============================================================================
def build_coater_scene():
    scene = trimesh.Scene()
    # Bed with cleanroom drip tray
    bed = make_box((4.8, 0.48, 3.6), pos=(0, 0.24, 0), mat=MAT_FRAME)
    scene.add_geometry(bed, node_name='Machine bed', geom_name='Machine bed')
    drip_pan = make_box((3.6, 0.06, 2.6), pos=(0, 0.51, 0), mat=MAT_STEEL)
    scene.add_geometry(drip_pan, node_name='Cleanroom drip tray', geom_name='Cleanroom drip tray')

    # Precision Ground Backing Roll (Aligned along Z)
    backing_roll = make_cyl(radius=0.55, height=2.35, pos=(0, 1.8, 0), axis='z', sections=48, mat=MAT_CHROME)
    scene.add_geometry(backing_roll, node_name='Backing roll', geom_name='Backing roll')

    for z in [-1.45, 1.45]:
        ped = make_box((0.95, 0.95, 0.48), pos=(0, 1.1, z), mat=MAT_FRAME)
        chock = make_box((0.72, 0.72, 0.36), pos=(0, 1.8, z), mat=MAT_PANEL)
        scene.add_geometry(ped, node_name=f'Bearing pedestal {z}', geom_name=f'Bearing pedestal {z}')
        scene.add_geometry(chock, node_name=f'Backing roll chock {z}', geom_name=f'Backing roll chock {z}')

        # Die Positioning Towers & Cross Slides
        tower = make_box((0.34, 2.8, 0.34), pos=(0.78, 2.1, z), mat=MAT_FRAME)
        slide = make_box((1.45, 0.24, 0.32), pos=(0.32, 2.88, z), mat=MAT_STEEL)
        carriage = make_box((0.42, 0.44, 0.42), pos=(-0.14, 2.65, z), mat=MAT_TEAL)
        micrometer = make_cyl(radius=0.08, height=0.45, pos=(-0.14, 3.12, z), axis='y', sections=24, mat=MAT_STEEL)
        scene.add_geometry(tower, node_name=f'Die positioning tower {z}', geom_name=f'Die positioning tower {z}')
        scene.add_geometry(slide, node_name=f'Die cross slide {z}', geom_name=f'Die cross slide {z}')
        scene.add_geometry(carriage, node_name=f'Micrometer carriage {z}', geom_name=f'Micrometer carriage {z}')
        scene.add_geometry(micrometer, node_name=f'Micrometer spindle {z}', geom_name=f'Micrometer spindle {z}')

    # Precision Slot-Die Manifold Head
    # Body wedge
    die_body = make_box((0.68, 0.58, 2.42), pos=(-0.25, 2.52, 0), mat=MAT_STEEL)
    die_lip = make_box((0.28, 0.22, 2.38), pos=(-0.04, 2.22, 0), mat=MAT_CHROME)
    scene.add_geometry(die_body, node_name='Slot die manifold', geom_name='Slot die manifold')
    scene.add_geometry(die_lip, node_name='Slot die precision lips', geom_name='Slot die precision lips')

    # Row of 12 micrometer adjustment screws along die face
    for z_sc in np.linspace(-1.05, 1.05, 12):
        screw = make_cyl(radius=0.032, height=0.12, pos=(-0.52, 2.68, z_sc), axis='x', sections=16, mat=MAT_BRASS)
        scene.add_geometry(screw, node_name=f'Die screw {z_sc:.2f}', geom_name=f'Die screw {z_sc:.2f}')

    # Slurry Feed Hose & Metering Pump Cabinet
    feed_pipe = make_cyl(radius=0.065, height=1.6, pos=(0.35, 3.1, -0.9), axis='x', sections=16, mat=MAT_TEAL)
    pump_cab = make_box((1.2, 1.2, 0.7), pos=(1.6, 1.1, -2.1), mat=MAT_PANEL)
    scene.add_geometry(feed_pipe, node_name='Slurry feed hose', geom_name='Slurry feed hose')
    scene.add_geometry(pump_cab, node_name='Metering pump cabinet', geom_name='Metering pump cabinet')

    # Unwinder Reel Section (at X = -4.0)
    for z in [-1.35, 1.35]:
        a_frame1 = make_box((0.22, 2.2, 0.28), pos=(-3.8, 1.1, z), mat=MAT_FRAME)
        scene.add_geometry(a_frame1, node_name=f'Unwinder stanchion {z}', geom_name=f'Unwinder stanchion {z}')
    master_coil = make_cyl(radius=0.68, height=2.2, pos=(-3.8, 1.8, 0), axis='z', sections=36, mat=MAT_STEEL)
    disc_brake = make_cyl(radius=0.42, height=0.08, pos=(-3.8, 1.8, -1.25), axis='z', sections=24, mat=MAT_CHROME)
    scene.add_geometry(master_coil, node_name='Master uncoiler drum', geom_name='Master uncoiler drum')
    scene.add_geometry(disc_brake, node_name='Tension disc brake', geom_name='Tension disc brake')

    # Web Infeed & Outfeed Guides
    for x in [-2.1, 2.1]:
        guide = make_cyl(radius=0.14, height=2.35, pos=(x, 1.84, 0), axis='z', sections=32, mat=MAT_CHROME)
        scene.add_geometry(guide, node_name=f'Web guide roll {x}', geom_name=f'Web guide roll {x}')

    # Console
    add_console(scene, 'CoaterHMI', pos=(2.3, 1.8, 1.85))

    return scene


# ==============================================================================
# 3. BUILD DRYING TUNNEL GLB
# ==============================================================================
def build_dryer_scene():
    scene = trimesh.Scene()
    # 7-Meter Modular Cleanroom Tunnel (X = -3.6 to +3.6)
    tunnel_body = make_box((7.2, 2.4, 2.8), pos=(0, 1.9, 0), mat=MAT_PANEL)
    scene.add_geometry(tunnel_body, node_name='Dryer removable housing', geom_name='Dryer removable housing')

    # Structural perimeter bands & corner trims
    for x in [-3.6, -1.2, 1.2, 3.6]:
        band = make_box((0.14, 2.5, 2.88), pos=(x, 1.9, 0), mat=MAT_FRAME)
        scene.add_geometry(band, node_name=f'Modular partition frame {x}', geom_name=f'Modular partition frame {x}')

    # Support I-Beam Legs
    for x in [-3.2, -1.2, 1.2, 3.2]:
        for z in [-1.2, 1.2]:
            leg = make_box((0.26, 0.72, 0.26), pos=(x, 0.36, z), mat=MAT_FRAME)
            pad = make_box((0.44, 0.12, 0.44), pos=(x, 0.06, z), mat=MAT_DARK)
            scene.add_geometry(leg, node_name=f'Support leg {x} {z}', geom_name=f'Support leg {x} {z}')
            scene.add_geometry(pad, node_name=f'Leveling pad {x} {z}', geom_name=f'Leveling pad {x} {z}')

    # 3 Recessed Inspection Windows with Glowing IR Tubes Inside
    for x in [-2.4, 0.0, 2.4]:
        # Window Frame
        frame = make_box((1.4, 0.9, 0.12), pos=(x, 2.0, 1.42), mat=MAT_STEEL)
        glass = make_box((1.2, 0.72, 0.04), pos=(x, 2.0, 1.44), mat=MAT_GLASS)
        scene.add_geometry(frame, node_name=f'Inspection frame {x}', geom_name=f'Inspection frame {x}')
        scene.add_geometry(glass, node_name=f'Inspection window {x}', geom_name=f'Inspection window {x}')
        # Quartz IR Heating Element Tubes (inside oven)
        for y_ir in [1.75, 2.05, 2.35]:
            quartz = make_cyl(radius=0.035, height=1.6, pos=(x, y_ir, 0.4), axis='x', sections=16, mat=MAT_IR_GLOW)
            reflector = make_box((1.6, 0.08, 0.25), pos=(x, y_ir, 0.25), mat=MAT_CHROME)
            scene.add_geometry(quartz, node_name=f'Quartz IR tube {x} {y_ir}', geom_name=f'Quartz IR tube {x} {y_ir}')
            scene.add_geometry(reflector, node_name=f'IR reflector {x} {y_ir}', geom_name=f'IR reflector {x} {y_ir}')

    # Top Exhaust Chimneys with Centrifugal Blower Housings
    for x in [-1.8, 1.8]:
        blower = make_cyl(radius=0.45, height=0.48, pos=(x, 3.4, 0), axis='z', sections=24, mat=MAT_FRAME)
        motor = make_cyl(radius=0.22, height=0.55, pos=(x, 3.4, 0.45), axis='z', sections=24, mat=MAT_TEAL)
        duct = make_cyl(radius=0.32, height=1.2, pos=(x, 4.0, 0), axis='y', sections=24, mat=MAT_STEEL)
        damper = make_cyl(radius=0.36, height=0.08, pos=(x, 4.6, 0), axis='y', sections=24, mat=MAT_DARK)
        scene.add_geometry(blower, node_name=f'Centrifugal blower {x}', geom_name=f'Centrifugal blower {x}')
        scene.add_geometry(motor, node_name=f'Blower motor {x}', geom_name=f'Blower motor {x}')
        scene.add_geometry(duct, node_name=f'Exhaust chimney {x}', geom_name=f'Exhaust chimney {x}')
        scene.add_geometry(damper, node_name=f'Duct damper {x}', geom_name=f'Duct damper {x}')

    # Cleanroom Access Latch Handles
    for x in [-2.4, 0.0, 2.4]:
        latch = make_box((0.08, 0.24, 0.06), pos=(x + 0.8, 2.0, 1.44), mat=MAT_CHROME)
        scene.add_geometry(latch, node_name=f'Door latch {x}', geom_name=f'Door latch {x}')

    return scene


# ==============================================================================
# 4. BUILD MIXER & FORMULATION GLB
# ==============================================================================
def build_mixer_scene():
    scene = trimesh.Scene()
    # Machine Base
    base = make_box((3.4, 0.45, 3.4), pos=(0, 0.22, 0), mat=MAT_FRAME)
    scene.add_geometry(base, node_name='Mixer base', geom_name='Mixer base')

    # Cylindrical Stainless Vacuum Mixing Vessel
    vessel = make_cyl(radius=1.2, height=2.4, pos=(0, 1.6, 0), axis='y', sections=36, mat=MAT_STEEL)
    flange = make_annulus(r_min=1.18, r_max=1.35, height=0.08, pos=(0, 2.8, 0), axis='y', sections=36, mat=MAT_STEEL)
    dome = make_cyl(radius=1.2, height=0.35, pos=(0, 2.95, 0), axis='y', sections=36, mat=MAT_STEEL)
    scene.add_geometry(vessel, node_name='Mixer vessel cutaway shell', geom_name='Mixer vessel cutaway shell')
    scene.add_geometry(flange, node_name='Vessel clamp flange', geom_name='Vessel clamp flange')
    scene.add_geometry(dome, node_name='Domed vacuum cover', geom_name='Domed vacuum cover')

    # Dual Sight Glasses (Inspection Windows)
    for z_ang, z_name in [(np.pi * 0.25, 'view_a'), (np.pi * 0.75, 'view_b')]:
        sx = 1.15 * np.cos(z_ang)
        sz = 1.15 * np.sin(z_ang)
        port = make_cyl(radius=0.18, height=0.12, pos=(sx, 1.8, sz), axis='y', sections=24, mat=MAT_BRASS)
        glass = make_cyl(radius=0.14, height=0.04, pos=(sx, 1.8, sz), axis='y', sections=24, mat=MAT_GLASS)
        scene.add_geometry(port, node_name=f'Sight glass port {z_name}', geom_name=f'Sight glass port {z_name}')
        scene.add_geometry(glass, node_name=f'Sight glass lens {z_name}', geom_name=f'Sight glass lens {z_name}')

    # Overhead Drive Motor & Planetary Gearbox
    motor = make_cyl(radius=0.55, height=1.4, pos=(0, 3.85, 0), axis='y', sections=28, mat=MAT_FRAME)
    gearbox = make_box((1.1, 0.45, 1.1), pos=(0, 3.25, 0), mat=MAT_TEAL)
    scene.add_geometry(motor, node_name='Overhead drive motor', geom_name='Overhead drive motor')
    scene.add_geometry(gearbox, node_name='Planetary gearbox', geom_name='Planetary gearbox')

    # Agitator Drive Shaft
    shaft = make_cyl(radius=0.14, height=2.0, pos=(0, 1.9, 0), axis='y', sections=20, mat=MAT_STEEL)
    scene.add_geometry(shaft, node_name='Agitator drive shaft', geom_name='Agitator drive shaft')

    # Vacuum Port & Pressure Gauge
    vac_pipe = make_cyl(radius=0.08, height=0.9, pos=(0.85, 3.1, 0), axis='x', sections=16, mat=MAT_STEEL)
    scene.add_geometry(vac_pipe, node_name='Vacuum line', geom_name='Vacuum line')
    add_gauge(scene, 'VacuumGauge', pos=(1.35, 3.1, 0), axis='x', radius=0.16)

    # Slurry Discharge Line leading toward coater
    slurry_pipe = make_cyl(radius=0.08, height=2.4, pos=(1.6, 0.8, 0), axis='x', sections=16, mat=MAT_TEAL)
    scene.add_geometry(slurry_pipe, node_name='Slurry transfer pipe', geom_name='Slurry transfer pipe')

    # Gravimetric Dosing Formulation Station (at X = -4.5)
    bench = make_box((3.2, 0.42, 2.8), pos=(-4.5, 0.21, 0), mat=MAT_FRAME)
    scale = make_box((2.6, 0.16, 1.2), pos=(-4.5, 0.48, 0.4), mat=MAT_TEAL)
    scene.add_geometry(bench, node_name='Formulation bench', geom_name='Formulation bench')
    scene.add_geometry(scale, node_name='Gravimetric scale', geom_name='Gravimetric scale')

    # 3 Conical Powder Hoppers
    for idx, (dx, h_mat, h_name) in enumerate([
        (-5.3, MAT_PANEL, 'Active material vessel'),
        (-4.5, MAT_PANEL, 'Conductive additive vessel'),
        (-3.7, MAT_STEEL, 'Binder vessel')
    ]):
        cone = trimesh.creation.cone(radius=0.42, height=1.6, sections=24)
        cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        cone.apply_translation((dx, 1.8, -0.4))
        cone.visual = h_mat
        top_rim = make_cyl(radius=0.44, height=0.08, pos=(dx, 2.6, -0.4), axis='y', sections=24, mat=MAT_STEEL)
        scene.add_geometry(cone, node_name=h_name, geom_name=h_name)
        scene.add_geometry(top_rim, node_name=f'{h_name} rim', geom_name=f'{h_name} rim')

    # Pneumatic Transfer Chute
    chute = make_cyl(radius=0.09, height=3.6, pos=(-2.4, 1.8, 0), axis='x', sections=16, mat=MAT_STEEL)
    scene.add_geometry(chute, node_name='Pneumatic transfer chute', geom_name='Pneumatic transfer chute')

    return scene


# ==============================================================================
# 5. BUILD CYCLER & METROLOGY RACK GLB
# ==============================================================================
def build_cycler_scene():
    scene = trimesh.Scene()
    # 19-Inch 42U Server Rack Enclosure
    rack = make_box((2.4, 4.4, 1.8), pos=(0, 2.2, 0), mat=MAT_FRAME)
    scene.add_geometry(rack, node_name='Cycler server rack', geom_name='Cycler server rack')

    # Side Perforated Ventilation Mesh Panels
    for z in [-0.91, 0.91]:
        vent_panel = make_box((2.1, 4.0, 0.02), pos=(0, 2.2, z), mat=MAT_DARK)
        scene.add_geometry(vent_panel, node_name=f'Side vent panel {z}', geom_name=f'Side vent panel {z}')

    # 6 Individual Rack-Mounted Cycler Sub-Chassis Modules
    for row in range(6):
        y_mod = 0.85 + row * 0.54
        # Brushed aluminum chassis faceplate
        chassis = make_box((2.15, 0.46, 0.06), pos=(0, y_mod, 0.92), mat=MAT_PANEL)
        scene.add_geometry(chassis, node_name=f'Cycler module {row}', geom_name=f'Cycler module {row}')
        # Pull handles
        for x_h in [-0.95, 0.95]:
            handle = make_cyl(radius=0.02, height=0.32, pos=(x_h, y_mod, 0.98), axis='y', sections=12, mat=MAT_CHROME)
            scene.add_geometry(handle, node_name=f'Handle_{row}_{x_h}', geom_name=f'Handle_{row}_{x_h}')
        # Mini OLED Channel Display
        mini_disp = make_box((0.45, 0.18, 0.02), pos=(-0.45, y_mod, 0.96), mat=MAT_SCREEN)
        scene.add_geometry(mini_disp, node_name=f'MiniDisplay_{row}', geom_name=f'MiniDisplay_{row}')
        # Row of 8 Channel LEDs
        for col_idx, col_x in enumerate(np.linspace(0.05, 0.85, 8)):
            c_mat = MAT_LED_GRN if (row + col_idx) % 3 != 0 else MAT_AMBER
            led = make_cyl(radius=0.025, height=0.02, pos=(col_x, y_mod + 0.08, 0.96), axis='z', sections=12, mat=c_mat)
            bnc = make_cyl(radius=0.03, height=0.03, pos=(col_x, y_mod - 0.08, 0.96), axis='z', sections=12, mat=MAT_GOLD)
            scene.add_geometry(led, node_name=f'LED_{row}_{col_idx}', geom_name=f'LED_{row}_{col_idx}')
            scene.add_geometry(bnc, node_name=f'BNC_{row}_{col_idx}', geom_name=f'BNC_{row}_{col_idx}')

    # Slide-Out Coin-Cell Test Fixture Shelf (at Y = 1.4)
    shelf = make_box((2.1, 0.10, 0.85), pos=(0, 1.4, 1.25), mat=MAT_STEEL)
    scene.add_geometry(shelf, node_name='Test fixture shelf', geom_name='Test fixture shelf')
    for x_c in np.linspace(-0.8, 0.8, 6):
        coin_cell = make_cyl(radius=0.12, height=0.04, pos=(x_c, 1.48, 1.25), axis='y', sections=24, mat=MAT_GOLD)
        bracket = make_box((0.3, 0.06, 0.3), pos=(x_c, 1.46, 1.25), mat=MAT_DARK)
        scene.add_geometry(bracket, node_name=f'Cell bracket {x_c}', geom_name=f'Cell bracket {x_c}')
        scene.add_geometry(coin_cell, node_name=f'Coin cell {x_c}', geom_name=f'Coin cell {x_c}')

    # Vertical Cable Duct Trays
    for x_tray in [-1.15, 1.15]:
        tray = make_box((0.08, 3.8, 0.25), pos=(x_tray, 2.2, 0.88), mat=MAT_TEAL)
        scene.add_geometry(tray, node_name=f'Cable tray {x_tray}', geom_name=f'Cable tray {x_tray}')

    # Overhead Status Beacon Tower
    add_console(scene, 'CyclerHMI', pos=(1.5, 2.0, 1.4))

    return scene


# ==============================================================================
# MAIN EXPORT
# ==============================================================================
def main():
    generators = [
        ('calender.glb', build_calender_scene),
        ('coater.glb', build_coater_scene),
        ('dryer.glb', build_dryer_scene),
        ('mixer.glb', build_mixer_scene),
        ('cycler.glb', build_cycler_scene),
    ]

    for filename, builder in generators:
        print(f'Building {filename}...')
        sc = builder()
        data = sc.export(file_type='glb')
        pub_path = PUBLIC_MODELS / filename
        dist_path = DIST_MODELS / filename
        with open(pub_path, 'wb') as f:
            f.write(data)
        with open(dist_path, 'wb') as f:
            f.write(data)
        print(f'[OK] Wrote {filename} ({len(data):,} bytes) to public & dist')

if __name__ == '__main__':
    main()
