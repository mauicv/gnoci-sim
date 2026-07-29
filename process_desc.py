#!/usr/bin/env python3
"""
Copy onshape-to-robot output to src desc and inject sensors.

Adds:
  - jointpos sensor for every hinge joint
  - 2 touch sites + 2 touch sensors per foot body (toe and heel)

Usage:
    python process_desc.py [--src desc/robot.xml] [--dst src/gnoci_gym/desc/gnoci.xml]
"""

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET

# ── tuneable constants ────────────────────────────────────────────────────────

SRC          = "onshape_export/robot.xml"
DST          = "src/gnoci_gym/desc/gnoci.xml"
ASSETS_SRC   = "onshape_export/assets"
ASSETS_DST   = "src/gnoci_gym/desc/assets"
SCENE_SRC    = "onshape_export/scene.xml"
SCENE_DST    = "src/gnoci_gym/desc/scene.xml"

# Bodies whose name contains this string are treated as feet
FOOT_PATTERN = "foot"

# Toe/heel sites are placed at ±TOUCH_OFFSET along the X axis of the foot's
# local frame, relative to the collision geom centre.  Adjust after visualising.
TOUCH_OFFSET = 0.02   # metres

SITE_SIZE = 0.01      # metres

ACTUATOR_CLASS = "miuzei_25kg"

# Joint limits as (lo, hi) offsets from the default position, in units of π rad.
# Absolute range written to XML = (default + offset) * π.
_JOINT_DEFAULTS: dict[str, float] = {
    'head__left_yoke':              0,
    'left_yoke__hip':               0,
    'left_hip__upper_leg':          0.3 * 0.75,
    'left_upper_leg__lower_leg':    -0.6 * 0.75,
    'left_lower_leg__foot':         -0.3 * 0.75,
    'head__right_yoke':             0,
    'right_yoke__hip':              0,
    'right_hip__upper_leg':         0.3 * 0.75,
    'right_upper_leg__lower_leg':   -0.6 * 0.75,
    'right_lower_leg__foot':        -0.3 * 0.75,
}

_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    'head__left_yoke':            (-0.287,  0.222),
    'left_yoke__hip':             (-0.135,  0.216),
    'left_hip__upper_leg':        (-0.260,  0.443),
    'left_upper_leg__lower_leg':  (-0.300,  0.428),
    'left_lower_leg__foot':       (-0.306,  0.495),
    'head__right_yoke':           (-0.265,  0.226),
    'right_yoke__hip':            (-0.135,  0.209),
    'right_hip__upper_leg':       (-0.274,  0.455),
    'right_upper_leg__lower_leg': (-0.280,  0.446),
    'right_lower_leg__foot':      (-0.287,  0.526),
}

# Left-side joints whose axis and range are flipped so that positive motion
# means the same physical direction as the corresponding right-side joint.
LEFT_JOINTS = {
    "head__left_yoke",
    "left_yoke__hip",
    "left_hip__upper_leg",
    "left_upper_leg__lower_leg",
    "left_lower_leg__foot",
}

# Touch sensor site size (metres).  Sites sit ~0.035 m above the floor contact
# surface; 0.04 m radius reaches the floor (0.035 m away) but not the opposite
# site (~0.063 m away), giving clean front/back separation.
C_SENSE_SITE_SIZE = 0.04

# Meshes whose collision geoms should be stripped — sensors, servos, and
# electronics that are rigidly mounted and only cause spurious contact forces.
# Structural parts (frames, leg shells, feet) are intentionally absent.
NO_COLLISION_MESHES: set[str] = {
    # ── head electronics / cosmetics ──────────────────────────────────────────
    "head_servo_left",    "head_servo_right",
    "head_battery",
    "head_fan",
    "head_r_sense_left",  "head_r_sense_right",
    "head_button",
    "head_voltmeter",
    "head_powerboard",
    "head_r_pi",
    "head_lid",
    "head_base",
    "head_main",
    "head_midframe",
    # ── yoke sensors / servos ─────────────────────────────────────────────────
    "left_yoke_r_sense",  "right_yoke_r_sense",
    "left_yoke_servo",    "right_yoke_servo",
    "left_yoke_upper_frame", "right_yoke_upper_frame",
    "left_yoke_lower_frame", "right_yoke_lower_frame",
    # ── hip sensors / servos ──────────────────────────────────────────────────
    "left_hip_servo",     "right_hip_servo",
    "left_hip_r_sense",   "right_hip_r_sense",
    # ── upper-leg sensors / servos ────────────────────────────────────────────
    "left_upper_leg_servo",    "right_upper_leg_servo",
    "left_upper_leg_r_sense",  "right_upper_leg_r_sense",
    # ── lower-leg sensors / servos ────────────────────────────────────────────
    "left_lower_leg_servo",    "right_lower_leg_servo",
    "left_lower_leg_r_sense",  "right_lower_leg_r_sense",
    "left_lower_leg_c_sense_1", "left_lower_leg_c_sense_2",
    "right_lower_leg_c_sense_1", "right_lower_leg_c_sense_2",
}

PART_MASSES: dict[str, float] = {
    "servo": 0.063,
    "button": 0.012,
    "r_pi": 0.075,
    "battery": 0.089,
    "r_sense": 0.00001,
    "c_sense": 0.00001,
    "voltmeter": 0.003,
    "fan": 0.006,
    "power_board": 0.013,
    "servo_horn": 0.003,
    "head_base": 0.027,
    "head_main": 0.08,
    "head_top": 0.08,
    "head_lid": 0.011,
    "head_midframe": 0.037,
    "yoke_upper_frame": 0.01,
    "yoke_lower_frame": 0.02,
    "hip_back": 0.023,
    "hip_front": 0.036,
    "upper_leg": 0.067,
    "lower_leg": 0.044,
    "foot_base": 0.023,
    "left_foot_right_side": 0.024,
    "left_foot_left_side": 0.037,
    "right_foot_left_side": 0.024,
    "right_foot_right_side": 0.037,
}

JOINT_ATTRS = {
    'damping': 0.17889452739994438,
    'frictionloss': 0.0072408653310164755,
    'armature': 0.014643797951585229,
}

def _indent(elem, level=0):
    """Add pretty-print indentation in-place (Python < 3.9 compat)."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad
    if not level:
        elem.tail = "\n"


# ── main ──────────────────────────────────────────────────────────────────────

def process(src, dst):
    tree = ET.parse(src)
    root = tree.getroot()

    # ── freejoint on root body ────────────────────────────────────────────────
    worldbody = root.find("worldbody")
    root_body = worldbody.find("body")
    freejoint = ET.Element("freejoint")
    freejoint.set("name", "root")
    root_body.insert(0, freejoint)

    # ── joint position sensors ────────────────────────────────────────────────
    sensor_el = root.find("sensor")
    if sensor_el is None:
        sensor_el = ET.SubElement(root, "sensor")

    joints_instrumented = []
    c_sense_sites = []
    for joint in root.iter("joint"):
        jtype = joint.get("type", "hinge")
        if jtype == "free":
            continue
        name = joint.get("name")
        if not name:
            continue
        s = ET.SubElement(sensor_el, "jointpos")
        s.set("name", f"{name}-pos")
        s.set("joint", name)
        joints_instrumented.append(name)

    for site in root.iter("site"):
        name = site.get("name", "")
        if "c_sense" in name:
            site.set("size", str(C_SENSE_SITE_SIZE))
            c_sense_sites.append(name)
            t = ET.SubElement(sensor_el, "touch")
            t.set("name", f"{name}-touch")
            t.set("site", name)

    for tag, sname in [("gyro", "imu-gyro"), ("accelerometer", "imu-acc")]:
        s = ET.SubElement(sensor_el, tag)
        s.set("name", sname)
        s.set("site", "imu")

    for child in root.findall('compiler'):
        child.attrib['meshdir'] = str(os.path.join('src', 'gnoci_gym', 'desc', 'assets'))

    # ── actuator class ───────────────────────────────────────────────────────
    actuator_el = root.find("actuator")
    if actuator_el is not None:
        for actuator in actuator_el:
            actuator.set("class", ACTUATOR_CLASS)
            actuator.attrib["inheritrange"] = "1"
            actuator.attrib.pop("ctrlrange", None)

    # ── flip left-side joint axes and ranges ─────────────────────────────────
    for joint in root.iter("joint"):
        if joint.get("name") not in LEFT_JOINTS:
            continue
        axis = joint.get("axis", "0 0 1")
        joint.set("axis", " ".join(str(-float(x)) for x in axis.split()))
        rng = joint.get("range")
        if rng:
            lo, hi = map(float, rng.split())
            joint.set("range", f"{-hi} {-lo}")

    # ── set joint ranges ─────────────────────────────────────────────────────
    # Range = (default + relative_offset) * π, applied after axis flip so the
    # absolute qpos bounds are consistent for both left and right joints.
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name not in _JOINT_LIMITS:
            continue
        default = _JOINT_DEFAULTS[name]
        lo_rel, hi_rel = _JOINT_LIMITS[name]
        lo = lo_rel * math.pi
        hi = hi_rel * math.pi
        joint.set("range", f"{lo:.10f} {hi:.10f}")
        joint.set("ref", f"{-default * math.pi:.10f}")

    # ── strip collision geoms for cosmetic / sensor parts ────────────────────
    removed = 0
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("class") == "collision" and geom.get("mesh") in NO_COLLISION_MESHES:
                body.remove(geom)
                removed += 1

    # ── remove explicit inertials (let MuJoCo compute from geom masses) ─────
    for body in root.iter("body"):
        for inertial in list(body.findall("inertial")):
            body.remove(inertial)

    # ── assign masses from PART_MASSES ──────────────────────────────────────
    masses_set = 0
    for geom in root.iter("geom"):
        mesh = geom.get("mesh", "")
        for key, mass in PART_MASSES.items():
            if key in mesh:
                geom.set("mass", str(mass))
                masses_set += 1
                break

    # ── IMU site on head_base ────────────────────────────────────────────────
    for body in root.iter("body"):
        if body.get("name") == "head_base":
            imu = ET.SubElement(body, "site")
            imu.set("group", "3")
            imu.set("name", "imu")
            imu.set("pos", "0.0202673 -0.0559394 0.0971252")
            imu.set("quat", "0 1 0 0")
            break

    # ── set joint dynamics attributes (damping/frictionloss/armature) ────────
    joints_tuned = 0
    for joint in root.iter("joint"):
        if joint.get("type", "hinge") == "free":
            continue
        if not joint.get("name"):
            continue
        for attr, value in JOINT_ATTRS.items():
            joint.set(attr, str(value))
        joints_tuned += 1

    # ── write output ──────────────────────────────────────────────────────────
    _indent(root)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tree.write(dst, encoding="unicode", xml_declaration=True)

    print(f"Written: {dst}")
    print(f"  Joints ({len(joints_instrumented)}): {', '.join(joints_instrumented)}")
    print(f"  Collision geoms removed: {removed}")
    print(f"  Geom masses assigned:   {masses_set}")
    print(f"  Touch sensors added:    {len(c_sense_sites)} ({', '.join(c_sense_sites)})")


def copy_assets(assets_src, assets_dst):
    if os.path.isdir(assets_dst):
        shutil.rmtree(assets_dst)
    shutil.copytree(assets_src, assets_dst)
    count = sum(len(f) for _, _, f in os.walk(assets_dst))
    print(f"Copied:  {assets_src} -> {assets_dst} ({count} files)")


def copy_scene(scene_src, scene_dst):
    shutil.copy2(scene_src, scene_dst)
    print(f"Copied:  {scene_src} -> {scene_dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src",        default=SRC,        help="Source robot XML")
    parser.add_argument("--dst",        default=DST,        help="Destination robot XML")
    parser.add_argument("--assets-src", default=ASSETS_SRC, help="Source assets dir")
    parser.add_argument("--assets-dst", default=ASSETS_DST, help="Destination assets dir")
    args = parser.parse_args()

    process(args.src, args.dst)
    copy_assets(args.assets_src, args.assets_dst)
