#!/usr/bin/env python3
"""
Copy onshape-to-robot output to src desc and inject sensors.

Adds:
  - jointpos sensor for every hinge joint
  - a touch sensor per foot c_sense site, with the site recentred onto the
    contact-sphere group it senses (front / back) and resized

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

# Foot contact spheres: one per corner of the sole's flat bottom rectangle,
# positioned in the foot body's local frame.  Initial values were measured
# from the sole meshes at the default pose (edges flush with the mesh
# footprint, bottoms level with the sole); adjust after visualising.
FOOT_SPHERE_RADIUS = 0.006  # metres

# Contact softness, tuned against chatter (make/break flicker and touch-force
# jitter) over seeded random-action rollouts.  solref = (timeconst, dampratio):
# dampratio 2 = overdamped, no touchdown rebound.  solimp width 6 mm so force
# builds gradually with penetration instead of switching on/off within the
# default 1 mm.  Static penetration ~2 mm when standing.
FOOT_SPHERE_SOLREF = "0.03 2"
FOOT_SPHERE_SOLIMP = "0.9 0.95 0.006 0.5 2"

# name="left_foot_front_inner_sphere" pos="-0.07810816 -0.05216554 0.00500100"
# name="left_foot_back_inner_sphere" pos="0.07810813 -0.05216554 0.00500100"
# name="left_foot_front_outer_sphere" pos="-0.07810816 -0.05216554 0.06300101"
# name="left_foot_back_outer_sphere" pos="0.07810813 -0.05216554 0.06300101"

# name="right_foot_front_outer_sphere" pos="-0.07810816 0.05216554 0.00500100"
# name="right_foot_back_outer_sphere" pos="0.07810813 0.05216554 0.00500100"
# name="right_foot_front_inner_sphere" pos="-0.07810816 0.05216554 0.06300101"
# name="right_foot_back_inner_sphere" pos="0.07810813 0.05216554 0.06300101"

FOOT_SPHERES: dict[str, dict[str, str]] = {
    "left_foot_base": {
        "left_foot_front_inner_sphere": "-0.07810816 -0.05216554 0.00500100",
        "left_foot_back_inner_sphere":  "0.07810813 -0.05216554 0.00500100",
        "left_foot_front_outer_sphere": "-0.07810816 -0.05216554 0.06300101",
        "left_foot_back_outer_sphere":  "0.07810813 -0.05216554 0.06300101",
    },
    "right_foot_base": {
        "right_foot_front_outer_sphere": "-0.07810816 0.05216554 0.00500100",
        "right_foot_back_outer_sphere": "0.07810813 0.05216554 0.00500100",
        "right_foot_front_inner_sphere": "-0.07810816 0.05216554 0.06300101",
        "right_foot_back_inner_sphere": "0.07810813 0.05216554 0.06300101",
    },
}

# Touch sensor site size (metres).  Each c_sense site is recentred onto the
# midpoint of its contact-sphere pair (see _c_sense_site_targets); the two
# spheres in a pair are ~0.058 m apart, so a 0.04 m radius comfortably contains
# them while staying well clear of the opposite pair (~0.156 m away), giving
# clean front/back separation.
C_SENSE_SITE_SIZE = 0.04

# Which c_sense site senses which contact-sphere group.  Sites and spheres
# share the foot body's local frame, so a site's pos can be set straight to the
# centroid of its group.
_C_SENSE_BODY_SIDE = {"left_foot_base": "left", "right_foot_base": "right"}
_C_SENSE_GROUPS = (("front", "forward"), ("back", "back"))


def _c_sense_site_targets() -> dict[str, str]:
    """{site_name: "x y z"} placing each foot touch site at the centroid of the
    contact spheres it should register (forward_* -> the 'front' spheres,
    back_* -> the 'back' spheres of the same foot)."""
    targets: dict[str, str] = {}
    for body_name, spheres in FOOT_SPHERES.items():
        side = _C_SENSE_BODY_SIDE.get(body_name)
        if side is None:
            continue
        for tag, prefix in _C_SENSE_GROUPS:
            pts = [
                tuple(float(v) for v in pos.split())
                for name, pos in spheres.items()
                if tag in name
            ]
            if not pts:
                continue
            c = [sum(axis) / len(pts) for axis in zip(*pts)]
            targets[f"{prefix}_{side}_c_sense"] = "{:.8f} {:.8f} {:.8f}".format(*c)
    return targets


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
    "damping": 0.1587636966237435,
    "frictionloss": 0.006426057652042094,
    "armature": 0.012995945315907826
}

# Gear/joint mechanical backlash ("slack"), modeled as a second, unactuated
# hinge joint sharing each main joint's axis (see add_slack_joints below).
# Placeholder magnitude — retune once measured on the real robot.
SLACK_RANGE_RAD = math.radians(0.5)   # +/- per side; ~1 deg total play
SLACK_DAMPING = 1e-4                  # tiny — only to damp numerical rattle against the hard limit stops (frictionloss stays exactly 0)
SLACK_SUFFIX = "__slack"

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

    c_sense_targets = _c_sense_site_targets()
    for site in root.iter("site"):
        name = site.get("name", "")
        if "c_sense" in name:
            site.set("size", str(C_SENSE_SITE_SIZE))
            if name in c_sense_targets:
                site.set("pos", c_sense_targets[name])
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

    # ── disable all mesh contacts ────────────────────────────────────────────
    # Contacts happen only between the floor and the foot corner spheres
    # added by _add_foot_spheres below.
    for geom in root.iter("geom"):
        if geom.get("class") == "collision":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

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

    # ── add unactuated backlash ("slack") joints, one per existing hinge ─────
    # A second <joint> on the SAME body as the main joint, not a new body:
    # MuJoCo composes multiple joints on one body as a serial chain, and since
    # both share the same axis and the (default) origin pos, their rotations
    # commute and sum exactly — qpos_main + qpos_slack is the true
    # externally-observed angle (see env.py _get_joint_positions()). If any
    # joint here ever gains an explicit `pos` upstream, this assumption breaks
    # and the slack joint's `pos` must be copied too.
    parent_map = {child: parent for parent in root.iter() for child in parent}
    target_joints = [
        j for j in root.iter("joint")
        if j.get("type", "hinge") != "free" and j.get("name")
    ]
    slack_count = 0
    for joint in target_joints:
        name = joint.get("name")
        parent = parent_map[joint]
        slack = ET.Element("joint")
        slack.set("name", f"{name}{SLACK_SUFFIX}")
        slack.set("type", "hinge")
        slack.set("axis", joint.get("axis", "0 0 1"))
        slack.set("range", f"{-SLACK_RANGE_RAD:.10f} {SLACK_RANGE_RAD:.10f}")
        slack.set("damping", str(SLACK_DAMPING))
        slack.set("frictionloss", "0")
        slack.set("armature", "0")
        parent.insert(list(parent).index(joint) + 1, slack)
        slack_count += 1

    # ── foot corner contact spheres ──────────────────────────────────────────
    # contype=2 conaffinity=1: collide with the floor (default mask,
    # contype=1) but not with each other or the robot.
    sphere_names = []
    for body in root.iter("body"):
        for name, pos in FOOT_SPHERES.get(body.get("name"), {}).items():
            gel = ET.SubElement(body, "geom")
            gel.set("name", name)
            gel.set("type", "sphere")
            gel.set("size", str(FOOT_SPHERE_RADIUS))
            gel.set("pos", pos)
            gel.set("solref", FOOT_SPHERE_SOLREF)
            gel.set("solimp", FOOT_SPHERE_SOLIMP)
            gel.set("priority", "1")   # sphere solref/solimp win over the floor's
            gel.set("group", "3")
            gel.set("contype", "2")
            gel.set("conaffinity", "1")
            gel.set("mass", "0")
            sphere_names.append(name)

    # ── write output ──────────────────────────────────────────────────────────
    _indent(root)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tree.write(dst, encoding="unicode", xml_declaration=True)

    print(f"Written: {dst}")
    print(f"  Joints ({len(joints_instrumented)}): {', '.join(joints_instrumented)}")
    print(f"  Collision geoms removed: {removed}")
    print(f"  Geom masses assigned:   {masses_set}")
    print(f"  Foot contact spheres:   {len(sphere_names)} (all mesh contacts disabled)")
    print(f"  Touch sensors added:    {len(c_sense_sites)} ({', '.join(c_sense_sites)})")
    print(f"  Slack joints added:     {slack_count}")


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
