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

ACTUATOR_CLASS = "hps0618sg"  # class added to every actuator

# ── helpers ───────────────────────────────────────────────────────────────────

def _geom_pos(body):
    """Return the pos of the first collision geom in *body*, or (0, 0, 0)."""
    for geom in body.findall("geom"):
        if geom.get("class") == "collision":
            raw = geom.get("pos", "0 0 0")
            return [float(v) for v in raw.split()]
    return [0.0, 0.0, 0.0]

def _add_site(body, name, pos):
    site = ET.SubElement(body, "site")
    site.set("name", name)
    site.set("size", str(SITE_SIZE))
    site.set("pos", f"{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}")
    return site


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

    # ── collision friction ────────────────────────────────────────────────────
    for default in root.iter("default"):
        if default.get("class") == "collision":
            geom = default.find("geom")
            if geom is None:
                geom = ET.SubElement(default, "geom")
            geom.set("friction", "1.0 0.005 0.0001")
            break

    # ── joint position sensors ────────────────────────────────────────────────
    sensor_el = root.find("sensor")
    if sensor_el is None:
        sensor_el = ET.SubElement(root, "sensor")

    joints_instrumented = []
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

    # ── foot touch sensors ────────────────────────────────────────────────────
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in source XML.")

    feet_instrumented = []
    for body in worldbody.iter("body"):
        name = body.get("name", "")
        if FOOT_PATTERN not in name:
            continue
        ref = _geom_pos(body)

        toe_pos  = [0.01,0.02,-0.03]
        heel_pos = [-0.01,0.02,-0.03]

        _add_site(body, f"{name}-toe",  toe_pos)
        _add_site(body, f"{name}-heel", heel_pos)

        for suffix in ("toe", "heel"):
            t = ET.SubElement(sensor_el, "touch")
            t.set("name", f"{name}-{suffix}-contact")
            t.set("site", f"{name}-{suffix}")

        feet_instrumented.append(name)

    for child in root.findall('compiler'):
        child.attrib['meshdir'] = str(os.path.join('src', 'gnoci_gym', 'desc', 'assets'))


    # ── actuator class ───────────────────────────────────────────────────────
    actuator_el = root.find("actuator")
    if actuator_el is not None:
        for actuator in actuator_el:
            actuator.set("class", ACTUATOR_CLASS)
            actuator.attrib.pop("inheritrange", None)  # conflicts with ctrlrange
            actuator.set("ctrlrange", "-1.5708 1.5708")

    # ── write output ──────────────────────────────────────────────────────────
    _indent(root)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tree.write(dst, encoding="unicode", xml_declaration=True)

    print(f"Written: {dst}")
    print(f"  Joints ({len(joints_instrumented)}): {', '.join(joints_instrumented)}")
    print(f"  Feet   ({len(feet_instrumented)}): {', '.join(feet_instrumented)}")


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
