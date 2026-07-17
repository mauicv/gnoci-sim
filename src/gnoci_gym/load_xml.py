import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def _expand_includes(root, desc_dir):
    """Replace <include> elements with the children of the referenced file."""
    for include in list(root.findall('include')):
        filepath = os.path.join(desc_dir, os.path.basename(include.get('file', '')))
        included_root = ET.parse(filepath).getroot()
        idx = list(root).index(include)
        root.remove(include)
        for i, child in enumerate(included_root):
            root.insert(idx + i, child)


def _load_and_perturb_basic_xml(
        filename,
        inertial_mass_range=(0.0, 0.0),
        inertial_mass_noise=0.0,
        floor_tilt_range=0.0,
        floor_friction_range=(1.0, 1.0),
        fix_root_body=False,
        strip_contact_sensors=False,
    ):
    package_dir = os.path.dirname(os.path.abspath(__file__))
    desc_dir    = os.path.join(package_dir, 'desc')
    assets_dir  = os.path.join(desc_dir, 'assets')
    xml_path    = os.path.join(desc_dir, f'{filename}.xml')

    xml_root = ET.parse(xml_path).getroot()
    _expand_includes(xml_root, desc_dir)

    if fix_root_body:
        for body in xml_root.iter("body"):
            for fj in list(body.findall("freejoint")):
                body.remove(fj)
        for parent in xml_root.iter():
            for child in list(parent):
                if child.tag == "geom" and child.get("name") == "floor":
                    parent.remove(child)

    for compiler in xml_root.iter('compiler'):
        compiler.set('meshdir', assets_dir)

    for child in xml_root.findall('.//geom'):
        if child.attrib.get('mass', '0') == '0':
            continue
        base = float(child.attrib['mass'])
        inertial_mass_value = np.random.uniform(*inertial_mass_range) * base
        new_val = max(base + inertial_mass_value + np.random.normal(0, inertial_mass_noise) * base, mujoco.mjMINVAL)
        child.attrib['mass'] = str(new_val)

    for geom in xml_root.iter("geom"):
        if geom.get("name") == "floor":
            parts = geom.get("friction", "1.0 0.005 0.0001").split()
            parts[0] = str(np.random.uniform(*floor_friction_range))
            geom.set("friction", " ".join(parts))
            break

    if floor_tilt_range > 0:
        roll  = np.random.uniform(-floor_tilt_range, floor_tilt_range)
        pitch = np.random.uniform(-floor_tilt_range, floor_tilt_range)
        qr = [np.cos(roll  / 2), np.sin(roll  / 2), 0.0, 0.0]
        qp = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        w = qr[0]*qp[0] - qr[1]*qp[1] - qr[2]*qp[2] - qr[3]*qp[3]
        x = qr[0]*qp[1] + qr[1]*qp[0] + qr[2]*qp[3] - qr[3]*qp[2]
        y = qr[0]*qp[2] - qr[1]*qp[3] + qr[2]*qp[0] + qr[3]*qp[1]
        z = qr[0]*qp[3] + qr[1]*qp[2] - qr[2]*qp[1] + qr[3]*qp[0]
        for geom in xml_root.iter("geom"):
            if geom.get("name") == "floor":
                geom.set("quat", f"{w} {x} {y} {z}")
                break

    if strip_contact_sensors:
        sensor_elem = xml_root.find("sensor")
        if sensor_elem is not None:
            for child in list(sensor_elem):  # list() to avoid mutating while iterating
                if child.tag in ("touch", "force"):
                    sensor_elem.remove(child)

    return ET.tostring(xml_root, encoding='utf8')
