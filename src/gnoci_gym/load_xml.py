import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def _load_and_perturb_basic_xml(
        filename,
        inertial_mass_range=(0.04, 0.06),
        inertial_mass_noise=0.01,
        floor_tilt_range=0.0,
    ):
    package_dir = os.path.dirname(os.path.abspath(__file__))
    desc_dir   = os.path.join(package_dir, 'desc')
    xml_path   = os.path.join(desc_dir, f'{filename}.xml')
    xml_tree = ET.parse(xml_path)
    xml_root = xml_tree.getroot()

    for include in xml_root.iter('include'):
        rel = include.get('file', '')
        include.set('file', os.path.join(desc_dir, os.path.basename(rel)))

    for child in xml_root.findall('.//geom'):
        if child.attrib.get('mass', '0') == '0':
            continue
        inertial_mass_value = np.random.uniform(*inertial_mass_range) * float(child.attrib['mass']) * 0.1
        new_val = max(float(child.attrib['mass']) + inertial_mass_value + np.random.normal(0, inertial_mass_noise), mujoco.mjMINVAL)
        child.attrib['mass'] = str(new_val)

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

    return ET.tostring(xml_root, encoding='utf8')
