import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def _load_and_perturb_basic_xml(
        filename,
        motor_gear_range=(500000, 700000),
        motor_gear_noise=10000,
        inertial_mass_range=(0.04, 0.06),
        inertial_mass_noise=0.01,
    
    ):
    package_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(package_dir, 'desc', f'{filename}.xml')
    xml_tree = ET.parse(xml_path)
    xml_root = xml_tree.getroot()

    # motor_gear_value = np.random.uniform(*motor_gear_range)
    # for child in xml_root.findall('actuator/motor'):
    #     new_val = motor_gear_value + np.random.normal(0, motor_gear_noise)
    #     child.attrib['gear'] = str(new_val)

    # for child in xml_root.findall('.//inertial'):
    #     inertial_mass_value = np.random.uniform(*inertial_mass_range) * float(child.attrib['mass']) * 0.1
    #     new_val = max(float(child.attrib['mass']) + inertial_mass_value + np.random.normal(0, inertial_mass_noise), mujoco.mjMINVAL)
    #     child.attrib['mass'] = str(new_val)

    return ET.tostring(xml_root, encoding='utf8')
