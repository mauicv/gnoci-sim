import os
import xml.etree.ElementTree as ET
import numpy as np


basic_attributes = {
    'actuator/motor': {
        'gear': {
            'value': 600000,
            'perturbation': 10000,
        }
    }
}


complex_attributes = {
    'actuator/position': {
        'kp': {
            'value': 250000,
            'perturbation': 0.1,
        }
    },
}


def _load_and_perturb_xml(env_name, attributes):
    package_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(package_dir, 'desc', f'{env_name}.xml')
    xml_tree = ET.parse(xml_path)
    xml_root = xml_tree.getroot()

    for child in xml_root.findall('compiler'):
        child.attrib['meshdir'] = str(os.path.join(package_dir, 'desc', 'meshes'))

    for key, value in attributes.items():
        for child in xml_root.findall(key):
            for attribute, attribute_value in value.items():
                child.attrib[attribute] = str(attribute_value['value'])
                if attribute_value['perturbation'] > 0:
                    new_val = np.random.normal(attribute_value['value'], attribute_value['perturbation'])
                    child.attrib[attribute] = str(new_val)
    return ET.tostring(xml_root, encoding='utf8')
