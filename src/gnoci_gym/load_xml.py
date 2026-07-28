import os
import xml.etree.ElementTree as ET


def _expand_includes(root, desc_dir):
    """Replace <include> elements with the children of the referenced file."""
    for include in list(root.findall('include')):
        filepath = os.path.join(desc_dir, os.path.basename(include.get('file', '')))
        included_root = ET.parse(filepath).getroot()
        idx = list(root).index(include)
        root.remove(include)
        for i, child in enumerate(included_root):
            root.insert(idx + i, child)


def _load_xml(filename, fix_root_body=False):
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

    return ET.tostring(xml_root, encoding='utf8')
