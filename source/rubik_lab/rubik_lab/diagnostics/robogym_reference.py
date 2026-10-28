"""Pinned robogym collision-model loader for isolated reference diagnostics."""
import xml.etree.ElementTree as ET

import mujoco


def load_model(assets, cubelet_joints="native-euler"):
    xml = ET.parse(assets / 'rubik_perpendicular.xml').getroot()
    for parent in xml.iter():
        for element in list(parent):
            if element.tag in ('texture', 'material'):
                parent.remove(element)
            elif element.tag == 'joint' and 'spring:' in element.get('name', ''):
                parent.remove(element)
            elif element.tag == 'mesh':
                element.set('file', str(assets / 'roundcube.msh'))
        parent.attrib.pop('material', None)
    if cubelet_joints == 'ball':
        for body in xml.iter('body'):
            rotations = [j for j in body.findall('joint') if j.get('name', '').startswith(('cubelet:rotx:', 'cubelet:roty:', 'cubelet:rotz:'))]
            if rotations:
                if len(rotations) != 3:
                    raise ValueError('Expected three rotation joints per free cubelet')
                for joint in rotations:
                    body.remove(joint)
                ET.SubElement(body, 'joint', name='cubelet:rotation:' + body.get('name'), type='ball', limited='false')
    # Keep the original dynamic root: deleting its joints changes parent collision filtering.
    ET.SubElement(ET.SubElement(xml, 'equality'), 'weld', body1='middle', solref='0.002 1')
    ET.SubElement(xml, 'option', timestep='0.001', gravity='0 0 0', integrator='implicitfast',
                  iterations='100', tolerance='1e-10')
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
