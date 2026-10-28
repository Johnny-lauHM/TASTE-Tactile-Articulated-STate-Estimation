"""FR3 + printed adapter + Sharpa with the adapter transforms of ~/workspace/fr3-sharpa-rubik/assets/adapters/assembly.json
(the user's re-clocked assembly; that project's robot.urdf predates it): copies its robot.urdf with the two adapter
joints set from the JSON, without touching the other project.

    python scripts/franka/build_adapter_v2.py   ->  assets_franka/adapter_v2/<side>/robot.urdf
"""
import json
import os
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3 = os.path.expanduser("~/workspace/fr3-sharpa-rubik")
A = json.load(open(os.path.join(FR3, "assets/adapters/assembly.json")))["adapters"]

for side in ("left", "right"):
    r = ET.parse(os.path.join(FR3, f"assets/{side}/robot.urdf")).getroot()
    cfg = A[side]
    for j in r.findall("joint"):
        par, ch = j.find("parent").get("link"), j.find("child").get("link")
        if (par, ch) == ("fr3_link8", f"{side}_adapter"):
            xyz, rpy = cfg["arm_to_adapter_xyz_m"], cfg["arm_to_adapter_rpy_rad"]
        elif (par, ch) == (f"{side}_adapter", f"{side}_hand_flange"):
            xyz, rpy = cfg["adapter_to_hand_xyz_m"], cfg["adapter_to_hand_rpy_rad"]
        else:
            continue
        o = j.find("origin")
        old = (o.get("xyz"), o.get("rpy"))
        o.set("xyz", " ".join(repr(float(v)) for v in xyz))
        o.set("rpy", " ".join(repr(float(v)) for v in rpy))
        print(f"[adapter_v2] {side} {par} -> {ch}: {old} -> ({o.get('xyz')}, {o.get('rpy')})")
    out = os.path.join(ROOT, f"assets_franka/adapter_v2/{side}/robot.urdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ET.ElementTree(r).write(out, xml_declaration=True, encoding="utf-8")
