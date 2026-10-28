"""FR3 + Sharpa Wave with the hand's own flange bolted straight onto the FR3 flange (no printed adapter): the
fr3-sharpa-rubik robot URDF with the adapter link removed and fr3_link8 -> <side>_hand_flange fixed, rotated
`yaw` about the flange axis (the bolt clocking; fr3_layout.py --direct searches it through Chain.set_ee_yaw).

    python scripts/franka/build_direct.py [--from_json]  ->  assets_franka/direct0 (or direct_json)/<side>/robot.urdf
"""
import json
import math
import os
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3 = os.path.expanduser("~/workspace/fr3-sharpa-rubik")
import sys

# flange on flange with no rotation (user, 2026-10-23); --from_json: the clocking of the direct_mount.json assembly
FROM_JSON = "--from_json" in sys.argv
OUT_DIR = "assets_franka/direct_json" if FROM_JSON else "assets_franka/direct0"
POSE = json.load(open(os.path.join(FR3, "assets/poses/direct_mount.json")))["robots"] if FROM_JSON else None

for side in ("left", "right"):
    r = ET.parse(os.path.join(FR3, f"assets/{side}/robot.urdf")).getroot()
    adapter = f"{side}_adapter"
    for e in list(r):
        if e.tag == "link" and e.get("name") == adapter:
            r.remove(e)
        elif e.tag == "joint" and adapter in (e.find("parent").get("link"), e.find("child").get("link")):
            r.remove(e)
    j = ET.SubElement(r, "joint", name=f"{side}_direct_flange_joint", type="fixed")
    m = POSE[side]["arm_to_hand"] if FROM_JSON else {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]}
    ET.SubElement(j, "origin", xyz=" ".join(map(str, m["xyz_m"])), rpy=" ".join(map(str, m["rpy_rad"])))
    ET.SubElement(j, "parent", link="fr3_link8")
    ET.SubElement(j, "child", link=f"{side}_hand_flange")
    out = os.path.join(ROOT, OUT_DIR, side, "robot.urdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ET.ElementTree(r).write(out, xml_declaration=True, encoding="utf-8")
    links = {l.get("name") for l in r.findall("link")}
    child = [jj.find("child").get("link") for jj in r.findall("joint")]
    roots = [l for l in links if l not in child]
    print(f"[direct] {out}: {len(links)} links, roots {roots}, flange rpy {m['rpy_rad']}")
