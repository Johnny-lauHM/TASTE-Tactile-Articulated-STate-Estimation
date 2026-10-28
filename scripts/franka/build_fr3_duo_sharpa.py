"""Build one URDF: Franka FR3 duo mount + two FR3 v2.1 arms + Sharpa Wave hands on their flanges.

Arm: superdex_assets/test/urdf/fr3v2_1_urdf (franka_description 2.8.1, expanded), Franka hand removed.
Mount: franka_description accessories/fr3_duo_mount_v0_3 (the official duo layout: arms tilted out 51 deg).
Hand: sharpa_wave/<side>_sharpa_wave/<side>_sharpa_wave_with_flange.urdf, its <side>_hand_flange fixed to the arm's
link8 (the Franka flange frame) rotated --ee_yaw about the flange axis (Franka's own hand mounts at -45 deg).

    python scripts/franka/build_fr3_duo_sharpa.py [--ee_yaw -0.785398] [--out assets_franka/fr3_duo_sharpa.urdf]
"""
import argparse
import copy
import os
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3_URDF = os.path.join(ROOT, "superdex_assets/test/urdf/fr3v2_1_urdf/robots/fr3v2_1_franka_hand.urdf")
FR3_PKG = os.path.join(ROOT, "superdex_assets/test/urdf/fr3v2_1_urdf")
FD = os.path.expanduser("~/workspace/third_party/franka_description")
MOUNT_MESH = os.path.join(FD, "meshes/accessories/fr3_duo_mount_v0_3")
# accessories/fr3_duo_mount_v0_3/kinematics.yaml
MOUNT_ORIGIN = ("0.037 0 0.05", "0 0 0")  # minus the mounting point offset
MOUNT_ARM = {"left": ("0 0.05018 0", "-0.89334809 -0.17456074 -0.46334506"),
             "right": ("0 -0.05018 0", "0.89334809 -0.17456074 0.46334506")}
ARM_LINKS = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"]


def sub(parent, tag, **attrib):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrib.items()})


def fixed(robot, name, parent, child, xyz="0 0 0", rpy="0 0 0"):
    j = sub(robot, "joint", name=name, type="fixed")
    sub(j, "origin", xyz=xyz, rpy=rpy)
    sub(j, "parent", link=parent)
    sub(j, "child", link=child)


def mesh_link(robot, name, visuals, collisions, mass=1.0):
    link = sub(robot, "link", name=name)
    for fn, xyz in visuals:
        v = sub(link, "visual")
        sub(v, "origin", xyz=xyz, rpy="0 0 0")
        sub(sub(v, "geometry"), "mesh", filename=fn)
    for fn, xyz in collisions:
        c = sub(link, "collision")
        sub(c, "origin", xyz=xyz, rpy="0 0 0")
        sub(sub(c, "geometry"), "mesh", filename=fn)
    i = sub(link, "inertial")
    sub(i, "origin", xyz="0 0 0", rpy="0 0 0")
    sub(i, "mass", value=mass)
    sub(i, "inertia", ixx=0.01, ixy=0, ixz=0, iyy=0.01, iyz=0, izz=0.01)


def fix_meshes(elem, pkg_from, pkg_to):
    for m in elem.iter("mesh"):
        fn = m.get("filename")
        if fn.startswith(pkg_from):
            m.set("filename", os.path.join(pkg_to, fn[len(pkg_from):]))


def arm(robot, side):
    src = ET.parse(FR3_URDF).getroot()
    pre_old, pre_new = "fr3v2_1_", f"{side}_fr3_"
    keep = {pre_old + n for n in ARM_LINKS}
    for e in src:
        if e.tag == "link" and (e.get("name") in keep or any(e.get("name").startswith(pre_old + n + "_") for n in ARM_LINKS[:-1])):
            pass
        elif e.tag == "joint" and e.find("child").get("link") in keep | {l.get("name") for l in src.findall("link")
                                                                         if "_accelerometer_" in l.get("name")}:
            if e.find("parent").get("link") == "base":
                continue
        else:
            continue
        e = copy.deepcopy(e)
        e.set("name", e.get("name").replace(pre_old, pre_new))
        for t in ("parent", "child"):
            if e.find(t) is not None:
                e.find(t).set("link", e.find(t).get("link").replace(pre_old, pre_new))
        for tag in ("safety_controller", "position_based_velocity_limits"):  # non-standard for the importer
            for x in e.findall(tag):
                e.remove(x)
        fix_meshes(e, "package://franka_description/", FR3_PKG + "/")
        robot.append(e)


def hand(robot, side):
    src = ET.parse(os.path.join(ROOT, f"sharpa_wave/{side}_sharpa_wave/{side}_sharpa_wave_with_flange.urdf")).getroot()
    for e in src:
        if e.tag in ("link", "joint"):
            e = copy.deepcopy(e)
            fix_meshes(e, f"package://{side}_sharpa_wave/", os.path.join(ROOT, f"sharpa_wave/{side}_sharpa_wave") + "/")
            robot.append(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ee_yaw", type=float, default=-0.7853981633974483, help="rad: hand flange about the Franka flange axis")
    p.add_argument("--out", default=os.path.join(ROOT, "assets_franka/fr3_duo_sharpa.urdf"))
    a = p.parse_args()
    robot = ET.Element("robot", name="fr3_duo_sharpa")
    mesh_link(robot, "fr3_duo_mount_mounting_point",
              [(f"{MOUNT_MESH}/visual/fr3_duo_mount.dae", "0 0 0"), (f"{MOUNT_MESH}/visual/fr3_duo_cover.dae", "0 0 0.068")],
              [(f"{MOUNT_MESH}/collision/fr3_duo_mount.stl", "0 0 0"), (f"{MOUNT_MESH}/collision/fr3_duo_cover.stl", "0 0 0.068")], mass=5.0)
    sub(robot, "link", name="fr3_duo_mount_origin")
    fixed(robot, "fr3_duo_mount_origin_joint", "fr3_duo_mount_mounting_point", "fr3_duo_mount_origin", *MOUNT_ORIGIN)
    for side in ("left", "right"):
        arm(robot, side)
        fixed(robot, f"{side}_fr3_base_joint", "fr3_duo_mount_origin", f"{side}_fr3_link0", *MOUNT_ARM[side])
        hand(robot, side)
        fixed(robot, f"{side}_flange_joint", f"{side}_fr3_link8", f"{side}_hand_flange", rpy=f"0 0 {a.ee_yaw}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(a.out, xml_declaration=True, encoding="utf-8")
    links = [l.get("name") for l in robot.findall("link")]
    joints = [j for j in robot.findall("joint") if j.get("type") != "fixed"]
    print(f"[build] {a.out}: {len(links)} links, {len(joints)} movable joints")


if __name__ == "__main__":
    main()
