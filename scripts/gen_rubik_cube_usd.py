"""Generate a two-body articulated Rubik's cube USD.

Bodies (both frames at the cube center):
  core : bottom two layers (z in [-s/2, s/6])
  top  : top layer         (z in [ s/6, s/2]), attached to `core` by revolute `twist_joint` about +z

Collision: one box per body. Visual: 27 black cubies with colored stickers (no collision).
Rotate the cube at spawn time to choose which physical face is the twisting layer.

Usage: python scripts/gen_rubik_cube_usd.py [--size 0.057] [--mass 0.08] [--out assets/rubik_cube/rubik_cube_2body.usda]
"""

import argparse
import os

# U, D, F, B, L, R  (standard color scheme: white top, yellow bottom, green front, blue back, orange left, red right)
FACE_COLORS = {
    "+z": (0.95, 0.95, 0.95),
    "-z": (1.0, 0.85, 0.0),
    "+x": (0.0, 0.6, 0.2),
    "-x": (0.0, 0.25, 0.8),
    "+y": (1.0, 0.45, 0.0),
    "-y": (0.8, 0.05, 0.05),
}


def _xform_ops(translate, scale):
    return (
        f"            double3 xformOp:translate = ({translate[0]:.6f}, {translate[1]:.6f}, {translate[2]:.6f})\n"
        f"            float3 xformOp:scale = ({scale[0]:.6f}, {scale[1]:.6f}, {scale[2]:.6f})\n"
        '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
    )


def _visual_cube(name, translate, scale, color):
    return (
        f'        def Cube "{name}"\n'
        "        {\n"
        "            double size = 1\n"
        f"            color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]\n"
        + _xform_ops(translate, scale)
        + "        }\n"
    )


def _cubie_visuals(s, layers):
    c = s / 3.0
    gap = 0.0006
    sticker_t = 0.0008
    sticker_w = c * 0.82
    out = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in layers:
                if i == 0 and j == 0 and k == 0:
                    continue
                center = (i * c, j * c, k * c)
                name = f"cubie_{i + 1}{j + 1}{k + 1}"
                out.append(_visual_cube(name, center, (c - gap,) * 3, (0.05, 0.05, 0.05)))
                for axis, idx in (("x", i), ("y", j), ("z", k)):
                    if idx == 0:
                        continue
                    sign = "+" if idx > 0 else "-"
                    a = "xyz".index(axis)
                    pos = list(center)
                    pos[a] += idx * (c / 2.0)
                    scl = [sticker_w] * 3
                    scl[a] = sticker_t
                    out.append(_visual_cube(f"{name}_st_{axis}{'p' if idx > 0 else 'n'}", pos, scl, FACE_COLORS[sign + axis]))
    return "".join(out)


def _body(name, s, z_lo, z_hi, mass, layers, extra=""):
    h = z_hi - z_lo
    zc = 0.5 * (z_lo + z_hi)
    return f'''    def Xform "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = {mass:.6f}
        point3f physics:centerOfMass = (0, 0, {zc:.6f})
{extra}
        def Cube "collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            double size = 1
            uniform token purpose = "guide"
            rel material:binding:physics = </rubik_cube/physics_material> (
                bindMaterialAs = "weakerThanDescendants"
            )
            double3 xformOp:translate = (0, 0, {zc:.6f})
            float3 xformOp:scale = ({s:.6f}, {s:.6f}, {h:.6f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}

        def Xform "visuals"
        {{
{_cubie_visuals(s, layers)}        }}
    }}

'''


def _joint(name, body0, body1, twist_damping, twist_friction):
    return f'''    def PhysicsRevoluteJoint "{name}" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxJointAPI"]
    )
    {{
        rel physics:body0 = </rubik_cube/{body0}>
        rel physics:body1 = </rubik_cube/{body1}>
        uniform token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        float drive:angular:physics:stiffness = 0
        float drive:angular:physics:damping = {twist_damping}
        float drive:angular:physics:maxForce = 1000
        uniform token drive:angular:physics:type = "force"
        float physxJoint:jointFriction = {twist_friction}
        float physxJoint:armature = 0.0001
    }}
'''


def build(size: float, mass: float, friction: float, twist_damping: float, twist_friction: float, three_body: bool = False) -> str:
    s = size
    if three_body:  # core = bottom layer, `mid` on mid_joint (core -> mid), `top` on twist_joint (mid -> top): the middle
        # layer can be dragged along by the top one (the joint frictions couple neighbouring layers)
        core = _body("core", s, -s / 2, -s / 6, mass / 3.0, (-1,))
        mid = _body("mid", s, -s / 6, s / 6, mass / 3.0, (0,))
        top = _body("top", s, s / 6, s / 2, mass / 3.0, (1,))
        bodies = core + mid + top
        joints = _joint("mid_joint", "core", "mid", twist_damping, twist_friction) + _joint("twist_joint", "mid", "top", twist_damping, twist_friction)
    else:
        core = _body("core", s, -s / 2, s / 6, mass * 2.0 / 3.0, (-1, 0))
        top = _body("top", s, s / 6, s / 2, mass / 3.0, (1,))
        bodies = core + top
        joints = _joint("twist_joint", "core", "top", twist_damping, twist_friction)
    return f'''#usda 1.0
(
    defaultPrim = "rubik_cube"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "rubik_cube" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
)
{{
    bool physxArticulation:enabledSelfCollisions = 0

    def Material "physics_material" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {friction}
        float physics:dynamicFriction = {friction}
        float physics:restitution = 0
    }}

{bodies}{joints}}}
'''


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=float, default=0.070, help="edge length [m] (standard 3x3 = 57 mm; this project: 70 mm)")
    p.add_argument("--mass", type=float, default=0.148, help="total mass [kg] (80 g at 57 mm, scaled by volume)")
    p.add_argument("--friction", type=float, default=0.8)
    # USD angular drive damping is per degree: 1e-5 Nm*s/deg ~= 5.7e-4 Nm*s/rad
    p.add_argument("--twist_damping", type=float, default=1e-5)
    p.add_argument("--twist_friction", type=float, default=0.0, help="PhysX joint friction coefficient (0: the Coulomb turning resistance is modelled in env.py; a PhysX coefficient scales with the joint load, so a finger push would fight its own force)")
    p.add_argument("--three_body", action="store_true", help="separate middle layer on its own joint (mid_joint)")
    p.add_argument("--out", default=os.path.join(root, "assets/rubik_cube/rubik_cube_2body.usda"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(build(a.size, a.mass, a.friction, a.twist_damping, a.twist_friction, a.three_body))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
