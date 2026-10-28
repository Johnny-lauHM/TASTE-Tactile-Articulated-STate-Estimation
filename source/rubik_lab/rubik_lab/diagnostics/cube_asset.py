"""Convert the pinned rounded-cube geometry and inertias to a PhysX prototype.

This asset preserves source dimensions/mass. It is not the production70mm cube.
MuJoCo frictionloss and6Dcontact friction are deliberately not mislabelled as
PhysX coefficients; the receiving diagnostic must declare their treatment.
"""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from rubik_lab.diagnostics.robogym_reference import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
assets = Path(__file__).resolve().parents[4] / 'assets/reference/robogym'
model = load_model(assets, 'ball')


def vector(values):
    return '(' + ', '.join(f'{float(x):.10g}' for x in values) + ')'


def matrix(quaternion):
    result = np.zeros(9)
    mujoco.mju_quat2Mat(result, quaternion)
    return result.reshape(3, 3)


def body_header(name, mass, center, inertia, axes):
    return [f'    def Xform "{name}" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])',
            '    {', f'        float physics:mass = {mass:.10g}',
            f'        point3f physics:centerOfMass = {vector(center)}',
            f'        float3 physics:diagonalInertia = {vector(inertia)}',
            f'        quatf physics:principalAxes = {vector(axes)}',
            '        float physxRigidBody:linearDamping = 0',
            '        float physxRigidBody:angularDamping = 0',
            '        float physxRigidBody:maxAngularVelocity = 1000']


lines = ['#usda 1.0', '# Derived from openai/robogym9ff1111fd557b1a01911d7d8c9d59a453071c96e; see assets/reference/robogym/LICENSE.',
         '# Prototype: native57.1mm geometry; no production task replacement.',
         '(defaultPrim = "cube"', ' metersPerUnit = 1', ' upAxis = "Z")',
         'def Xform "cube" (prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"])', '{',
         '    bool physxArticulation:enabledSelfCollisions = true',
         '    int physxArticulation:solverPositionIterationCount = 8',
         '    int physxArticulation:solverVelocityIterationCount = 1',
         '    def Material "material" (prepend apiSchemas = ["PhysicsMaterialAPI"])', '    {',
         '        float physics:staticFriction = 0.2', '        float physics:dynamicFriction = 0.2',
         '        float physics:restitution = 0', '    }']
core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'middle')
lines += body_header('core', model.body_mass[core], model.body_ipos[core], model.body_inertia[core], model.body_iquat[core])
lines += ['        def Sphere "collision" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"])',
          '        {', '            double radius = 0.01',
          '            rel material:binding:physics = </cube/material>', '        }', '    }']
count = 0
for body in range(model.nbody):
    source_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
    if not source_name.startswith('cubelet:'):
        continue
    name = 'piece_' + source_name.split(':', 1)[1]
    geometry = int(model.body_geomadr[body])
    mesh = int(model.geom_dataid[geometry])
    vertex_start, vertex_count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    vertices = model.mesh_vert[vertex_start:vertex_start + vertex_count].astype(np.float64)
    vertices = vertices @ matrix(model.geom_quat[geometry]).T + model.geom_pos[geometry]
    vertices = vertices @ matrix(model.body_quat[body]).T + model.body_pos[body]
    triangles = model.mesh_face[face_start:face_start + face_count]
    center = model.body_pos[body] + matrix(model.body_quat[body]) @ model.body_ipos[body]
    lines += body_header(name, model.body_mass[body], center, model.body_inertia[body], model.body_iquat[body])
    lines += ['        def Mesh "collision" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"])',
              '        {', '            uniform token subdivisionScheme = "none"',
              '            token physics:approximation = "convexHull"',
              '            float physxCollision:contactOffset = 0.0001',
              '            float physxCollision:restOffset = 0',
              '            rel material:binding:physics = </cube/material>',
              '            point3f[] points = [' + ', '.join(vector(v) for v in vertices) + ']',
              '            int[] faceVertexCounts = [' + ', '.join(['3'] * len(triangles)) + ']',
              '            int[] faceVertexIndices = [' + ', '.join(str(int(v)) for v in triangles.ravel()) + ']',
              '        }', '    }']
    parent = int(model.body_parentid[body])
    joint = int(model.body_jntadr[parent])
    hinge = model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
    joint_name = ('driver_' if hinge else 'ball_') + source_name.split(':', 1)[1]
    joint_type = 'PhysicsRevoluteJoint' if hinge else 'PhysicsSphericalJoint'
    lines += [f'    def {joint_type} "{joint_name}" (prepend apiSchemas = ["PhysxJointAPI"])', '    {',
              '        rel physics:body0 = </cube/core>', f'        rel physics:body1 = </cube/{name}>',
              '        bool physics:collisionEnabled = false',
              '        float physxJoint:armature = 0.0001', '        float physxJoint:jointFriction = 0']
    if hinge:
        axis = 'XYZ'[int(np.argmax(np.abs(model.jnt_axis[joint])))]
        lines += [f'        uniform token physics:axis = "{axis}"']
    else:
        lines += ['        float physics:coneAngle0Limit = -1', '        float physics:coneAngle1Limit = -1']
    lines += ['    }']
    count += 1
if count != 26:
    raise ValueError('Expected26cubelets')
lines += ['}']
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text('\n'.join(lines) + '\n')
print({'asset': str(args.out), 'cubelets': count, 'source_mass_kg': float(model.body_mass.sum()),
       'differences': ['Spherical ratherthanEuler cubelet joints', 'Nativejointfrictionloss not mapped to PhysX coefficient',
                       'Only slidingcontactfriction represented; rolling/torsionalfriction unresolved', 'PhysXconvex collision solver differs']})
