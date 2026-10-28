"""Asset configs: Sharpa Wave float-base hands (official sharpa-urdf-usd-xml) and the 2-body Rubik's cube."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .constants import ASSET_DIR, FINGER_JOINTS, FINGERTIPS, FLOAT_BASE_JOINTS  # noqa: F401





from collections.abc import Callable  # noqa: E402

from isaaclab.sim.spawners.from_files import from_files as _ff  # noqa: E402
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg  # noqa: E402
from isaaclab.sim.utils import bind_physics_material, clone  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


@clone
def _spawn_usd_with_pad_material(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """as isaaclab's spawn_from_usd_with_compliant_contact_material, but the pad material also gets a friction
    coefficient (the upstream spawner leaves it at the 0.5 default)."""
    prim = _ff._spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    if not cfg.physics_material_prim_path:
        return prim
    mat = RigidBodyMaterialCfg(
        static_friction=cfg.pad_friction, dynamic_friction=cfg.pad_friction, restitution=0.0,
        compliant_contact_stiffness=cfg.compliant_contact_stiffness or 0.0,
        compliant_contact_damping=cfg.compliant_contact_damping or 0.0,
    )  # fmt: skip
    for path in cfg.physics_material_prim_path:
        body = path if path.startswith("/") else f"{prim_path}/{path}"
        mat.func(f"{body}/pad_material", mat)
        bind_physics_material(body, f"{body}/pad_material")
    return prim


@configclass
class UsdFileWithPadMaterialCfg(sim_utils.UsdFileWithCompliantContactCfg):
    func: Callable = _spawn_usd_with_pad_material
    pad_friction: float = 1.0  # silicone elastomer pads; sharpa-rl-lab uses 0.8 (randomized x0.5-2)


# contact models
#   "tacmap": official sharpa-tacmap env_cfg: compliant elastomer pads (stiffness 280, damping 20), contact offset 0.2 mm,
#             rest offset -0.1 mm on the hand and -0.5 mm on the object, so the pads deform and TacMap sees depth
#   "rigid" : official sharpa-rl-lab env_cfg: rigid contacts, contact offset 2 mm, rest offset 0
CONTACT_MODELS = {
    "tacmap": dict(hand_contact_offset=0.0002, hand_rest_offset=-0.0001, pad_stiffness=280.0, pad_damping=20.0, pad_friction=1.0,
                   torsional_patch_radius=0.0,
                   obj_contact_offset=0.0001, obj_rest_offset=-0.0005),
    "rigid": dict(hand_contact_offset=0.002, hand_rest_offset=0.0, pad_stiffness=None, pad_damping=None, pad_friction=1.0,
                  torsional_patch_radius=0.0,
                  obj_contact_offset=0.002, obj_rest_offset=0.0),
}  # fmt: skip


def sharpa_hand_cfg(side: str, mit_mode: bool = False, contact_model: str = "tacmap", wrist_stiffness: float | None = None) -> ArticulationCfg:
    """Float-base Sharpa Wave (with wrist). Wrist frame: +z along fingers, +x palm normal, thumb at +y (right) / -y (left).

    Dynamics follow the official recommendation: finger gains come from the USD (IdealPD, stiffness/damping=None).
    The 6 virtual base joints use the USD drive gains with an implicit (stable) PD.
    """
    cm = CONTACT_MODELS[contact_model]
    assert side in ("left", "right")
    suffix = "_MITmode" if mit_mode else ""
    usd = os.path.join(ASSET_DIR, f"sharpa_wave_float_base/{side}/{side}_sharpa_wave_with_float_base_with_wrist{suffix}.usda")
    return ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{side}_hand",
        spawn=UsdFileWithPadMaterialCfg(
            usd_path=usd,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cm["hand_contact_offset"], rest_offset=cm["hand_rest_offset"],
                # soft finger pads have a contact patch -> torsional friction (PhysX point contacts have none by default)
                torsional_patch_radius=cm["torsional_patch_radius"], min_torsional_patch_radius=cm["torsional_patch_radius"],
            ),
            compliant_contact_stiffness=cm["pad_stiffness"],
            compliant_contact_damping=cm["pad_damping"],
            physics_material_prim_path=[f"{side}_{f}_elastomer" for f in FINGERTIPS],
            pad_friction=cm["pad_friction"],
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
        actuators={
            # the USD float base is soft in translation (3000 N/m: a 25 N finger push moves the wrist 8 mm and the held
            # cube with it); wrist_stiffness overrides the x/y/z drives (the real wrist sits on a stiff Franka arm)
            "float_base": ImplicitActuatorCfg(
                joint_names_expr=[f"{side}_{j}" for j in FLOAT_BASE_JOINTS],
                # (all six joints listed: a partial dict would leave the angular drives without gains)
                stiffness=None if wrist_stiffness is None else {f"{side}_[xyz]_joint": wrist_stiffness, f"{side}_(roll|pitch|yaw)_joint": math.degrees(5.0)},
                damping=None if wrist_stiffness is None else {f"{side}_[xyz]_joint": 2.0 * math.sqrt(wrist_stiffness * 1.2), f"{side}_(roll|pitch|yaw)_joint": math.degrees(0.1)},
            ),
            "fingers": IdealPDActuatorCfg(
                joint_names_expr=[f"{side}_{j}" for j in FINGER_JOINTS],
                stiffness=None,
                damping=None,
                # the USDs carry no effort limit (-> 1e9); use the URDF motor limits
                effort_limit={".*_CMC_(FE|AA)": 3.3, ".*_MCP_(FE|AA)": 1.864, ".*_(PIP|IP)": 0.638, ".*_DIP": 0.189369, ".*_pinky_CMC": 0.5285},
                velocity_limit={".*_CMC_(FE|AA)": 11.84, ".*_MCP_(FE|AA)": 16.08, ".*_(PIP|IP)": 11.62, ".*_DIP": 14.67, ".*_pinky_CMC": 35.07},
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def rubik_cube_cfg(contact_model: str = "tacmap") -> ArticulationCfg:
    cm = CONTACT_MODELS[contact_model]
    cfg = RUBIK_CUBE_CFG.copy()
    cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=cm["obj_contact_offset"], rest_offset=cm["obj_rest_offset"]
    )
    return cfg


RUBIK_CUBE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/cube",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSET_DIR, "rubik_cube/rubik_cube_2body.usda"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=1.0,  # restoring a cached tight grasp must not launch the cube
            # a hand-held cube never moves this fast; caps impulses from fast fingers (tiny tacmap contact offsets)
            max_linear_velocity=3.0,
            max_angular_velocity=2000.0,  # deg/s
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=CONTACT_MODELS["tacmap"]["obj_contact_offset"], rest_offset=CONTACT_MODELS["tacmap"]["obj_rest_offset"]
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"twist_joint": 0.0}),
    # passive layer: resistance comes from the USD joint friction / damping
    actuators={"twist": ImplicitActuatorCfg(joint_names_expr=["twist_joint"], stiffness=0.0, damping=None)},
)


# FR3 v2 joints (franka_description fr3v2_1): torque / velocity limits; reflected rotor inertia (motor_inertia * gear^2)
FR3_EFFORT = {"fr3_joint[1-4]": 87.0, "fr3_joint[5-7]": 12.0}
FR3_VELOCITY = {"fr3_joint[1-4]": 2.62, "fr3_joint5": 5.26, "fr3_joint6": 4.18, "fr3_joint7": 5.26}
FR3_ARMATURE = {"fr3_joint[1-2]": 0.606, "fr3_joint[3-4]": 0.462, "fr3_joint[5-7]": 0.206}


def sharpa_fr3_cfg(side: str, base_pos, base_quat, arm_q, contact_model: str = "tacmap",
                   arm_stiffness: tuple[float, float] = (4000.0, 1000.0), arm_collision: bool = True,
                   usd_dir: str = "assets_franka") -> ArticulationCfg:
    """FR3 + printed adapter + Sharpa Wave (~/workspace/fr3-sharpa-rubik), hand joint dynamics as the official
    float-base hand (assets_franka/<side>/fr3_sharpa_<side>.usda, scripts/franka/make_fr3_hand_usd.py). The arm is a
    stiff joint-space PD (gravity compensated, as the real arm's controller) with the FR3 torque / speed limits."""
    cfg = sharpa_hand_cfg(side, contact_model=contact_model)
    cfg.spawn.usd_path = os.path.join(ASSET_DIR, "..", usd_dir, side, f"fr3_sharpa_{side}{'' if arm_collision else '_nocol'}.usda")
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=tuple(base_pos), rot=tuple(base_quat), joint_pos={**{f"fr3_joint{i + 1}": float(v) for i, v in enumerate(arm_q)}, f"{side}_.*": 0.0})
    k1, k2 = arm_stiffness
    dmp = float(os.environ.get("FR3_DAMP_SCALE", "2.5"))  # 2.5: less arm jitter, fewer drops (2026-10-24); the links' own inertia comes on top of the armature: >1 is closer to critical
    del cfg.actuators["float_base"]
    cfg.actuators["arm"] = ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[1-7]"],
        stiffness={"fr3_joint[1-4]": k1, "fr3_joint[5-7]": k2},
        damping={"fr3_joint[1-4]": dmp * 2.0 * math.sqrt(k1 * 0.6), "fr3_joint[5-7]": dmp * 2.0 * math.sqrt(k2 * 0.2)},
        armature=FR3_ARMATURE, effort_limit_sim=FR3_EFFORT, velocity_limit_sim=FR3_VELOCITY)
    return cfg
