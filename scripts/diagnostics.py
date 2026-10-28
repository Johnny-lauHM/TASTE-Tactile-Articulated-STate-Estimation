"""Single entry point for focused mechanical and sensory experiments."""
import runpy
import sys
from pathlib import Path
COMMANDS = {
    "press-reference": "official_press.py",
    "mechanical-cycle": "mechanical_cycle.py",
    "mechanical-readout": "mechanical_readout.py",
    "mechanical-geometry": "mechanical_geometry.py",
    "sensory-probe": "sensory_probe.py",
    "continuation": "continuation.py",
    "ready-return-geometry": "ready_return_geometry.py",
    "probe-readout": "probe_readout.py",
    "release-replay": "release_replay.py",
    "rotation-reference": "official_rotation.py",
    "readout": "readout.py",
    "passivity": "passivity.py",
    "numerics-readout": "numerics_readout.py",
    "cube-reference": "cube_reference.py",
    "tracking-readout": "tracking_readout.py",
    "geometry-readout": "geometry_readout.py",
    "joint-reference": "joint_reference.py",
    "contact-reference": "contact_reference.py",
    "actuator-reference": "actuator_reference.py",
    "cube-asset": "cube_asset.py",
    "cube-physx": "cube_physx.py",
    "reach-reference": "reach_reference.py",
    "contact-path": "contact_path.py",
    "constrained-ik": "constrained_ik.py",
    "hand-collision": "hand_collision.py",
    "hand-collision-assets": "hand_collision_assets.py"
}
if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    raise SystemExit("Usage: python scripts/diagnostics.py {" + ",".join(COMMANDS) + "} [options]")
command = sys.argv.pop(1)
runpy.run_path(str(Path(__file__).resolve().parents[1] / "source/rubik_lab/rubik_lab/diagnostics" / COMMANDS[command]), run_name="__main__")
