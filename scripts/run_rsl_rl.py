"""Run IsaacLab's official rsl_rl train/play scripts with the rubik_lab tasks registered.

    python scripts/run_rsl_rl.py train --task Rubik-Twist-Bimanual-Sharpa-v0 --headless --num_envs 4096
    python scripts/run_rsl_rl.py play  --task Rubik-Twist-Bimanual-Sharpa-v0 --num_envs 16 [--checkpoint path.pt]
Logs go to ./logs/rsl_rl/<experiment_name>.
"""

import os
import runpy
import sys

import rubik_lab  # noqa: F401  (registers gym tasks; imports nothing from Isaac Sim, so it is safe before AppLauncher)

ISAACLAB = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/workspace/third_party/IsaacLab"))

if len(sys.argv) < 2 or sys.argv[1] not in ("train", "play"):
    sys.exit(__doc__)
script = os.path.join(ISAACLAB, "scripts/reinforcement_learning/rsl_rl", f"{sys.argv[1]}.py")
sys.path.insert(0, os.path.dirname(script))
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
