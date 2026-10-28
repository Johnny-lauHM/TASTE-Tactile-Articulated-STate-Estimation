"""Dependency-free constants (importable without Isaac Sim)."""

import os

ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets"))
CUBE_SIZE = 0.070  # edge length [m]; must match scripts/gen_rubik_cube_usd.py --size (regenerate the USD and all grasp caches when changed)
FINGER_JOINTS = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
]  # fmt: skip
# virtual float-base joints, chained base_root -> x -> y -> z -> roll(x) -> pitch(y) -> yaw(z) -> <side>_hand_wrist
FLOAT_BASE_JOINTS = ["x_joint", "y_joint", "z_joint", "roll_joint", "pitch_joint", "yaw_joint"]
FINGERTIPS = ["thumb", "index", "middle", "ring", "pinky"]
