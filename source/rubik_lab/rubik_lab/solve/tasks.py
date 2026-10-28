import gymnasium as gym

gym.register(
    id="Rubik-Solve-Skill-v0",
    entry_point="rubik_lab.solve.env:SolveEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "rubik_lab.solve.env_cfg:SolveEnvCfg"},
)
