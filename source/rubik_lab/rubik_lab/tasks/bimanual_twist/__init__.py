import gymnasium as gym

from . import agents

gym.register(
    id="Rubik-Twist-Bimanual-Sharpa-v0",
    entry_point=f"{__name__}.env:BimanualTwistEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BimanualTwistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BimanualTwistPPORunnerCfg",
    },
)

for _name, _cfg in (("U", "FlickUEnvCfg"), ("D", "FlickDEnvCfg"), ("FB", "FlickFBEnvCfg"),
                    ("Us", "FlickUsEnvCfg"), ("Ds", "FlickDsEnvCfg"), ("Fs", "FlickFsEnvCfg"), ("Bs", "FlickBsEnvCfg"),
                    ("UDs", "FlickUDsEnvCfg"), ("FBs", "FlickFBsEnvCfg"), ("UDRs", "FlickUDRsEnvCfg")):
    gym.register(
        id=f"Rubik-Flick-{_name}-Sharpa-v0",
        entry_point=f"{__name__}.env:BimanualTwistEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlickPPORunnerCfg",
        },
    )

# Complete-cycle tasks are versioned separately from historical one-push tasks.
for _name, _cfg in (("U", "CycleUEnvCfg"), ("Up", "CycleUpEnvCfg")):
    gym.register(
        id=f"Rubik-Cycle-{_name}-Sharpa-v0",
        entry_point=f"{__name__}.cycle_env:PrimitiveCycleEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cycle_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.cycle_cfg:CyclePPORunnerCfg",
        },
    )
