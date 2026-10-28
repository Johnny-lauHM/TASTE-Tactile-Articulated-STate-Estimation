from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BimanualTwistPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 250
    experiment_name = "rubik_twist_bimanual_sharpa"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        noise_std_type="log",  # v2 crashed: the scalar std went negative ("normal expects all elements of std >= 0")
        init_noise_std=0.15,  # residual actions: std 0.3+ already pries the grasp open (diag: 2.8 drops/env in 3 s)
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


@configclass
class FlickPPORunnerCfg(BimanualTwistPPORunnerCfg):
    """single-finger flick tasks: a 4-dim action, short episodes"""

    experiment_name = "rubik_flick"
    max_iterations = 3000
    save_interval = 100
