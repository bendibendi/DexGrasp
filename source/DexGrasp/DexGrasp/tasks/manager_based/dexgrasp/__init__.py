import gymnasium as gym

from . import agents

import os
DEX_DIR = os.path.dirname(os.path.abspath(__file__))
print(DEX_DIR)

gym.register(
    id="DexGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexgrasp_env_cfg:DexGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
    disable_env_checker=True,
)