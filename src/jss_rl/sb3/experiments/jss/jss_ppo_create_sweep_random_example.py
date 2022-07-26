import wandb as wb

from typing import Dict
from pathlib import Path

from jss_utils.jss_logger import log


def create_sweep(sweep_config: Dict, project: str) -> str:
    created_with_script = Path(__file__).stem
    sweep_config['parameters']['created_with_script'] = {
        "values": [created_with_script]
    }
    return wb.sweep(sweep_config, project=project)


if __name__ == '__main__':
    from jss_rl.sb3.experiments.jss.jss_ppo_sweep_config_random_example import random_ppo_sweep_config
    s_config = random_ppo_sweep_config # "cpcvcb17"
    # from jss_rl.sb3.experiments.jss.jss_ppo_sweep_config_grid_example import grid_ppo_sweep_config
    # s_config = grid_ppo_sweep_config # 3n6q34m3

    #from jss_rl.sb3.experiments.jss.jss_ppo_dynamic_instance_grid_sweep_example import \
    #   dynamic_instance_sweep_config_random
    # s_config = dynamic_instance_sweep_config_random # v5afiq7u

    # from jss_rl.sb3.experiments.jss.jss_ppo_icm_sweep_config_random_example import ppo_icm_sweep_config_random
    # s_config = ppo_icm_sweep_config_random

    sweep_id = create_sweep(sweep_config=s_config, project="test")
    log.info(f"sweep_id: {sweep_id}")
