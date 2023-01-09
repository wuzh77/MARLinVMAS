import numpy as np
import torch

from vmas import make_env, Wrapper


def make_Balance_env(scenario_name, number_agents, continuous_actions):
    scenario_name = scenario_name
    num_agents = number_agents
    continuous_actions = continuous_actions
    env = make_env(
        scenario_name=scenario_name,
        num_envs=1,
        device="cpu",
        continuous_actions=continuous_actions,
        wrapper=None,
        n_agents=num_agents,
        max_step=200
    )
    print('environment {} has been created.'.format(scenario_name))
    return env