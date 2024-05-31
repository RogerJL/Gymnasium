"""Test suite of HumanRendering wrapper."""
import re

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import HumanRendering


def test_human_rendering():
    for mode in ["rgb_array", "rgb_array_list"]:
        env = HumanRendering(
            gym.make("CartPole-v1", render_mode=mode, disable_env_checker=True)
        )
        assert env.render_mode == "human"
        env.reset()

        for _ in range(75):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()

        env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expected env.render_mode to be one of ['rgb_array', 'rgb_array_list', 'depth_array', 'depth_array_list'] but got 'human'"
        ),
    ):
        HumanRendering(env)
    env.close()


@pytest.mark.parametrize("env_id", ["CartPole-v1"])
@pytest.mark.parametrize("num_envs", [1, 3, 9])
@pytest.mark.parametrize("screen_size", [None])
def test_human_rendering_manual(env_id, num_envs, screen_size):
    env = HumanRendering(
        gym.make("CartPole-v1", render_mode="rgb_array", disable_env_checker=True),
        auto_rendering=False,
    )
    assert env.render_mode == "human"
    env.reset()

    for _ in range(75):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    rendering = env.render()
    # output should match mode
    assert isinstance(rendering, np.ndarray)

    env.close()
