"""Functional to Environment compatibility."""
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from gymnasium.functional import ActType, FuncEnv, StateType
from gymnasium.utils import seeding
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.jax_to_numpy import jax_to_numpy


class FunctionalJaxEnv(gym.Env):
    """A conversion layer for jax-based environments."""

    state: StateType
    rng: jrng.PRNGKey

    def __init__(
        self,
        func_env: FuncEnv,
        metadata: dict[str, Any] | None = None,
        render_mode: str | None = None,
        reward_range: tuple[float, float] = (-float("inf"), float("inf")),
        spec: EnvSpec | None = None,
    ):
        """Initialize the environment from a FuncEnv."""
        if metadata is None:
            metadata = {"render_mode": []}

        self.func_env = func_env
        self.params = {}

        self.observation_space = func_env.observation_space
        self.action_space = func_env.action_space

        self.metadata = metadata
        self.render_mode = render_mode
        self.reward_range = reward_range

        self.spec = spec

        self._is_box_action_space = isinstance(self.action_space, gym.spaces.Box)

        if self.render_mode == "rgb_array":
            self.render_state = self.func_env.render_init()
        else:
            self.render_state = None

        np_random, _ = seeding.np_random()
        seed = np_random.integers(0, 2**32 - 1, dtype="uint32")

        self.rng = jrng.PRNGKey(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment using the seed."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = jrng.PRNGKey(seed)

        rng, self.rng = jrng.split(self.rng)

        self.params = {} if options is None else options.get("params")
        self.state = self.func_env.initial(rng=rng, params=self.params)
        obs = self.func_env.observation(self.state, params=self.params)
        info = self.func_env.state_info(self.state, params=self.params)

        obs = jax_to_numpy(obs)

        return obs, info

    def step(self, action: ActType):
        """Steps through the environment using the action."""
        if self._is_box_action_space:
            assert isinstance(self.action_space, gym.spaces.Box)  # For typing
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:  # Discrete
            # For now we assume jax envs don't use complex spaces
            err_msg = f"{action!r} ({type(action)}) invalid"
            assert self.action_space.contains(action), err_msg

        rng, self.rng = jrng.split(self.rng)

        next_state = self.func_env.transition(
            self.state, action, rng, params=self.params
        )
        observation = self.func_env.observation(next_state, params=self.params)
        reward = self.func_env.reward(
            self.state, action, next_state, params=self.params
        )
        terminated = self.func_env.terminal(next_state, params=self.params)
        info = self.func_env.transition_info(
            self.state, action, next_state, params=self.params
        )
        self.state = next_state

        observation = jax_to_numpy(observation)

        return observation, float(reward), bool(terminated), False, info

    def render(self):
        """Returns the render state if `render_mode` is "rgb_array"."""
        if self.render_mode == "rgb_array":
            self.render_state, image = self.func_env.render_image(
                self.state, self.render_state
            )
            return image
        else:
            raise NotImplementedError

    def close(self):
        """Closes the environments and render state if set."""
        if self.render_state is not None:
            self.func_env.render_close(self.render_state)
            self.render_state = None


class FunctionalJaxVectorEnv(gym.vector.VectorEnv):
    """A vector env implementation for functional Jax envs."""

    state: StateType
    rng: jrng.PRNGKey

    def __init__(
        self,
        func_env: FuncEnv,
        num_envs: int,
        max_episode_steps: int = 0,
        metadata: dict[str, Any] | None = None,
        render_mode: str | None = None,
        reward_range: tuple[float, float] = (-float("inf"), float("inf")),
        spec: EnvSpec | None = None,
    ):
        """Initialize the environment from a FuncEnv."""
        super().__init__()
        if metadata is None:
            metadata = {}
        self.func_env = func_env
        self.params = {}
        self.num_envs = num_envs

        self.single_observation_space = func_env.observation_space
        self.single_action_space = func_env.action_space
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.metadata = metadata
        self.render_mode = render_mode
        self.reward_range = reward_range
        self.spec = spec
        self.time_limit = max_episode_steps

        self.steps = jnp.zeros(self.num_envs, dtype=jnp.int32)

        self.autoreset_envs = jnp.zeros(self.num_envs, dtype=jnp.bool_)

        self._is_box_action_space = isinstance(self.action_space, gym.spaces.Box)

        if self.render_mode == "rgb_array":
            self.render_state = self.func_env.render_init()
        else:
            self.render_state = None

        np_random, _ = seeding.np_random()
        seed = np_random.integers(0, 2**32 - 1, dtype="uint32")

        self.rng = jrng.PRNGKey(seed)

        self.func_env.transform(jax.vmap)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = jrng.PRNGKey(seed)

        rng, self.rng = jrng.split(self.rng)

        rng = jrng.split(rng, self.num_envs)

        self.params = {} if options is None else options.get("params")
        self.state = self.func_env.initial(rng=rng, params=self.params)
        obs = self.func_env.observation(self.state, params=self.params)
        info = self.func_env.state_info(self.state, params=self.params)

        self.steps = jnp.zeros(self.num_envs, dtype=jnp.int32)

        obs = jax_to_numpy(obs)

        return obs, info

    @partial(jax.jit, static_argnames=("self",))
    def step(self, action: ActType):
        """Steps through the environment using the action."""
        # These will be optimized to always give same result
        if self._is_box_action_space:
            assert isinstance(self.action_space, gym.spaces.Box)  # For typing
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:  # Discrete
            # For now we assume jax envs don't use complex spaces
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid"

        self.steps = jnp.where(self.autoreset_envs, 0, self.steps + 1)

        rng, self.rng = jrng.split(self.rng)
        rng = jrng.split(rng, self.num_envs)

        # compute all, can probably be optimized away if it is not using :attr:`rng`
        new_initials = self.func_env.initial(rng, params=self.params)
        next_state = jnp.where(
            self.autoreset_envs,
            new_initials,
            self.func_env.transition(self.state, action, rng, params=self.params),
        )
        reward = jnp.where(
            self.autoreset_envs,
            0,  # restarting, jumping from end state to initial state...
            self.func_env.reward(self.state, action, next_state, params=self.params),
        )

        terminated = self.func_env.terminal(next_state, params=self.params)
        truncated = (
            self.steps >= self.time_limit
            if self.time_limit > 0
            else jnp.zeros_like(terminated)
        )

        info = self.func_env.transition_info(
            self.state, action, next_state, params=self.params
        )

        done = jnp.logical_or(terminated, truncated)
        self.autoreset_envs = done

        observation = self.func_env.observation(next_state, params=self.params)
        observation = jax_to_numpy(observation)

        reward = jax_to_numpy(reward)

        terminated = jax_to_numpy(terminated)
        truncated = jax_to_numpy(truncated)

        self.state = next_state

        return observation, reward, terminated, truncated, info

    def render(self):
        """Returns the render state if `render_mode` is "rgb_array"."""
        if self.render_mode == "rgb_array":
            self.render_state, image = self.func_env.render_image(
                self.state, self.render_state, params=self.params
            )
            return image
        else:
            raise NotImplementedError

    def close(self):
        """Closes the environments and render state if set."""
        if self.render_state is not None:
            self.func_env.render_close(self.render_state, params=self.params)
            self.render_state = None
