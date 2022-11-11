from typing import Tuple, Optional, Union, List
import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from building_energy_storage_simulation.simulation import Simulation


class Environment(gym.Env):
    """
    Wraps the simulation as `gymnasium` environment, so it can be used easily for reinforcement learning.

    :param simulation: Simulation to be used for the environment.
    :type simulation: Simulation
    """

    NUM_FORECAST_STEPS = 4
    MAX_TIMESTEPS = 2000

    def __init__(self,
                 simulation: Simulation = Simulation()):
        self.simulation = simulation
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Using np.inf as bounds as the observations must be rescaled externally anyways. E.g. Using the VecNormalize
        # wrapper from StableBaselines3
        # (see https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)
        self.observation_space = gym.spaces.Box(shape=(Environment.NUM_FORECAST_STEPS*2,),
                                                low=-np.inf,
                                                high=np.inf,
                                                dtype=np.float64)
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.simulation.reset()
        return self.get_observation(), {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        electricity_consumption = self.simulation.simulate_one_step(action)
        reward = Environment.calc_reward(electricity_consumption)
        observation = self.get_observation()
        return observation, reward, self.get_terminated(), False, {}

    def get_terminated(self):
        if self.simulation.step_count > self.MAX_TIMESTEPS:
            return True
        return False

    def get_observation(self):
        current_index = self.simulation.start_index + self.simulation.step_count
        electric_load_forecast = self.simulation.electricity_load_profile[current_index: current_index +
                                                                                         self.NUM_FORECAST_STEPS]
        solar_gen_forecast = self.simulation.solar_generation_profile[current_index: current_index +
                                                                                     self.NUM_FORECAST_STEPS]
        return np.concatenate((electric_load_forecast, solar_gen_forecast), axis=0)

    @staticmethod
    def calc_reward(electricity_consumption):
        return electricity_consumption
