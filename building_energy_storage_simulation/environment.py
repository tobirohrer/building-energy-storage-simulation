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
    :param max_timesteps: The number of steps after which the environment terminates
    :type max_timesteps: int
    :param num_forecasting_steps: The number of timesteps into the future included in the forecast. Note that the
    forecast is perfect.
    """

    def __init__(self,
                 simulation: Simulation = Simulation(),
                 max_timesteps: int = 2000,
                 num_forecasting_steps: int = 4):
        self.max_timesteps = max_timesteps
        self.num_forecasting_steps = num_forecasting_steps
        self.simulation = simulation
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Using np.inf as bounds as the observations must be rescaled externally anyways. E.g. Using the VecNormalize
        # wrapper from StableBaselines3
        # (see https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)
        self.observation_space = gym.spaces.Box(shape=(self.num_forecasting_steps * 2,),
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
        if self.simulation.step_count > self.max_timesteps:
            return True
        return False

    def get_observation(self):
        current_index = self.simulation.start_index + self.simulation.step_count
        electric_load_forecast = self.simulation.electricity_load_profile[current_index: current_index +
                                                                                         self.num_forecasting_steps]
        solar_gen_forecast = self.simulation.solar_generation_profile[current_index: current_index +
                                                                                     self.num_forecasting_steps]
        return np.concatenate(([self.simulation.building.battery.state_of_charge],
                               electric_load_forecast,
                               solar_gen_forecast),
                              axis=0)

    @staticmethod
    def calc_reward(electricity_consumption):
        return -1 * electricity_consumption
