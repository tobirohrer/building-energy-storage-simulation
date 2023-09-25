from typing import Tuple, Optional, Union, List
import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from building_energy_storage_simulation.building_simulation import BuildingSimulation


class Environment(gym.Env):
    """
    Wraps the simulation as `gymnasium` environment, so it can be used easily for reinforcement learning.

    :param max_timesteps: The number of steps after which the environment terminates
    :type max_timesteps: int
    :param num_forecasting_steps: The number of timesteps into the future included in the forecast. Note that the
       forecast is perfect.
    :type num_forecasting_steps: int
    :param building_simulation: Instance of `BuildingSimulation` to be wrapped as `gymnasium` environment.
    :type building_simulation: BuildingSimulation
    """

    def __init__(self,
                 building_simulation: BuildingSimulation,
                 max_timesteps: int = 2000,
                 num_forecasting_steps: int = 4,
                 ):

        self.building_simulation = building_simulation
        self.max_timesteps = max_timesteps
        self.num_forecasting_steps = num_forecasting_steps

        assert self.max_timesteps + self.num_forecasting_steps < len(self.building_simulation.solar_generation_profile), \
            "`max_timesteps` plus the forecast length cannot be greater than the length of the simulation profile."

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Using np.inf as bounds as the observations must be rescaled externally anyways. E.g. Using the VecNormalize
        # wrapper from StableBaselines3
        # (see https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)
        self.observation_space = gym.spaces.Box(shape=(self.num_forecasting_steps * 2 + 1,),
                                                low=-np.inf,
                                                high=np.inf,
                                                dtype=np.float32)
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """
        Not implemented (yet).
        """

        pass

    def reset(self, seed=None, options=None) -> Tuple[ObsType, dict]:
        """
        Resetting the state of the simulation by calling `reset()` method from the simulation class.

        :returns:
            Tuple of:
                1. An observation
                2. Empty hashmap: {}

        :rtype: (observation, dict)
        """

        self.building_simulation.reset()
        return self.get_observation(), {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        Perform one step, which is done by:

        1. Performing one `simulate_one_step()`
        2. Calculating the reward
        3. Retrieving the observation

        :param action: Fraction of energy to be stored or retrieved from the battery. The action lies in [-1;1]. The
            action represents the fraction of `max_battery_charge_per_timestep` which should be used to charge or
            discharge the battery. 1 represents the maximum possible amount of energy which can be used to charge the
             battery per time step.
        :type action: float
        :returns:
            Tuple of:
                1. observation
                2. reward
                3. terminated. If true, the episode is over.
                4. truncated. Is always false, it is not implemented yet.
                5. Additional Information about the `electricity_comsumption` and the `excess_energy` of the current
                   time step

        :rtype: (observation, float, bool, bool, dict)
        """

        if hasattr(action, "__len__"):
            action = action[0]
        electricity_consumption, excess_energy = self.building_simulation.simulate_one_step(action)
        reward = Environment.calc_reward(electricity_consumption, excess_energy)
        observation = self.get_observation()
        return observation, reward, self.get_terminated(), False, {'electricity_consumption': electricity_consumption,
                                                                   'excess_energy': excess_energy}

    def get_terminated(self):
        if self.building_simulation.step_count > self.max_timesteps:
            return True
        return False

    def get_observation(self):
        current_index = self.building_simulation.start_index + self.building_simulation.step_count
        electric_load_forecast = self.building_simulation.electricity_load_profile[current_index: current_index +
                                                                                                  self.num_forecasting_steps]
        solar_gen_forecast = self.building_simulation.solar_generation_profile[current_index: current_index +
                                                                                              self.num_forecasting_steps]
        return np.concatenate(([self.building_simulation.battery.state_of_charge],
                               electric_load_forecast,
                               solar_gen_forecast),
                              axis=0)

    @staticmethod
    def calc_reward(electricity_consumption, excess_energy):
        return -1 * electricity_consumption
