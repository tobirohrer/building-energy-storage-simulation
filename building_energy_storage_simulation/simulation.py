from typing import Tuple, Optional, Union, List
import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame

from building_energy_storage_simulation.building import Building
from building_energy_storage_simulation.utils import load_profile


class Simulation(gym.Env):
    NUM_FORECAST_STEPS = 4
    MAX_TIMESTEPS = 2000

    def __init__(self,
                 electricity_load_profile_file_name: str = 'electricity_load_profile.csv',
                 solar_generation_profile_file_name: str = 'solar_generation_profile.csv',
                 building: Building = Building()):
        """
        Constructor initializing parameter_a with the given value.
        :param building:
        :type building: Building
        :returns: Nothing
        :rtype: None
        """
        self.building = building
        self.electricity_load_profile = load_profile(electricity_load_profile_file_name, 'Load [kWh]')
        self.solar_generation_profile = load_profile(solar_generation_profile_file_name, 'Inverter Power (W)')
        # Solar Generation Profile is in W per 1KW of Solar power installed
        self.solar_generation_profile = self.solar_generation_profile * self.building.solar_power_installed / 1000
        self.step_count = 0
        self.start_index = 0
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.building = Building()
        self.step_count = 0
        return self.get_observation(), {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        electricity_load_of_this_timestep = self.electricity_load_profile[self.start_index + self.step_count]
        solar_generation_of_this_timestep = self.solar_generation_profile[self.start_index + self.step_count]

        electricity_consumed_for_battery = self.building.battery.use(action)
        electricity_consumption = electricity_consumed_for_battery + electricity_load_of_this_timestep - \
                                  solar_generation_of_this_timestep
        reward = Simulation.calc_reward(electricity_consumption)
        observation = self.get_observation()
        self.step_count += 1
        return observation, reward, self.get_terminated(), False, {}

    def get_terminated(self):
        if self.step_count > self.MAX_TIMESTEPS:
            return True
        return False

    def get_observation(self):
        current_index = self.start_index + self.step_count
        electric_load_forecast = self.electricity_load_profile[current_index: current_index + self.NUM_FORECAST_STEPS]
        solar_gen_forecast = self.solar_generation_profile[current_index: current_index + self.NUM_FORECAST_STEPS]
        return np.concatenate((electric_load_forecast, solar_gen_forecast), axis=0)

    @staticmethod
    def calc_reward(electricity_consumption):
        return electricity_consumption
