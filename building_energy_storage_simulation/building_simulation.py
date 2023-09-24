from typing import Tuple, Iterable

import numpy as np

from building_energy_storage_simulation.battery import Battery
from building_energy_storage_simulation.utils import load_profile


class BuildingSimulation:
    """
    Represents the simulation of the building and wires the data profiles and the usage of the battery together.
    
    :param electricity_load_profile: Load profile in kWh per time step.
    :type electricity_load_profile: Iterable
    :param solar_generation_profile: Generation profile in kWh per time step.
    :type solar_generation_profile: Iterable
    :param battery_capacity: The capacity of the battery in kWh.
    :type battery_capacity: float
    :param max_battery_charge_per_timestep: Maximum amount of energy (kWh) which can be obtained from the battery or
        which can be used to charge the battery in one time step.
    :type max_battery_charge_per_timestep: float
    """

    def __init__(self,
                 electricity_load_profile: Iterable = load_profile('electricity_load_profile.csv', 'Load [kWh]'),
                 solar_generation_profile: Iterable = load_profile('solar_generation_profile.csv', 'Generation [kWh]'),
                 battery_capacity: float = 100,
                 max_battery_charge_per_timestep: float = 20
                 ):
        self.electricity_load_profile = np.array(electricity_load_profile)
        self.solar_generation_profile = np.array(solar_generation_profile)
        self.battery = Battery(capacity=battery_capacity,
                               max_battery_charge_per_timestep=max_battery_charge_per_timestep)

        assert len(self.solar_generation_profile) == len(self.electricity_load_profile), \
            "Solar generation profile and electricity load profile must be of the same length."

        self.step_count = 0
        self.start_index = 0
        pass

    def reset(self):
        """

        1. Resetting the state of the building by calling `reset()` method from the building class.
        2. Resetting the `step_count` to 0. The `step_count` is used for temporal orientation in the electricity
           load and solar generation profile.

        """

        self.battery.reset()
        self.step_count = 0
        pass

    def simulate_one_step(self, action: float) -> Tuple[float, float]:
        """
        Performs one simulation step by:
            1. Charging or discharging the battery depending on the amount.
            2. Calculating the amount of energy consumed by the building in this time step.
            3. Trimming the amount of energy to 0, in case it is negative.
            4. Calculating the amount of excess energy which is considered lost.
            5. Increasing the step counter.

        :param action: Fraction of energy to be stored or retrieved from the battery. The action lies in [-1;1]. The
            action represents the fraction of `max_battery_charge_per_timestep` which should be used to charge or
            discharge the battery. 1 represents the maximum possible amount of energy which can be used to charge the
             battery per time step.
        :type action: float
        :returns:
            Tuple of:
                1. Amount of energy consumed in this time step. This is calculated by: `battery_energy`
                   + `electricity_load` - `solar_generation`. Note that negative values are trimmed to 0. This means, that energy
                   can not be "gained". Excess energy from the solar energy system which is not used
                   to charge the battery is considered lost. Better use it to charge the battery ;-)
                2. Amount of excess energy which is considered lost.

        :rtype: (float, float)
        """
        electricity_load_of_this_timestep = self.electricity_load_profile[self.start_index + self.step_count]
        solar_generation_of_this_timestep = self.solar_generation_profile[self.start_index + self.step_count]

        electricity_consumed_for_battery = self.battery.use(action * self.battery.max_battery_charge_per_timestep)
        electricity_consumption = electricity_consumed_for_battery + electricity_load_of_this_timestep - \
                                  solar_generation_of_this_timestep
        excess_energy = 0
        if electricity_consumption < 0:
            excess_energy = -1 * electricity_consumption
            electricity_consumption = 0
        self.step_count += 1
        return electricity_consumption, excess_energy
