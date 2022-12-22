from building_energy_storage_simulation.battery import Battery


class Building:
    """
    Building class.

    :param solar_power_installed: The installed peak photovoltaic power in kWp.
    :type solar_power_installed: float
    :param battery_capacity: The capacity of the battery in kWh.
    :type battery_capacity: float
    :param max_battery_charge_per_timestep: Maximum amount of energy (kWh) which can be obtained from the battery or
        which can be used to charge the battery in one time step.
    :type max_battery_charge_per_timestep: float
    """

    def __init__(self,
                 solar_power_installed: float = 240.0,
                 battery_capacity: float = 100,
                 max_battery_charge_per_timestep: float = 20):
        self.battery = Battery(capacity=battery_capacity,
                               max_battery_charge_per_timestep=max_battery_charge_per_timestep)
        self.solar_power_installed = solar_power_installed
        pass

    def reset(self):
        """
        Resetting the state of the battery by calling `reset()` method from the battery class.
        """
        self.battery.reset()
