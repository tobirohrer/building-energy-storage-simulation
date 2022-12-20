from building_energy_storage_simulation.battery import Battery


class Building:
    """
    Building class.
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
        self.battery.reset()
