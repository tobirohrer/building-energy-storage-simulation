from building_energy_storage_simulation.battery import Battery


class Building:
    """
    Building class.
    """

    def __init__(self, battery: Battery = Battery(), solar_power_installed: float = 140.0):
        """
        Constructor initializing parameter_a with the given value.
        :returns: Nothing
        :rtype: None
        """
        self.battery = battery
        self.solar_power_installed = solar_power_installed
        pass
