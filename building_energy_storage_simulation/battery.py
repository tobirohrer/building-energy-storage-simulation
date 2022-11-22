class Battery:
    """
    Battery class.

    :param capacity:
    :type capacity: float
    :returns: Nothing
    :rtype: None
    """

    def __init__(self, capacity: float = 100, initial_state_of_charge: float = 0):
        self.capacity = capacity
        self.initial_state_of_charge = initial_state_of_charge
        self.state_of_charge = initial_state_of_charge
        pass

    def use(self, action):
        """
        Using means charging or discharging the battery.

        :param action: The action space is in [-1;1]. 1 means fully charging the battery during a time step. -1 means
        fully discharging the battery. 0 means do nothing.
        :type action: float
        :returns: Amount of energy consumed to charge or amount of energy gained by discharging the battery in kWh.
        :rtype: float
        """
        # Action Space is in [-1;1].
        amount = action * self.capacity
        # In case battery would be "more than" fully discharged. This applies only if amount is negative
        if self.state_of_charge + amount < 0:
            electricity_used = self.state_of_charge
            self.state_of_charge = 0
        # In case the battery would be "more than" fully charged.
        # This applies only to the case where amount is positive
        elif self.state_of_charge + amount > self.capacity:
            electricity_used = self.capacity - self.state_of_charge
            self.state_of_charge = self.capacity
        else:
            electricity_used = amount
            self.state_of_charge += amount
        return electricity_used

    def reset(self):
        self.state_of_charge = self.initial_state_of_charge
