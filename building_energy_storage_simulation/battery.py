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

    def use(self, amount):
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
