from building_energy_storage_simulation import Battery


def test_battery_charge_electricity_usage():
    battery = Battery(capacity=3)
    electricity_used = battery.use(2)
    electricity_used += battery.use(2)
    assert electricity_used == 3


def test_battery_charge_state():
    battery = Battery(capacity=3)
    electricity_used = battery.use(2)
    electricity_used += battery.use(2)
    assert battery.state_of_charge == 3


def test_battery_use():
    battery = Battery(capacity=3, initial_state_of_charge=3)
    electricity_gain = battery.use(-5)
    assert electricity_gain == 3


def test_battery_empty_state_of_charge():
    battery = Battery(capacity=3, initial_state_of_charge=3)
    battery.use(-5)
    assert battery.state_of_charge == 0
