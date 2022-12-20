from building_energy_storage_simulation import Battery


def test_battery_charge_electricity_usage():
    battery = Battery(capacity=100)  # in kWh
    electricity_used = battery.use(10)
    electricity_used += battery.use(10)
    assert electricity_used == 20.0


def test_battery_charge_state():
    battery = Battery(capacity=100)
    electricity_used = battery.use(10)
    electricity_used += battery.use(10)
    assert battery.state_of_charge == 20.0


def test_battery_use():
    battery = Battery(capacity=100, initial_state_of_charge=100)
    electricity_used = battery.use(-10)
    assert electricity_used == -10.0


def test_battery_empty_state_of_charge():
    battery = Battery(capacity=100, initial_state_of_charge=100, max_battery_charge_per_timestep=10)
    battery.use(-10)
    battery.use(-10)
    assert battery.state_of_charge == 80


def test_battery_max_charge():
    battery = Battery(capacity=100, initial_state_of_charge=100, max_battery_charge_per_timestep=10)
    battery.use(-20)
    battery.use(-50)
    assert battery.state_of_charge == 80
