from building_energy_storage_simulation import Simulation


def test_energy_consumption_is_trimmed_to_0():
    sim = Simulation()
    # Setting load of next time step to 0kWh
    sim.electricity_load_profile = [0]
    # Setting solar generation of next time step to 100kWh
    sim.solar_generation_profile = [100]
    # Don´t charge the battery, meaning don´t do anything with the 100kWh energy we gained from the solar system.
    electricity_consumption, excess_energy = sim.simulate_one_step(0)
    # Still the consumption is 0, as we loose excess electricity which we do not use to charge the battery.
    assert electricity_consumption == 0
