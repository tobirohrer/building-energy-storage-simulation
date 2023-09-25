from building_energy_storage_simulation import BuildingSimulation


def test_energy_consumption_is_trimmed_to_0():
    sim = BuildingSimulation(electricity_load_profile=[0], solar_generation_profile=[100])
    # Don´t charge the battery, meaning don´t do anything with the 100kWh energy we gained from the solar system.
    electricity_consumption, excess_energy = sim.simulate_one_step(0)
    # Still the consumption is 0, as we loose excess electricity which we do not use to charge the battery.
    assert electricity_consumption == 0


def test_simulation_reads_default_data_profiles():
    sim = BuildingSimulation()
    assert len(sim.electricity_load_profile) == 8760
    assert len(sim.solar_generation_profile) == 8760
