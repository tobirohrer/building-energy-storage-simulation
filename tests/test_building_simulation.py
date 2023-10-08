import pytest

from building_energy_storage_simulation import BuildingSimulation


def test_energy_consumption_is_trimmed_to_0():
    sim = BuildingSimulation(electricity_load_profile=[0], solar_generation_profile=[100], electricity_price=[1])
    # Don´t charge the battery, meaning don´t do anything with the 100kWh energy we gained from the solar system.
    electricity_consumption, excess_energy = sim.simulate_one_step(0)
    # Still the consumption is 0, as we loose excess electricity which we do not use to charge the battery.
    assert electricity_consumption == 0


def test_simulation_reads_default_data_profiles():
    sim = BuildingSimulation()
    assert len(sim.electricity_load_profile) == 8760
    assert len(sim.solar_generation_profile) == 8760


@pytest.mark.parametrize(
    "electricity_price", [1, 1.1]
)
def test_simulation_scalar_electricity_price_converted_into_profile(electricity_price):
    sim = BuildingSimulation(electricity_load_profile=[0, 0, 0],
                             solar_generation_profile=[0, 0, 0],
                             electricity_price=electricity_price)
    assert len(sim.electricity_price) == 3


def test_simulation_throws_when_data_profiles_have_unequal_length():
    with pytest.raises(Exception) as exception_info:
        sim = BuildingSimulation(electricity_load_profile=[0, 0, 0],
                                 solar_generation_profile=[0, 0])
    assert exception_info.errisinstance(ValueError)
