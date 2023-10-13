from building_energy_storage_simulation.building_simulation import BuildingSimulation

from building_energy_storage_simulation import Environment
import pytest
import numpy as np


@pytest.fixture(scope='module')
def building_simulation():
    return BuildingSimulation(electricity_load_profile=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              solar_generation_profile=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                              electricity_price=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              max_battery_charge_per_timestep=20)


def test_environment_noop_step(building_simulation):
    """
    Perfect forecast of energy consumption of time step t+1 equals the actual energy consumption of that time step
    if the energy is not charged or discharged.
    """
    env = Environment(building_simulation=building_simulation, num_forecasting_steps=3, max_timesteps=6)
    initial_obs = env.reset()
    obs = env.step(0)
    assert initial_obs[0][2] == obs[0][1]


@pytest.mark.parametrize(
    "data_profile_length, num_forecasting_steps", [(2, 1), (9, 0)]
)
def test_terminated_at_timelimit_reached(data_profile_length, num_forecasting_steps):
    dummy_profile = np.zeros(data_profile_length)
    building_sim = BuildingSimulation(electricity_price=dummy_profile,
                                      solar_generation_profile=dummy_profile,
                                      electricity_load_profile=dummy_profile)
    env = Environment(building_simulation=building_sim,
                      num_forecasting_steps=num_forecasting_steps,
                      max_timesteps=data_profile_length - num_forecasting_steps)
    env.reset()
    print(range(data_profile_length - num_forecasting_steps))
    for i in range(data_profile_length - num_forecasting_steps):
        obs, reward, terminated, trunc, info = env.step(0)
    assert terminated is True


def test_observation_size(building_simulation):
    env = Environment(building_simulation=building_simulation, max_timesteps=5, num_forecasting_steps=4)
    initial_obs, info = env.reset()
    obs, reward, terminated, trunc, info = env.step(0)
    assert len(obs) == 13


def test_initial_obs_step_obs_same_size(building_simulation):
    env = Environment(building_simulation=building_simulation, max_timesteps=5, num_forecasting_steps=4)
    initial_obs, info = env.reset()
    obs, reward, terminated, trunc, info = env.step(0)
    assert len(obs) == len(initial_obs)


def test_max_battery_charge_per_timestep(building_simulation):
    env = Environment(building_simulation=building_simulation, max_timesteps=5, num_forecasting_steps=4)
    initial_obs, info = env.reset()
    obs, reward, terminated, trunc, info = env.step(1)
    # Position 0 in observation is the state_of_charge. Check, if the state of charge increased by MAX_BATTERY_CHARGE
    # when fully charging
    assert obs[0] == initial_obs[0] + 20


def test_reset(building_simulation):
    env = Environment(building_simulation=building_simulation, max_timesteps=5, num_forecasting_steps=4)
    env.reset()
    env.step(1)
    env.reset()
    assert env.building_simulation.step_count == 0
    assert env.building_simulation.battery.state_of_charge == 0


def test_default_initialization_runs_without_throwing():
    sim = BuildingSimulation()
    env = Environment(sim)
    env.reset()
    env.step(1)
    assert env.building_simulation.step_count == 1


def test_set_random_first_time_step_always_0_for_data_profile_length_2():
    dummy_profile = [0, 0]
    sim = BuildingSimulation(electricity_price=dummy_profile,
                             electricity_load_profile=dummy_profile,
                             solar_generation_profile=dummy_profile)
    env = Environment(sim, randomize_start_time_step=True, max_timesteps=1, num_forecasting_steps=1)
    env.reset()
    assert env.building_simulation.start_index == 0


def test_set_random_first_time_step():
    dummy_profile = np.zeros(1000)
    sim = BuildingSimulation(electricity_price=dummy_profile,
                             electricity_load_profile=dummy_profile,
                             solar_generation_profile=dummy_profile)
    env = Environment(sim, randomize_start_time_step=True, max_timesteps=1, num_forecasting_steps=1)
    env.reset()
    # This test is very unlikely to fail ;)
    assert env.building_simulation.start_index != 0


@pytest.mark.parametrize(
    "reset", [True, False]
)
def test_forecasts_are_randomized_in_observation(reset):
    dummy_profile = np.zeros(10)
    building_sim = BuildingSimulation(electricity_price=dummy_profile,
                                      solar_generation_profile=dummy_profile,
                                      electricity_load_profile=dummy_profile)
    env = Environment(building_simulation=building_sim,
                      num_forecasting_steps=4,
                      max_timesteps=6,
                      randomize_forecasts_in_observation=True)

    if reset:
        obs, _ = env.reset()
    else:
        env.reset()
        obs, _, _, _, _ = env.step(0)

    load_forecast = np.array(obs[1:5])
    generation_forecast = obs[5:9]
    price_forecast = obs[9:14]
    assert not np.array_equal(generation_forecast, load_forecast)
    assert not np.array_equal(price_forecast, load_forecast)
    assert not np.array_equal(price_forecast, dummy_profile)

