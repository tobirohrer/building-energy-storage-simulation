from building_energy_storage_simulation.building_simulation import BuildingSimulation

from building_energy_storage_simulation import Environment
import pytest


@pytest.fixture(scope='module')
def building_simulation():
    return BuildingSimulation(electricity_load_profile=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              solar_generation_profile=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
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


def test_terminated_at_timelimit_reached(building_simulation):
    env = Environment(building_simulation=building_simulation, num_forecasting_steps=0, max_timesteps=9)
    env.reset()
    for i in range(10):
        obs, reward, terminated, trunc, info = env.step(0)
    assert terminated is True


def test_observation_size(building_simulation):
    env = Environment(building_simulation=building_simulation, max_timesteps=5, num_forecasting_steps=4)
    initial_obs, info = env.reset()
    obs, reward, terminated, trunc, info = env.step(0)
    assert len(obs) == 9


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
