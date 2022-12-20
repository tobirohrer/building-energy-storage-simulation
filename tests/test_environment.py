from building_energy_storage_simulation import Environment


def test_environment_noop_step():
    """
    Perfect forecast of energy consumption of time step t+1 equals the actual energy consumption of that time step
    if the energy is not charged or discharged.
    """
    env = Environment()
    initial_obs = env.reset()
    obs = env.step(0)
    assert initial_obs[0][2] == obs[0][1]


def test_terminated_at_timelimit_reached():
    env = Environment(max_timesteps=10)
    env.reset()
    for i in range(11):
        obs, reward, terminated, trunc, info = env.step(0)
        print(terminated)
    assert terminated is True


def test_observation_size():
    env = Environment(num_forecasting_steps=4)
    initial_obs, info = env.reset()
    obs, reward, terminated, trunc, info = env.step(0)
    assert len(obs) == 9


def test_initial_obs_step_obs_same_size():
    env = Environment(num_forecasting_steps=4)
    initial_obs, info = env.reset()
    obs, reward, terminated, trunc, info = env.step(0)
    assert len(obs) == len(initial_obs)


def test_max_battery_charge_per_timestep():
    MAX_BATTERY_CHARGE = 20
    env = Environment(max_battery_charge_per_timestep=MAX_BATTERY_CHARGE)
    initial_obs, info = env.reset()
    obs, reward, terminated, trunc, info = env.step(1)
    # Position 0 in observation is the state_of_charge. Check, if the state of charge increased by MAX_BATTERY_CHARGE
    # when fully charging
    assert obs[0] == initial_obs[0] + MAX_BATTERY_CHARGE
