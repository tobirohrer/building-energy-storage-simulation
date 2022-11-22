from building_energy_storage_simulation import Environment


def test_environment_noop_step():
    """
    Perfect forecast of energy consumption of time step t+1 equals the actual energy consumption of that time step
    if the energy is not charged or discharged.
    """
    env = Environment()
    initial_obs = env.reset()
    obs = env.step(0)
    assert initial_obs[0][1] == obs[0][0]


def test_terminated_at_timelimit_reached():
    env = Environment(max_timesteps=10)
    for i in range(10):
        obs, reward, terminated, trunc, info = env.step(0)
    assert terminated is True
