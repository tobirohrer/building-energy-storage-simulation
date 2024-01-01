import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from building_energy_storage_simulation import BuildingSimulation, Environment
from example_solutions.helper import read_data, TEST_INDEX_START, BATTERY_CAPACITY, BATTERY_POWER
from example_solutions.observation_wrapper import ObservationWrapper

NUM_FORECAST_STEPS = 8
RESULT_PATH = 'rl_example/'

if __name__ == "__main__":
    os.makedirs(RESULT_PATH, exist_ok=True)

    load, price, generation = read_data()
    load_train = load[:TEST_INDEX_START]
    price_train = price[:TEST_INDEX_START]
    generation_train = generation[:TEST_INDEX_START]

    # Create Training Environment
    sim = BuildingSimulation(electricity_load_profile=load_train,
                             solar_generation_profile=generation_train,
                             electricity_price=price_train,
                             max_battery_charge_per_timestep=BATTERY_POWER,
                             battery_capacity=BATTERY_CAPACITY)

    env = Environment(sim, num_forecasting_steps=NUM_FORECAST_STEPS, max_timesteps=len(load_train) - NUM_FORECAST_STEPS)
    # ObservationWrapper combines forecast of load and generation to one residual load forecast
    env = ObservationWrapper(env, NUM_FORECAST_STEPS)
    initial_obs, info = env.reset()
    print(initial_obs)

    # Wrap with Monitor() so a log of the training is saved
    env = Monitor(env, filename=RESULT_PATH)
    # Warp with DummyVecEnc() so the observations and reward can be normalized using VecNormalize()
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Train :-)
    model = SAC("MlpPolicy", env, verbose=1, gamma=0.95)
    model.learn(total_timesteps=200_000)
    # Store the trained Model and environment stats (which are needed as we are standardizing the observations and reward using VecNormalize())
    model.save(RESULT_PATH + 'model')
    env.save(RESULT_PATH + 'env.pkl')