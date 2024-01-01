# Plot the training process
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from building_energy_storage_simulation import BuildingSimulation, Environment
from example_solutions.deep_reinforcement_learning.train import RESULT_PATH, NUM_FORECAST_STEPS
from example_solutions.helper import read_data, TEST_INDEX_START, TEST_INDEX_END, BATTERY_POWER, BATTERY_CAPACITY, \
    plot_control_trajectory
from example_solutions.observation_wrapper import ObservationWrapper


def evaluate(env, agent=None):
    # Do the evaluation
    actions, observations, electricity_consumption, price, rewards = ([], [], [], [], [])
    done = False
    obs = env.reset()
    while not done:
        if agent is None:
            action = [[0]]
        else:
            action = [agent.predict(obs, deterministic=True)[0][0]]

        obs, r, done, info = env.step([action[0][0]])

        actions.append(action[0][0])
        original_obs = env.get_original_obs()[0]
        observations.append(original_obs)
        electricity_consumption.append(info[0]['electricity_consumption'])
        price.append(info[0]['electricity_price'])
        rewards.append(r)

    return pd.DataFrame({
        'action': actions,
        'observations': observations,
        'electricity_consumption': electricity_consumption,
        'electricity_price': price,
        'reward': rewards
    })


if __name__ == "__main__":
    # Plot evolution of reward during training
    try:
        plot_results(RESULT_PATH, x_axis='timesteps', task_name='title', num_timesteps=None)
    except:
        print('Training Reward Plot could not be created')

    load, price, generation = read_data()
    load_eval = load[TEST_INDEX_START:]
    price_eval = price[TEST_INDEX_START:]
    generation_eval = generation[TEST_INDEX_START:]

    num_eval_timesteps = TEST_INDEX_END - TEST_INDEX_START

    eval_sim = BuildingSimulation(electricity_load_profile=load_eval,
                                  solar_generation_profile=generation_eval,
                                  electricity_price=price_eval,
                                  max_battery_charge_per_timestep=BATTERY_POWER,
                                  battery_capacity=BATTERY_CAPACITY)

    eval_env = Environment(eval_sim, num_forecasting_steps=NUM_FORECAST_STEPS, max_timesteps=num_eval_timesteps)
    eval_env = ObservationWrapper(eval_env, NUM_FORECAST_STEPS)
    eval_env = DummyVecEnv([lambda: eval_env])
    # It is important to load the environmental statistics here as we use a rolling mean calculation !
    eval_env = VecNormalize.load(RESULT_PATH + 'env.pkl', eval_env)
    eval_env.training = False

    model = SAC.load(RESULT_PATH + 'model')

    trajectory = evaluate(eval_env, model)
    baseline_trajectory = evaluate(eval_env, None)

    cost = sum(trajectory['electricity_price'] * trajectory['electricity_consumption'])
    baseline_cost = sum(baseline_trajectory['electricity_price'] * baseline_trajectory['electricity_consumption'])

    print('baseline cost: ' + str(baseline_cost))
    print('cost: ' + str(cost))
    print('savings in %: ' + str(1 - cost / baseline_cost))

    observation_df = trajectory['observations'].apply(pd.Series)
    augmented_load = observation_df[1] + trajectory['action'] * BATTERY_POWER
    plot_control_trajectory(residual_load=observation_df[1],
                            augmented_load=augmented_load,
                            price=trajectory['electricity_price'],
                            battery_power=trajectory['action'] * BATTERY_POWER)
