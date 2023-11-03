import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

from building_energy_storage_simulation import BuildingSimulation, Environment
from optimal_control_problem import build_optimization_problem
from helper import read_data, TEST_INDEX_END, TEST_INDEX_START, BATTERY_POWER, BATTERY_CAPACITY

FORECAST_LENGTH = 24


def normalize_to_minus_one_to_one(x, min_value, max_value):
    return -1 + 2 * (x - min_value) / (max_value - min_value)


solver = pyo.SolverFactory('ipopt')

load, price, generation = read_data()
load_eval = load[TEST_INDEX_START:]
price_eval = price[TEST_INDEX_START:]
generation_eval = generation[TEST_INDEX_START:]

num_eval_timesteps = TEST_INDEX_END - TEST_INDEX_START

sim = BuildingSimulation(electricity_load_profile=load_eval,
                         solar_generation_profile=generation_eval,
                         electricity_price=price_eval,
                         max_battery_charge_per_timestep=BATTERY_POWER,
                         battery_capacity=BATTERY_CAPACITY)
env = Environment(sim, num_forecasting_steps=FORECAST_LENGTH, max_timesteps=num_eval_timesteps)

obs, info = env.reset()
done = False

actions, residual_loads, prices = (np.array([]), np.array([]), np.array([]))

t = 0
while not done:
    load_forecast = obs[1: FORECAST_LENGTH + 1]
    generation_forecast = obs[FORECAST_LENGTH + 1: 2 * FORECAST_LENGTH + 1]
    price_forecast = obs[2 * FORECAST_LENGTH + 1: 3 * FORECAST_LENGTH + 1]
    residual_load_forecast = load_forecast - generation_forecast
    soc = obs[0]

    instance = build_optimization_problem(residual_fixed_load=residual_load_forecast,
                                          price=price_forecast,
                                          soc=soc / BATTERY_CAPACITY * 100,  # Convert SOC due to different SOC definitions
                                          battery_capacity=BATTERY_CAPACITY,
                                          battery_power=BATTERY_POWER)

    solver.solve(instance, tee=True)
    action = pyo.value(instance.power[0])
    actions = np.append(actions, action)
    obs, reward, done, _, info = env.step(normalize_to_minus_one_to_one(action, -1 * BATTERY_POWER, BATTERY_POWER))
    residual_loads = np.append(residual_loads, residual_load_forecast[0])
    prices = np.append(prices, price_forecast[0])
    t += 1

baseline_cost = sum(residual_loads[residual_loads > 0] * prices[residual_loads > 0])
augmented_load = residual_loads + actions
cost = sum(augmented_load[augmented_load > 0] * prices[augmented_load > 0])

print('baseline cost: ' + str(baseline_cost))
print('cost: ' + str(cost))
print('savings in %: ' + str(1 - cost/baseline_cost))

time = range(len(actions))

fig1 = plt.figure()
ax = plt.subplot()
ax.plot(residual_loads, label='Residual Load')
ax.plot(residual_loads + actions, label='Augmented Load')
ax.plot(actions, label='Battery Power Applied')
ax.plot(prices, '--', label='Price')
plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
plt.xlabel('Time Step')
ax.legend()
ax.grid()
plt.show()
