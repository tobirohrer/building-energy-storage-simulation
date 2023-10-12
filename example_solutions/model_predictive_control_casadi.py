import casadi as csd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from building_energy_storage_simulation import BuildingSimulation, Environment
from optimal_control_problem_casadi import build_optimization_problem, read_data, DELTA_TIME_HOURS

FORECAST_LENGTH = 24


def normalize_to_minus_one_to_one(x, min_value, max_value):
    return -1 + 2 * (x - min_value) / (max_value - min_value)


load, price, generation = read_data()

sim = BuildingSimulation(max_battery_charge_per_timestep=50, battery_capacity=200)
env = Environment(sim, num_forecasting_steps=FORECAST_LENGTH)

obs, info = env.reset()
done = False

actions, residual_loads, prices = (np.array([]), np.array([]), np.array([]))

t = 0
while t < 50:
    load_forecast = obs[1: FORECAST_LENGTH + 1]
    generation_forecast = obs[FORECAST_LENGTH + 1: 2 * FORECAST_LENGTH + 1]
    price_forecast = obs[2 * FORECAST_LENGTH + 1: 3 * FORECAST_LENGTH + 1]
    residual_load_forecast = load_forecast - generation_forecast
    soc = obs[0]

    trajectory = build_optimization_problem(residual_fixed_load=residual_load_forecast, price=price_forecast, soc=soc)


    action = trajectory[1::2][0]
    actions = np.append(actions, action)
    obs, reward, done, _, info = env.step(normalize_to_minus_one_to_one(action, -50, 50))
    residual_loads = np.append(residual_loads, residual_load_forecast[0])
    prices = np.append(prices, price_forecast[0])
    t += 1


time = range(len(actions))

fig1 = plt.figure()
ax = plt.subplot()
ax.plot(time, residual_loads, label='Residual Load')
ax.plot(time, residual_loads + actions, label='Augmented Load')
ax.plot(time, actions, label='Battery Power Applied')
ax.plot(time, prices, '--', label='Price')
plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
plt.xlabel('Time Step')
ax.legend()
ax.grid()
plt.show()
