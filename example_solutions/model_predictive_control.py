import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from building_energy_storage_simulation import BuildingSimulation, Environment

load = pd.read_csv('building_energy_storage_simulation/data/preprocessed/electricity_load_profile.csv')['Load [kWh]']
price = pd.read_csv('building_energy_storage_simulation/data/preprocessed/electricity_price_profile.csv')[
    'Day Ahead Auction']
generation = pd.read_csv('building_energy_storage_simulation/data/preprocessed/solar_generation_profile.csv')[
    'Generation [kWh]']
load = np.array(load)
price = np.array(price)
generation = np.array(generation)

DELTA_TIME_HOURS = 1
FORECAST_LENGTH = 4


def normalize_to_minus_one_to_one(x, min_value, max_value):
    return -1 + 2 * (x - min_value) / (max_value - min_value)


def build_optimization_problem(residual_fixed_load, price, soc):
    # model parameter initilization
    time = range(len(residual_fixed_load))
    soc_time = range(len(residual_fixed_load) + 1)
    max_power_charge = 50
    max_power_discharge = -50
    max_soc = 100
    min_soc = 0
    soc_init = soc
    energy_capacity = 200

    m = pyo.AbstractModel()
    m.power = pyo.Var(time, domain=pyo.Reals, bounds=(max_power_discharge, max_power_charge))
    m.soc = pyo.Var(soc_time, bounds=(min_soc, max_soc))

    def obj_expression(m):
        # ToDo: how to do `max(0, m.pe_c[i] + m.pe_d[i] + residual_fixed_load[i])`
        return sum([price[i] * pyo.log(1 + pyo.exp((m.power[i] + residual_fixed_load[i]))) for i in time])

    m.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

    def soc_start_rule(m):
        return m.soc[0] == soc_init

    m.soc_start = pyo.Constraint(rule=soc_start_rule)

    def soc_constraint_rule(m, i):
        return m.soc[i + 1] == float(100) * DELTA_TIME_HOURS * (m.power[i]) / energy_capacity + m.soc[i]

    m.soc_constraints = pyo.Constraint(time, rule=soc_constraint_rule)

    return m.create_instance()


sim = BuildingSimulation()
env = Environment(sim)
solver = pyo.SolverFactory('ipopt')

obs, info = env.reset()
done = False

actions, residual_loads, prices = ([], [], [])

while not done:
    load_forecast = obs[1: FORECAST_LENGTH+1]
    generation_forecast = obs[FORECAST_LENGTH+1: 2*FORECAST_LENGTH+1]
    price_forecast = obs[2*FORECAST_LENGTH+1: 3*FORECAST_LENGTH+1]
    residual_load_forecast = load_forecast - generation_forecast
    soc = obs[0]

    instance = build_optimization_problem(residual_fixed_load=residual_load_forecast, price=price_forecast, soc=soc)

    solver.solve(instance, tee=True)
    action = pyo.value(instance.power[0])
    obs, reward, done, _, info = env.step(action)
    actions.append(action)
    residual_loads.append(residual_load_forecast[0])
    prices.append(price_forecast[0])



residual_fixed_load = load - generation
residual_fixed_load = residual_fixed_load[200:400]
price = price[200:400]
time = range(len(residual_fixed_load))

m = build_optimization_problem(residual_fixed_load, price, soc=0)
solver.solve(m, tee=True)
t = [time[i] * DELTA_TIME_HOURS for i in time]

fig1 = plt.figure()
ax = plt.subplot()
Line1 = ax.plot(time, [(residual_fixed_load[i]) for i in time], label='Residual Load')
Line2 = ax.plot(time, [(pyo.value(m.power[i]) + residual_fixed_load[i]) for i in time], label='Augmented Load')
Line3 = ax.plot(time, price, '--', label='Price')
Line4 = ax.plot(time, [(pyo.value(m.power[i])) for i in time], label='Battery Power')
plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
plt.xlabel('Time Step')
ax.legend()
ax.grid()
plt.show()
