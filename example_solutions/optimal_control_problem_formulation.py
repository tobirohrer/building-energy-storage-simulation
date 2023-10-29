import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DELTA_TIME_HOURS = 1


def read_data():
    load = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/electricity_load_profile.csv')[
        'Load [kWh]']
    price = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/electricity_price_profile.csv')[
        'Day Ahead Auction']
    generation = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/solar_generation_profile.csv')[
        'Generation [kWh]']
    return np.array(load), np.array(price), np.array(generation)


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
        return sum([price[i] * pyo.log(1 + pyo.exp((m.power[i] + residual_fixed_load[i]))) for i in time])

    m.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

    def soc_start_rule(m):
        return m.soc[0] == soc_init

    m.soc_start = pyo.Constraint(rule=soc_start_rule)

    def soc_constraint_rule(m, i):
        return m.soc[i + 1] == float(100) * DELTA_TIME_HOURS * (m.power[i]) / energy_capacity + m.soc[i]

    m.soc_constraints = pyo.Constraint(time, rule=soc_constraint_rule)

    return m.create_instance()


if __name__ == "__main__":
    solver = pyo.SolverFactory('ipopt')

    load, price, generation = read_data()

    load = load[4380:8700]
    price = price[4380:8700]
    generation = generation[4380:8700]

    residual_fixed_load = load - generation
    time = range(len(residual_fixed_load))

    m = build_optimization_problem(residual_fixed_load, price, soc=0)
    solver.solve(m, tee=True)
    t = [time[i] * DELTA_TIME_HOURS for i in time]

    baseline_cost = sum(residual_fixed_load[residual_fixed_load > 0] * price[residual_fixed_load > 0])
    augmented_load = residual_fixed_load + np.array([(pyo.value(m.power[i])) for i in time])
    cost = sum(augmented_load[augmented_load > 0] * price[augmented_load > 0])

    print('baseline cost: ' + str(baseline_cost))
    print('cost: ' + str(cost))
    print('savings in %: ' + str(cost/baseline_cost))

    fig1 = plt.figure()
    ax = plt.subplot()
    Line1 = ax.plot(time, [(residual_fixed_load[i]) for i in time], label='Residual Load')
    Line2 = ax.plot(time, augmented_load, label='Augmented Load')
    Line3 = ax.plot(time, price, '--', label='Price')
    Line4 = ax.plot(time, [(pyo.value(m.power[i])) for i in time], label='Battery Power')
    plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
    plt.xlabel('Time Step')
    ax.legend()
    ax.grid()
    plt.show()
