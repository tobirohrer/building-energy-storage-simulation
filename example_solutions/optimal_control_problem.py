import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

from helper import read_data, TEST_INDEX_END, TEST_INDEX_START, BATTERY_CAPACITY, BATTERY_POWER


def build_optimization_problem(residual_fixed_load, price, soc, battery_power, battery_capacity, delta_time_hours=1):
    time = range(len(residual_fixed_load))
    soc_time = range(len(residual_fixed_load) + 1)
    max_power_charge = battery_power
    max_power_discharge = -1 * battery_power
    max_soc = 100
    min_soc = 0
    soc_init = soc
    energy_capacity = battery_capacity

    m = pyo.AbstractModel()
    m.power = pyo.Var(time, domain=pyo.Reals, bounds=(max_power_discharge, max_power_charge))
    m.soc = pyo.Var(soc_time, bounds=(min_soc, max_soc))

    def obj_expression(m):
        # pyo.log to make the objective expression smooth and therefore solvable
        return sum([price[i] * pyo.log(1 + pyo.exp((m.power[i] + residual_fixed_load[i]))) for i in time])

    m.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

    def soc_start_rule(m):
        return m.soc[0] == soc_init

    m.soc_start = pyo.Constraint(rule=soc_start_rule)

    def soc_constraint_rule(m, i):
        # Define the system dynamics as constraint
        return m.soc[i + 1] == float(100) * delta_time_hours * (m.power[i]) / energy_capacity + m.soc[i]

    m.soc_constraints = pyo.Constraint(time, rule=soc_constraint_rule)

    return m.create_instance()


if __name__ == "__main__":
    solver = pyo.SolverFactory('ipopt')

    load, price, generation = read_data()

    load_eval = load[TEST_INDEX_START:TEST_INDEX_END]
    price_eval = price[TEST_INDEX_START:TEST_INDEX_END]
    generation_eval = generation[TEST_INDEX_START:TEST_INDEX_END]

    residual_fixed_load_eval = load_eval - generation_eval
    time = range(len(residual_fixed_load_eval))

    m = build_optimization_problem(residual_fixed_load_eval,
                                   price_eval,
                                   soc=0,
                                   battery_power=BATTERY_POWER,
                                   battery_capacity=BATTERY_CAPACITY)
    solver.solve(m, tee=True)
    t = [time[i] for i in time]

    baseline_cost = sum(residual_fixed_load_eval[residual_fixed_load_eval > 0] * price_eval[residual_fixed_load_eval > 0])
    augmented_load = residual_fixed_load_eval + np.array([(pyo.value(m.power[i])) for i in time])
    cost = sum(augmented_load[augmented_load > 0] * price_eval[augmented_load > 0])

    print('baseline cost: ' + str(baseline_cost))
    print('cost: ' + str(cost))
    print('savings in %: ' + str(1 - cost/baseline_cost))

    fig1 = plt.figure()
    ax = plt.subplot()
    ax.plot([(residual_fixed_load_eval[i]) for i in time], label='Residual Load')
    ax.plot(augmented_load, label='Augmented Load')
    ax.plot(price_eval, '--', label='Price')
    ax.plot([(pyo.value(m.power[i])) for i in time], label='Battery Power')
    plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
    plt.xlabel('Time Step')
    ax.legend()
    ax.grid()
    plt.show()
