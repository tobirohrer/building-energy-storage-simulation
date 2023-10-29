import math

import casadi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DELTA_TIME_HOURS = 1
HOURS_PER_YEAR = 24*365
energy_capacity = 200  # kWh
replacement_price = energy_capacity * 100  # kWh*€/kWh


def read_data():
    load = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/electricity_load_profile.csv')[
        'Load [kWh]']
    price = 0.4
    generation = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/solar_generation_profile.csv')[
        'Generation [kWh]']
    return np.array(load), np.array(price), np.array(generation)


def build_optimization_problem(residual_fixed_load, price, soc=0):
    # model parameter initialization
    soc_init = soc

    max_power_charge = 50  # kW
    max_power_discharge = -1 * max_power_charge
    max_soc = 1
    min_soc = 0
    min_soh = 0.8
    max_soh = 1
    max_life_time = 5 #years
    max_life_time_hours = max_life_time*HOURS_PER_YEAR
    max_cycles = 10000
    energy_capacity = 200  # kWh
    replacement_price = energy_capacity * 100 #kWh*€/kWh
    w = []  # list of control variables
    w0 = []  # initial guess for state variables
    lbw = []  # lower bound for state variables
    ubw = []  # upper bound for state variables
    J = 0  # initial value for cost function
    g = []  # list of constraints
    lbg = []  # lower bound for constraints
    ubg = []  # upper bound for constraints
    soc_k = casadi.SX.sym("soc_0")  # define first state variable as casadi symbol
    w += [soc_k]  # add first control variable into state vector
    lbw += [soc_init]  # add state bounds, in this case start with initial value
    ubw += [
        soc_init]  # also the upper one, which is also the same as the lower bound since this is an equality constraint
    w0 += [soc_init]  # use initial value for this
    soh_k = casadi.SX.sym("soh_0")
    w += [soh_k]
    lbw += [1.0]
    ubw += [1.0]
    w0 += [1.0]
    print(len(residual_fixed_load))
    for k in range(len(residual_fixed_load)):
        battery_charge_power_pos_k = casadi.SX.sym("power_pos_"+str(k))
        w += [battery_charge_power_pos_k]
        lbw += [0]
        ubw += [max_power_charge]
        w0 += [0]
        battery_charge_power_neg_k = casadi.SX.sym("power_neg_" + str(k))
        w += [battery_charge_power_neg_k]
        lbw += [0]
        ubw += [-max_power_discharge]
        w0 += [0]
        grid_power_k = casadi.SX.sym("grid_power_" + str(k))
        w += [grid_power_k]
        lbw += [-math.inf]
        ubw += [math.inf]
        w0 += [residual_fixed_load[k]]
        battery_charge_power_k = battery_charge_power_pos_k - battery_charge_power_neg_k
        # compute intermediate result of soc update formula
        d_soc_pos = DELTA_TIME_HOURS * battery_charge_power_pos_k / energy_capacity
        d_soc_neg = DELTA_TIME_HOURS * battery_charge_power_neg_k / energy_capacity
        soc_end = soc_k + d_soc_pos - d_soc_neg
        soc_k = casadi.SX.sym(f"soc_end{k + 1}")
        w += [soc_k]
        lbw += [min_soc]
        ubw += [max_soc]
        w0 += [0.5]
        d_soh_calendrical = 0.2/max_life_time_hours
        d_soh_cyclical = 0.2 * ((d_soc_pos+d_soc_neg)/2)/max_cycles
        d_soh = d_soh_calendrical + d_soh_cyclical
        soh_end = soh_k - d_soh
        soh_k = casadi.SX.sym(f"soh_end{k+1}")
        w+= [soh_k]
        lbw += [min_soh]
        ubw += [max_soh]
        w0 += [1.0]
        J = J + price * np.log(1 + np.exp(grid_power_k)) + replacement_price * d_soh * 5 # minimize grid power
        g += [soc_end - soc_k, soh_end-soh_k, grid_power_k - (residual_fixed_load[k] + battery_charge_power_k)]
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

    prob = {"f": J, "x": casadi.vertcat(*w), "g": casadi.vertcat(*g)}
    print(J.shape, len(w), len(g), len(w0))
    solver = casadi.nlpsol("solver", "ipopt", prob)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()
    return w_opt


if __name__ == "__main__":
    load, price, generation = read_data()

    load = load[0:8700]
    generation = generation[0:8700]

    residual_fixed_load = load - generation
    time = range(len(residual_fixed_load))

    traj = build_optimization_problem(residual_fixed_load, price, soc=0)
    t = [time[i] * DELTA_TIME_HOURS for i in time]
    soc_traj = traj[0::5]
    soh_traj = traj[1::5]
    battery_charge_power_pos_traj = traj[2::5]
    battery_charge_power_neg_traj = traj[3::5]
    print("Battery powers should be exclusive",sum([abs(battery_charge_power_neg_traj[i]*battery_charge_power_pos_traj[i]) for i in range(len(battery_charge_power_pos_traj))]))
    battery_charge_power_traj = battery_charge_power_pos_traj - battery_charge_power_neg_traj
    grid_power_traj = traj[4::5]
    print(traj.shape)
    baseline_cost = sum(residual_fixed_load[residual_fixed_load > 0] * price) + (1/15*replacement_price)
    augmented_load = np.array([(battery_charge_power_traj[i]) for i in time]) + residual_fixed_load
    cost = sum(grid_power_traj[grid_power_traj > 0]) * price + replacement_price * (1-soh_traj[-1])*5

    print('baseline cost: ' + str(baseline_cost))
    print('cost: ' + str(cost))
    print('savings in %: ' + str(1 - cost / baseline_cost))
    print('battery soh at end: '+str(soh_traj[-1]))
    fig1 = plt.figure()
    ax = plt.subplot()
    Line1 = ax.plot(time, [(residual_fixed_load[i]) for i in time], label='Residual Load')
    # Line2 = ax.plot(time, [price] * 8700, '--', label='Price')
    Line3 = ax.plot(time, [(battery_charge_power_traj[i]) for i in time], label='Battery Charge Power')
    Line4 = ax.plot(time, [(grid_power_traj[i]) for i in time], label='Grid Power')
    Line5 = ax.plot(time, [(soc_traj[i]) * 100 for i in time], label="Battery SOC")
    Line6 = ax.plot(time, [(soh_traj[i])*100 for i in time], label = "Battery SOH")
    plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
    plt.xlabel('Time Step')
    ax.legend()
    ax.grid()
    plt.show()