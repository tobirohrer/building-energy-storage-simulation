import math

import casadi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DELTA_TIME_HOURS = 1
HOURS_PER_YEAR = 24*365
energy_capacity = 200  # kWh
replacement_price_battA = energy_capacity * 50 #kWh*€/kWh
replacement_price_battB = energy_capacity * 1500  # kWh*€/kWh


def read_data():
    load = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/electricity_load_profile.csv')[
        'Load [kWh]']
    price = 0.4
    generation = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/solar_generation_profile.csv')[
        'Generation [kWh]']
    return np.array(load), np.array(price), np.array(generation)


def build_optimization_problem(residual_fixed_load, price, soc=0):
    # model parameter initialization
    soc_init = 0.5

    max_power_charge = 50  # kW
    max_power_discharge = -1 * max_power_charge
    max_soc = 1
    min_soc = 0
    min_soh = 0.8
    max_soh = 1
    max_life_time = 15 #years
    max_life_time_hours = max_life_time*HOURS_PER_YEAR
    max_cycles = 10000
    w = []  # list of control variables
    w0 = []  # initial guess for state variables
    lbw = []  # lower bound for state variables
    ubw = []  # upper bound for state variables
    J = 0  # initial value for cost function
    g = []  # list of constraints
    lbg = []  # lower bound for constraints
    ubg = []  # upper bound for constraints
    soc_a_k = casadi.SX.sym("soc_a_0")  # define first state variable as casadi symbol
    w += [soc_a_k]  # add first control variable into state vector
    lbw += [soc_init]  # add state bounds, in this case start with initial value
    ubw += [soc_init]  # also the upper one, which is also the same as the lower bound since this is an equality constraint
    w0 += [soc_init]  # use initial value for this
    soh_a_k = casadi.SX.sym("soh_a_0")
    w += [soh_a_k]
    lbw += [1.0]
    ubw += [1.0]
    w0 += [1.0]
    soc_b_k = casadi.SX.sym("soc_b_0")  # define first state variable as casadi symbol
    w += [soc_b_k]  # add first control variable into state vector
    lbw += [soc_init]  # add state bounds, in this case start with initial value
    ubw += [
        soc_init]  # also the upper one, which is also the same as the lower bound since this is an equality constraint
    w0 += [soc_init]  # use initial value for this
    soh_b_k = casadi.SX.sym("soh_b_0")
    w += [soh_b_k]
    lbw += [1.0]
    ubw += [1.0]
    w0 += [1.0]
    large_number = 1e2
    print(len(residual_fixed_load))
    for k in range(len(residual_fixed_load)):
        battery_a_charge_power_k = casadi.SX.sym("power_a_" + str(k))  # casadi symbol for control variable
        w += [battery_a_charge_power_k]
        lbw += [max_power_discharge]
        ubw += [max_power_charge]
        w0 += [0]  # use 0 power as initial guess
        battery_b_charge_power_k = casadi.SX.sym("power_b_" + str(k))  # casadi symbol for control variable
        w += [battery_b_charge_power_k]
        lbw += [max_power_discharge]
        ubw += [max_power_charge]
        w0 += [0]  # use 0 power as initial guess
        grid_power_k = casadi.SX.sym("grid_power_" + str(k))
        w += [grid_power_k]
        lbw += [-math.inf]
        ubw += [math.inf]
        w0 += [residual_fixed_load[k]]
        # compute intermediate result of soc update formula
        d_soc_a = DELTA_TIME_HOURS * battery_a_charge_power_k / energy_capacity
        soc_a_end = soc_a_k + d_soc_a
        soc_a_k = casadi.SX.sym(f"soc_a_end{k + 1}")
        w += [soc_a_k]
        lbw += [min_soc]
        ubw += [max_soc]
        w0 += [0.5]
        d_soh_a_calendrical = 0.2/max_life_time_hours
        d_soc_a_charging = np.log(1 + np.exp(d_soc_a*large_number))/large_number #max(0,d_soc)
        d_soc_a_discharging = np.log(1 + np.exp(-d_soc_a*large_number))/large_number # max(0,-d_soc)
        d_soh_a_cyclical_charging = 0.2 * (d_soc_a_charging/2)/max_cycles
        d_soh_a_cyclical_discharging = 0.2 * (d_soc_a_discharging/2)/max_cycles
        d_soh_a = d_soh_a_calendrical + d_soh_a_cyclical_charging + d_soh_a_cyclical_discharging
        soh_a_end = soh_a_k - d_soh_a
        soh_a_k = casadi.SX.sym(f"soh_a_end{k+1}")
        w += [soh_a_k]
        lbw += [min_soh]
        ubw += [max_soh]
        w0 += [1.0]

        d_soc_b = DELTA_TIME_HOURS * battery_b_charge_power_k / energy_capacity
        soc_b_end = soc_b_k + d_soc_b
        soc_b_k = casadi.SX.sym(f"soc_b_end{k + 1}")
        w += [soc_b_k]
        lbw += [min_soc]
        ubw += [max_soc]
        w0 += [0.5]
        d_soh_b_calendrical = 0.2 / max_life_time_hours
        d_soc_b_charging = np.log(1 + np.exp(d_soc_b*large_number))/large_number
        d_soc_b_discharging = np.log(1 + np.exp(-d_soc_b*large_number))/large_number # ln(1+e^x) ~ max(0,x)
        d_soh_b_cyclical_charging = 0.2 * (d_soc_b_charging / 2) / max_cycles
        d_soh_b_cyclical_discharging = 0.2 * (d_soc_b_discharging / 2) / max_cycles
        d_soh_b = d_soh_b_calendrical + d_soh_b_cyclical_charging + d_soh_b_cyclical_discharging
        soh_b_end = soh_b_k - d_soh_b
        soh_b_k = casadi.SX.sym(f"soh_b_end{k + 1}")
        w += [soh_b_k]
        lbw += [min_soh]
        ubw += [max_soh]
        w0 += [1.0]

        J = J + price * np.log(1 + np.exp(grid_power_k)) \
            + replacement_price_battA * d_soh_a * 5 \
            + replacement_price_battB * d_soh_b * 5
        g += [soc_a_end - soc_a_k,
              soh_a_end - soh_a_k,
              soc_b_end - soc_b_k,
              soh_b_end - soh_b_k,
              grid_power_k - (residual_fixed_load[k] + battery_a_charge_power_k + battery_b_charge_power_k)]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

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

    traj = build_optimization_problem(residual_fixed_load, price, soc=0.5)
    t = [time[i] * DELTA_TIME_HOURS for i in time]
    soc_a_traj = traj[0::7]
    soh_a_traj = traj[1::7]
    soc_b_traj = traj[2::7]
    soh_b_traj = traj[3::7]
    battery_a_charge_power_traj = traj[4::7]
    battery_b_charge_power_traj = traj[5::7]
    print(np.mean(np.abs(battery_a_charge_power_traj)),np.mean(np.abs(battery_b_charge_power_traj)))
    grid_power_traj = traj[6::7]
    print(traj.shape)
    baseline_cost = sum(residual_fixed_load[residual_fixed_load > 0] * price) + (1/15*(replacement_price_battA+replacement_price_battB))
    cost = sum(grid_power_traj[grid_power_traj > 0]) * price \
           + replacement_price_battA * (1 - soh_a_traj[-1]) * 5 \
           + replacement_price_battB * (1 - soh_b_traj[-1]) * 5

    print('baseline cost: ' + str(baseline_cost))
    print('cost: ' + str(cost))
    print('savings in %: ' + str(1 - cost / baseline_cost))
    print('battery A soh at end: ' + str(soh_a_traj[-1]))
    print('battery B soh at end: ' + str(soh_b_traj[-1]))
    fig1 = plt.figure()
    ax = plt.subplot()
    Line1 = ax.plot(time, [(residual_fixed_load[i]) for i in time], label='Residual Load')
    Line2 = ax.plot(time, [(battery_a_charge_power_traj[i]) for i in time], label='Battery A Charge Power')
    Line3 = ax.plot(time, [(battery_b_charge_power_traj[i]) for i in time], "--", label='Battery B Charge Power')
    Line4 = ax.plot(time, [(grid_power_traj[i]) for i in time], label='Grid Power')
    Line5 = ax.plot(time, [(soc_a_traj[i]) * 100 for i in time], label="Battery A SOC")
    Line6 = ax.plot(time, [(soh_a_traj[i]) * 100 for i in time], label ="Battery A SOH")

    Line7 = ax.plot(time, [(soc_b_traj[i]) * 100 for i in time], "--", label="Battery B SOC")
    Line8 = ax.plot(time, [(soh_b_traj[i]) * 100 for i in time], "--", label ="Battery B SOH")
    plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
    plt.xlabel('Time Step')
    ax.legend()
    ax.grid()
    plt.show()