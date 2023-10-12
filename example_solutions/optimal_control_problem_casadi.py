import casadi
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


def build_optimization_problem(residual_fixed_load, price, soc=0):
    # model parameter initilization
    soc_init = soc

    max_power_charge = 50
    max_power_discharge = -50
    max_soc = 100
    min_soc = 0
    energy_capacity = 200



    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    sock = casadi.SX.sym("soc0")
    w += [sock]
    lbw += [soc_init]
    ubw += [soc_init]
    w0 += [soc_init]

    for k in range(len(residual_fixed_load)):
        powerk = casadi.SX.sym("power_" + str(k))

        w += [powerk]
        lbw += [max_power_discharge]
        ubw += [max_power_charge]
        w0 += [0]

        soc_end = float(100) * DELTA_TIME_HOURS * powerk / energy_capacity + sock
        J = J + price[k]*np.log(1 + np.exp(powerk + residual_fixed_load[k]))
        sock = casadi.SX.sym(f"soc_end{k+1}")

        w += [sock]
        lbw += [min_soc]
        ubw += [max_soc]
        w0 += [0.5]

        g += [soc_end - sock]
        lbg += [0]
        ubg += [0]

    prob = {"f": J, "x": casadi.vertcat(*w), "g": casadi.vertcat(*g)}
    solver = casadi.nlpsol("solver", "ipopt", prob)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()
    return w_opt





if __name__ == "__main__":


    load, price, generation = read_data()

    load = load[0:8700]
    price = price[0:8700]
    generation = generation[0:8700]

    residual_fixed_load = load - generation
    time = range(len(residual_fixed_load))

    traj = build_optimization_problem(residual_fixed_load, price, soc=0)
    t = [time[i] * DELTA_TIME_HOURS for i in time]
    power_traj = traj[1::2]
    baseline_cost = sum(residual_fixed_load[residual_fixed_load > 0] * price[residual_fixed_load > 0])
    augmented_load = residual_fixed_load + np.array([(power_traj[i]) for i in time])
    cost = sum(augmented_load[augmented_load > 0] * price[augmented_load > 0])

    print('baseline cost: ' + str(baseline_cost))
    print('cost: ' + str(cost))
    print('savings in %: ' + str(cost/baseline_cost))

    fig1 = plt.figure()
    ax = plt.subplot()
    Line1 = ax.plot(time, [(residual_fixed_load[i]) for i in time], label='Residual Load')
    Line2 = ax.plot(time, augmented_load, label='Augmented Load')
    Line3 = ax.plot(time, price, '--', label='Price')
    Line4 = ax.plot(time, [(power_traj[i]) for i in time], label='Battery Power')
    plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
    plt.xlabel('Time Step')
    ax.legend()
    ax.grid()
    plt.show()
