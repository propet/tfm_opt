from pyoptsparse import SLSQP, Optimization
import numpy as np
import matplotlib.pyplot as plt
from utils import get_solar_field_powers, get_grid_prices_mwh

# Scienceplots style
import scienceplots
plt.style.use(["science", "ieee"])
plt.rcParams.update({'figure.dpi': '300'})


#############
# PARAMETERS
#############
N_HOURS = 100  # 8760 in a year
MAX_BAT_CAPACITY = 100  # kWh
SOC_MIN = 0.1
SOC_MAX = 0.9
P_BAT_MAX = 5  # kW
P_GRID_MAX = 10  # kW
MAX_SOLAR_RADIATION = 10  # kW


def plot_data():
    plt.figure(figsize=(10, 5))
    plt.plot(hours, p_gen, label="Solar field power (kW)")
    plt.plot(hours, grid_prices_mwh, label="Grid Prices (€/MWh)")
    plt.xlabel("Time (hours)")
    plt.grid(True)
    plt.legend()

    # Adding vertical lines every 24 x-values
    for i in range(24, max(hours), 24):
        plt.axvline(x=i, color="r", linestyle="--", label="Day boundary" if i == 24 else "")

    plt.show()


def plot_results(sol):
    p_bat = sol.xStar["p_bat"]
    plt.figure(figsize=(10, 5))
    plt.plot(hours, grid_prices_mwh, label="Grid Prices (€/MWh)")
    plt.plot(hours, p_gen, label="Solar field power (kW)")
    plt.plot(hours, (-p_gen + p_bat), label="Power from grid (kW)")
    plt.plot(hours, p_bat, label="Power to battery (kW)")
    plt.xlabel("Time (hours)")
    plt.grid(True)
    plt.legend()

    # Adding vertical lines every 24 x-values
    for i in range(24, max(hours), 24):
        plt.axvline(x=i, color="r", linestyle="--", label="Day boundary" if i == 24 else "")

    plt.show()


def objfunc(xdict):
    p_bat = xdict["p_bat"]

    funcs = {}
    funcs["cost"] = np.sum(grid_prices_mwh * (-p_gen + p_bat))

    stored_battery_energy = []
    for h in range(1, N_HOURS+1):
        stored_battery_energy.append(SOC_MIN * MAX_BAT_CAPACITY + np.sum(p_bat[:h]))
    funcs["stored_battery_energy"] = stored_battery_energy

    grid_power = []
    for h in range(N_HOURS):
        grid_power.append(-p_gen[h] + p_bat[h])
    funcs["grid_power"] = grid_power

    fail = False

    return funcs, fail


def run_optimization():
    # Optimization Object
    optProb = Optimization("All year battery powers", objfunc)

    # Design Variables
    # kW
    optProb.addVarGroup(
        "p_bat",
        N_HOURS,
        "c",
        lower=-P_BAT_MAX,
        upper=P_BAT_MAX,
        value=0
    )

    # Constraints
    optProb.addConGroup(
        "stored_battery_energy",
        N_HOURS,
        lower=(SOC_MIN * MAX_BAT_CAPACITY),
        upper=(SOC_MAX * MAX_BAT_CAPACITY)
    )

    optProb.addConGroup(
        "grid_power",
        N_HOURS,
        lower=-P_GRID_MAX,
        upper=P_GRID_MAX
    )

    # Objective
    optProb.addObj("cost")

    # Check optimization problem
    # print(optProb)

    # Optimizer
    optOptions = {"IPRINT": -1}
    opt = SLSQP(options=optOptions)

    # Solve
    sol = opt(optProb, sens="FD")
    return sol


if __name__ == "__main__":
    # Retrieve data
    grid_prices_mwh = get_grid_prices_mwh(N_HOURS)
    p_gen = get_solar_field_powers(MAX_SOLAR_RADIATION, N_HOURS)
    hours = np.arange(len(p_gen))
    plot_data()

    # Run optimization
    sol = run_optimization()

    # Check Solution
    print(sol)
    plot_results(sol)
