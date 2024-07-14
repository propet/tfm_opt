from pyoptsparse import SLSQP, Optimization
import numpy as np
from utils import get_solar_field_powers, get_grid_prices_kwh, generic_plot, sigmoid
from parameters import PARAMS
from custom_types import PlotData

BATTERY_CAPACITY_UPPER_BOUND = 1000


def objfunc(grid_prices_kwh, p_gen):

    def return_function(xdict):
        p_bat = xdict["p_bat"]
        battery_capacity = xdict["battery_capacity"]

        funcs = {}
        funcs["cost"] = np.sum(grid_prices_kwh * (-p_gen + p_bat))

        # Battery SOC constraints
        sigmoid_p = sigmoid(p_bat)
        eta = sigmoid_p * PARAMS["BAT_ETA_CHARGE"] + (1 - sigmoid_p) * PARAMS["BAT_ETA_DISCHARGE"]
        battery_soc = []
        for h in range(1, PARAMS["N_HOURS"] + 1):
            battery_soc.append((PARAMS["SOC_MIN"] * battery_capacity + np.sum(eta[:h] * p_bat[:h])) / battery_capacity)
        funcs["battery_soc"] = battery_soc

        # Grid power constraints
        grid_power = []
        for h in range(PARAMS["N_HOURS"]):
            grid_power.append(-p_gen[h] + p_bat[h])
        funcs["grid_power"] = grid_power

        fail = False
        return funcs, fail

    return return_function


def get_input_data():
    grid_prices_kwh = get_grid_prices_kwh(PARAMS["N_HOURS"])
    p_gen = get_solar_field_powers(PARAMS["MAX_SOLAR_RADIATION"], PARAMS["N_HOURS"])
    hours = np.arange(len(p_gen))
    return hours, grid_prices_kwh, p_gen


def run_optimization(plot=True):
    hours, grid_prices_kwh, p_gen = get_input_data()

    # Optimization Object
    optProb = Optimization("All year battery powers", objfunc(grid_prices_kwh, p_gen))

    # Design Variables
    optProb.addVarGroup("p_bat", PARAMS["N_HOURS"], "c", lower=-PARAMS["P_BAT_MAX"], upper=PARAMS["P_BAT_MAX"], value=0)
    optProb.addVar("battery_capacity", "c", lower=1e-3, upper=BATTERY_CAPACITY_UPPER_BOUND, value=0)

    # Battery SOC constraints
    optProb.addConGroup("battery_soc", PARAMS["N_HOURS"], lower=PARAMS["SOC_MIN"], upper=PARAMS["SOC_MAX"])

    # Grid power constraints
    optProb.addConGroup("grid_power", PARAMS["N_HOURS"], lower=-PARAMS["P_GRID_MAX"], upper=PARAMS["P_GRID_MAX"])

    # Objective
    optProb.addObj("cost")

    # Check optimization problem
    if plot:
        print(optProb)

    # Optimizer
    optOptions = {"IPRINT": -1}
    opt = SLSQP(options=optOptions)

    # Solve
    sol = opt(optProb, sens="FD", sensStep=1e-6)

    # Check Solution
    if plot:
        print(sol)

        sigmoid_p = sigmoid(sol.xStar["p_bat"])
        eta = sigmoid_p * PARAMS["BAT_ETA_CHARGE"] + (1 - sigmoid_p) * PARAMS["BAT_ETA_DISCHARGE"]
        battery_soc = []
        for h in range(1, PARAMS["N_HOURS"] + 1):
            battery_soc.append(
                (PARAMS["SOC_MIN"] * sol.xStar["battery_capacity"] + np.sum(eta[:h] * sol.xStar["p_bat"][:h]))
                / sol.xStar["battery_capacity"]
            )

        plot_data: PlotData = {
            "rows": 3,
            "columns": 1,
            "axes_data": [
                {
                    "i": 0,
                    "j": 0,
                    "ylabel": "Price[€/kWh]",
                    "arrays_data": [
                        {
                            "x": hours,
                            "y": grid_prices_kwh,
                            "label": None,
                        }
                    ],
                },
                {
                    "i": 0,
                    "j": 1,
                    "ylabel": "Power[kW]",
                    "arrays_data": [
                        {
                            "x": hours,
                            "y": p_gen,
                            "label": "From Solar",
                        },
                        {
                            "x": hours,
                            "y": (-p_gen + sol.xStar["p_bat"]),
                            "label": "From grid",
                        },
                        {
                            "x": hours,
                            "y": sol.xStar["p_bat"],
                            "label": "To battery",
                        },
                    ],
                },
                {
                    "i": 0,
                    "j": 2,
                    "xlabel": "Time[hours]",
                    "ylabel": "Battery SOC",
                    "arrays_data": [
                        {
                            "x": hours,
                            "y": battery_soc,
                            "label": None,
                        },
                    ],
                },
            ],
        }
        generic_plot(plot_data, sharex=True)

    return sol


if __name__ == "__main__":
    run_optimization(plot=True)
