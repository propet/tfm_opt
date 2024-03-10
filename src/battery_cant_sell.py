import numpy as np
from pyoptsparse import SLSQP, Optimization
from utils import get_solar_field_powers, get_grid_prices_kwh, get_electric_demand_powers, generic_plot
from parameters import PARAMS
from custom_types import PlotData
import matplotlib.pyplot as plt


def objfunc_ks(grid_prices_kwh, p_gen, p_electric_demand):

    def return_function(xdict):
        p_bat = xdict["p_bat"]

        funcs = {}
        # Don't give profit when p_grid violates constraint P_grid > 0
        # Use KS function to approximate max function
        funcs["cost"] = np.sum(
            1
            / PARAMS["DK_RHO"]
            * np.log(1 + np.exp(PARAMS["DK_RHO"] * grid_prices_kwh * (-p_gen + p_bat + p_electric_demand)))
        )

        battery_soc = []
        for h in range(1, PARAMS["N_HOURS"] + 1):
            battery_soc.append(
                (PARAMS["SOC_MIN"] * PARAMS["MAX_BAT_CAPACITY"] + np.sum(p_bat[:h])) / PARAMS["MAX_BAT_CAPACITY"]
            )
        funcs["battery_soc"] = battery_soc

        grid_power = []
        for h in range(PARAMS["N_HOURS"]):
            grid_power.append(-p_gen[h] + p_bat[h] + p_electric_demand[h])
        funcs["grid_power"] = grid_power

        fail = False
        return funcs, fail

    return return_function


def get_input_data():
    grid_prices_kwh = get_grid_prices_kwh(PARAMS["N_HOURS"])
    p_gen = get_solar_field_powers(PARAMS["MAX_SOLAR_RADIATION"], PARAMS["N_HOURS"])
    p_electric_demand = get_electric_demand_powers(PARAMS["MAX_ELECTRIC_DEMAND"], PARAMS["N_HOURS"])
    hours = np.arange(len(p_gen))
    return hours, grid_prices_kwh, p_gen, p_electric_demand


def run_optimization(plot=True):
    (hours, grid_prices_kwh, p_gen, p_electric_demand) = get_input_data()

    # Optimization Object
    optProb = Optimization("All year battery powers", objfunc_ks(grid_prices_kwh, p_gen, p_electric_demand))

    # Design Variables
    optProb.addVarGroup("p_bat", PARAMS["N_HOURS"], "c", lower=-PARAMS["P_BAT_MAX"], upper=PARAMS["P_BAT_MAX"], value=0)

    # Battery SOC constraint
    optProb.addConGroup("battery_soc", PARAMS["N_HOURS"], lower=PARAMS["SOC_MIN"], upper=PARAMS["SOC_MAX"])

    # Grid power constraint
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

        battery_soc = []
        for h in range(1, PARAMS["N_HOURS"] + 1):
            battery_soc.append(
                (PARAMS["SOC_MIN"] * PARAMS["MAX_BAT_CAPACITY"] + np.sum(sol.xStar["p_bat"][:h]))
                / PARAMS["MAX_BAT_CAPACITY"]
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
                            "y": (-p_gen + sol.xStar["p_bat"] + p_electric_demand),
                            "label": "From grid",
                        },
                        {
                            "x": hours,
                            "y": sol.xStar["p_bat"],
                            "label": "To battery",
                        },
                        {
                            "x": hours,
                            "y": p_electric_demand,
                            "label": "To demand",
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

    # #######################################################
    # # Iterate through battery capacities
    # #######################################################
    # cases = np.array(list(range(1, 120)))
    # obj_values = np.zeros_like(cases)
    # i = 0
    # for q in cases:
    #     print(f"\rProgress: {q}/{len(cases) - 1}", end='')
    #     PARAMS["MAX_BAT_CAPACITY"] = q
    #     sol = run_optimization(plot=False)
    #     obj_values[i] = sol.fStar
    #     i += 1
    #
    # plot_data: PlotData = {
    #     "rows": 1,
    #     "columns": 1,
    #     "axes_data": [
    #         {
    #             "i": 0,
    #             "j": 0,
    #             "ylabel": r"Cost[\$]",
    #             "xlabel": "Battery Capacity[kWh]",
    #             "arrays_data": [
    #                 {
    #                     "x": cases,
    #                     "y": obj_values,
    #                     "label": None,
    #                 }
    #             ],
    #         },
    #     ],
    # }
    # generic_plot(plot_data, sharex=True)

    #######################################################
    # Iterate through time horizon and battery capacities
    #######################################################
    # hours = np.array(list(range(30, 80, 10)))
    # capacities = np.linspace(20, 200, 20)
    # hours, capacities = np.meshgrid(hours, capacities)
    # cost_per_hour = np.zeros_like(hours, dtype=float)
    # for i in range(hours.shape[0]):
    #     for j in range(hours.shape[1]):
    #         print(f"\rProgress: {i*hours.shape[1] + j}/{hours.size - 1}", end='')
    #         PARAMS["N_HOURS"] = hours[i][j]
    #         PARAMS["MAX_BAT_CAPACITY"] = capacities[i][j]
    #         sol = run_optimization(plot=False)
    #         cost_per_hour[i][j] = sol.fStar / hours[i][j]
    #
    # # Create the contour plot
    # plt.figure()
    # CP = plt.contour(hours, capacities, cost_per_hour)
    #
    # # Adding labels to contours
    # plt.clabel(CP, inline=True, fontsize=10)
    #
    # plt.xlabel('hours')
    # plt.ylabel('kWh')
    # plt.show()
