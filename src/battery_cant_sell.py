from pyoptsparse import SLSQP, Optimization
import numpy as np
from utils import (
    get_solar_field_powers,
    get_grid_prices_mwh,
    get_electric_demand_powers,
    generic_plot,
)
from parameters import PARAMS
from custom_types import PlotData


def objfunc(xdict):
    p_bat = xdict["p_bat"]

    funcs = {}
    funcs["cost"] = np.sum(grid_prices_mwh * (-p_gen + p_bat + p_electric_demand))

    # Don't give profit when p_grid is negative
    # Use KS function to approximate max function
    # funcs["cost"] = np.sum( 1 / PARAMS["DK_RHO"] * np.log(1 + np.exp(PARAMS["DK_RHO"] * (grid_prices_mwh * (-p_gen + p_bat + p_electric_demand)))) )

    stored_battery_energy = []
    for h in range(1, PARAMS["N_HOURS"] + 1):
        stored_battery_energy.append(
            PARAMS["SOC_MIN"] * PARAMS["MAX_BAT_CAPACITY"] + np.sum(p_bat[:h])
        )
    funcs["stored_battery_energy"] = stored_battery_energy

    grid_power = []
    for h in range(PARAMS["N_HOURS"]):
        grid_power.append(-p_gen[h] + p_bat[h] + p_electric_demand[h])
    funcs["grid_power"] = grid_power

    fail = False

    return funcs, fail


def run_optimization():
    # Optimization Object
    optProb = Optimization("All year battery powers", objfunc)

    # Design Variables
    optProb.addVarGroup(
        "p_bat",
        PARAMS["N_HOURS"],
        "c",
        lower=-PARAMS["P_BAT_MAX"],
        upper=PARAMS["P_BAT_MAX"],
        value=0,
    )

    # Constraints
    optProb.addConGroup(
        "stored_battery_energy",
        PARAMS["N_HOURS"],
        lower=(PARAMS["SOC_MIN"] * PARAMS["MAX_BAT_CAPACITY"]),
        upper=(PARAMS["SOC_MAX"] * PARAMS["MAX_BAT_CAPACITY"]),
    )

    optProb.addConGroup(
        "grid_power", PARAMS["N_HOURS"], lower=0, upper=PARAMS["P_GRID_MAX"]
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
    grid_prices_mwh = get_grid_prices_mwh(PARAMS["N_HOURS"])
    p_gen = get_solar_field_powers(PARAMS["MAX_SOLAR_RADIATION"], PARAMS["N_HOURS"])
    p_electric_demand = get_electric_demand_powers(
        PARAMS["MAX_ELECTRIC_DEMAND"], PARAMS["N_HOURS"]
    )
    hours = np.arange(len(p_gen))

    # Run optimization
    sol = run_optimization()

    # Check Solution
    print(sol)

    # Plot results
    plot_data: PlotData = {
        "rows": 2,
        "columns": 1,
        "axes_data": [
            {
                "i": 0,
                "j": 0,
                "ylabel": "Price (€/MWh)",
                "arrays_data": [
                    {
                        "x": hours,
                        "y": grid_prices_mwh,
                        "label": None,
                    }
                ],
            },
            {
                "i": 0,
                "j": 1,
                "xlabel": "Time (hours)",
                "ylabel": "Power (kW)",
                "arrays_data": [
                    {
                        "x": hours,
                        "y": p_gen,
                        "label": "Solar field power (kW)",
                    },
                    {
                        "x": hours,
                        "y": (-p_gen + sol.xStar["p_bat"] + p_electric_demand),
                        "label": "Power from grid (kW)",
                    },
                    {
                        "x": hours,
                        "y": sol.xStar["p_bat"],
                        "label": "Power to battery (kW)",
                    },
                    {
                        "x": hours,
                        "y": p_electric_demand,
                        "label": "Electric power demand (kW)",
                    },
                ],
            },
        ],
    }

    generic_plot(plot_data, sharex=True)
