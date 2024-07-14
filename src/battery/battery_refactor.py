import numpy as np
from pyoptsparse import OPT
from opt import Opt
from utils import get_solar_field_powers, get_grid_prices_kwh, generic_plot, ks_max, ks_min
from parameters import PARAMS
from typing import List, Dict
from custom_types import DesignVariables, PlotData, DesignVariableInfo, ConstraintInfo, Parameters


def get_input_data():
    grid_prices_kwh = get_grid_prices_kwh(PARAMS["N_HOURS"])
    p_gen = get_solar_field_powers(PARAMS["MAX_SOLAR_RADIATION"], PARAMS["N_HOURS"])
    hours = np.arange(len(p_gen))
    return hours, grid_prices_kwh, p_gen


def obj(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    p_bat = design_variables["p_bat"]
    grid_prices_kwh = parameters["grid_prices_kwh"]
    p_gen = parameters["p_gen"]
    return np.sum(grid_prices_kwh * (-p_gen + p_bat))


def battery_soc_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    p_bat = design_variables["p_bat"]
    battery_soc = []

    for h in range(1, parameters["N_HOURS"] + 1):
        battery_soc.append(
            (parameters["SOC_MIN"] * parameters["MAX_BAT_CAPACITY"] + np.sum(p_bat[:h]))
            / parameters["MAX_BAT_CAPACITY"]
        )

    return np.array(battery_soc)


def battery_max_soc_ks_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    p_bat = design_variables["p_bat"]
    max_bat_capacity = parameters["MAX_BAT_CAPACITY"]
    soc_min = parameters["SOC_MIN"]
    battery_soc = []

    for h in range(1, parameters["N_HOURS"] + 1):
        battery_soc.append((soc_min * max_bat_capacity + np.sum(p_bat[:h])) / max_bat_capacity)

    battery_max_soc = ks_max(np.array(battery_soc))
    return np.array(battery_max_soc)


def battery_min_soc_ks_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    p_bat = design_variables["p_bat"]
    max_bat_capacity = parameters["MAX_BAT_CAPACITY"]
    soc_min = parameters["SOC_MIN"]
    battery_soc = []

    for h in range(1, parameters["N_HOURS"] + 1):
        battery_soc.append((soc_min * max_bat_capacity + np.sum(p_bat[:h])) / max_bat_capacity)

    battery_min_soc = ks_min(np.array(battery_soc))
    return np.array(battery_min_soc)


def grid_power_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    p_bat = design_variables["p_bat"]
    p_gen = parameters["p_gen"]
    grid_power = []

    for h in range(parameters["N_HOURS"]):
        grid_power.append(-p_gen[h] + p_bat[h])

    return np.array(grid_power)


def grid_max_power_ks_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    p_bat = design_variables["p_bat"]
    p_gen = parameters["p_gen"]
    grid_power = []

    for h in range(parameters["N_HOURS"]):
        grid_power.append(-p_gen[h] + p_bat[h])

    # Differentiable max(grid_power)
    max_power = ks_max(np.array(grid_power))
    return np.array(max_power)


def grid_min_power_ks_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    p_bat = design_variables["p_bat"]
    p_gen = parameters["p_gen"]
    grid_power = []

    for h in range(parameters["N_HOURS"]):
        grid_power.append(-p_gen[h] + p_bat[h])

    # Differentiable min(grid_power)
    min_power = ks_min(np.array(grid_power))
    return np.array(min_power)


def run_optimization(plot=True):
    opt = Opt("refactor_name", obj)

    # Parameters
    hours, grid_prices_kwh, p_gen = get_input_data()
    PARAMS["hours"] = hours
    PARAMS["grid_prices_kwh"] = grid_prices_kwh
    PARAMS["p_gen"] = p_gen
    opt.add_parameters(PARAMS)

    # Design Variables
    p_bat: DesignVariableInfo = {
        "name": "p_bat",
        "n_params": PARAMS["N_HOURS"],
        "type": "c",
        "lower": -PARAMS["P_BAT_MAX"],
        "upper": PARAMS["P_BAT_MAX"],
        "initial_value": 0,
        "scale": 1,
    }
    opt.add_design_variables_info(p_bat)

    # Constraints
    battery_soc_constraint: ConstraintInfo = {
        "name": "battery_soc",
        "n_constraints": PARAMS["N_HOURS"],
        "lower": PARAMS["SOC_MIN"],
        "upper": PARAMS["SOC_MAX"],
        "function": battery_soc_constraint_fun,
        "scale": 1,
    }
    opt.add_constraint_info(battery_soc_constraint)

    grid_power_constraint: ConstraintInfo = {
        "name": "grid_power",
        "n_constraints": PARAMS["N_HOURS"],
        "lower": -PARAMS["P_GRID_MAX"],
        "upper": PARAMS["P_GRID_MAX"],
        "function": grid_power_constraint_fun,
        "scale": 1,
    }
    opt.add_constraint_info(grid_power_constraint)

    opt.setup()

    # Check optimization problem
    if plot:
        opt.print()

    # Solve
    slsqpoptOptions = {"IPRINT": -1}
    ipoptOptions = {
        "print_level": 5,
        "tol": 1e-2,
        # "mu_strategy": "adaptive",
        # "alpha_red_factor": 0.2
    }
    optOptions = ipoptOptions
    optimizer = OPT("ipopt", options=optOptions)
    sol = optimizer(opt.optProb, sens="FD", sensStep=1e-6)

    # Check Solution
    if plot:
        # print(sol)

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

        filename = (
            "_".join(
                f"{key}_{value}"
                for key, value in PARAMS.items()
                if key
                in [
                    "N_HOURS",
                    "MAX_BAT_CAPACITY",
                    "SOC_MIN",
                    "SOC_MAX",
                    "P_BAT_MAX",
                    "P_GRID_MAX",
                    "MAX_SOLAR_RADIATION",
                    "MAX_ELECTRIC_DEMAND",
                    "DK_RHO",
                    "BAT_ETA_CHARGE",
                    "BAT_ETA_DISCHARGE",
                ]
            )
            + ".png"
        )
        generic_plot(plot_data, filename=filename, sharex=True)

    return sol


if __name__ == "__main__":
    run_optimization(plot=True)
