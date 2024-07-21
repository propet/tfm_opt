import numpy as np
from pyoptsparse import OPT
from opt import Opt
from utils import get_solar_field_powers, get_grid_prices_kwh, generic_plot
from parameters import PARAMS
from typing import List, Dict
from custom_types import DesignVariables, PlotData, DesignVariableInfo, ConstraintInfo, Parameters
import scipy.sparse as sp


def get_input_data():
    grid_prices_kwh = get_grid_prices_kwh(PARAMS["N_HOURS"])
    p_gen = get_solar_field_powers(PARAMS["MAX_SOLAR_RADIATION"], PARAMS["N_HOURS"])
    hours = np.arange(len(p_gen))
    return hours, grid_prices_kwh, p_gen


# def obj(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
def obj(opt, design_variables: DesignVariables) -> np.ndarray:
    parameters = opt.parameters

    p_bat = design_variables["p_bat"]
    grid_prices_kwh = parameters["grid_prices_kwh"]
    p_gen = parameters["p_gen"]
    # print(p_bat[:10])
    # print(np.mean(grid_prices_kwh * p_bat))
    return np.sum((grid_prices_kwh * (-p_gen + p_bat)) / 1e3)
    # return np.sum((grid_prices_kwh * (-p_gen + p_bat)) * 1e2)
    # return np.sum(grid_prices_kwh * (-p_gen + p_bat))


def battery_soc_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    parameters = opt.parameters

    # Inequality constraint
    # E_{bat_h} / E_{bat_{max}
    e_bat = design_variables["e_bat"]
    battery_soc = []

    for h in range(parameters["N_HOURS"]):
        battery_soc.append(e_bat[h] / parameters["MAX_BAT_CAPACITY"])

    return np.array(battery_soc)


def battery_energy_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    parameters = opt.parameters

    # Equality constraint
    # E_h - E_{h-1} - P_h = 0
    p_bat = design_variables["p_bat"]
    e_bat = design_variables["e_bat"]
    energy_constraints = []

    initial_energy = parameters["SOC_MIN"] * parameters["MAX_BAT_CAPACITY"]
    energy_constraints.append(e_bat[0] - initial_energy)

    for h in range(1, parameters["N_HOURS"]):
        constraint_value = e_bat[h] - e_bat[h - 1] - p_bat[h]
        energy_constraints.append(constraint_value)

    return np.array(energy_constraints)


def grid_power_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    parameters = opt.parameters

    # Inequality constraint
    # -P_{gen_h} + P_{bat_h}
    p_bat = design_variables["p_bat"]
    p_gen = parameters["p_gen"]
    grid_power = []

    for h in range(parameters["N_HOURS"]):
        grid_power.append(-p_gen[h] + p_bat[h])

    return np.array(grid_power)


def get_constraint_sparse_jacs():
    N_HOURS = PARAMS["N_HOURS"]
    MAX_BAT_CAPACITY = PARAMS["MAX_BAT_CAPACITY"]

    # soc constraints
    dsoc_de = sp.eye(N_HOURS, format="csr") / MAX_BAT_CAPACITY

    # grid power constraints
    dgrid_dp = sp.eye(N_HOURS, format="csr")

    # battery energy constraints
    denergy_dp = sp.eye(N_HOURS, format="csr") * -1
    denergy_dp[0, 0] = 0

    # Create denergy_de
    denergy_de_data = np.ones(2 * N_HOURS - 1)
    denergy_de_data[1::2] = -1
    denergy_de_row = np.repeat(np.arange(N_HOURS), 2)[1:]
    denergy_de_col = np.arange(N_HOURS).repeat(2)[:-1]
    denergy_de = sp.csr_matrix((denergy_de_data, (denergy_de_row, denergy_de_col)), shape=(N_HOURS, N_HOURS))
    # print(denergy_de.toarray())
    # exit(0)

    # Convert to required format
    def to_required_format(mat):
        return {"csr": [mat.indptr, mat.indices, mat.data], "shape": list(mat.shape)}

    return (
        to_required_format(dsoc_de),
        to_required_format(dgrid_dp),
        to_required_format(denergy_dp),
        to_required_format(denergy_de),
    )


def get_constraint_jacs():
    # soc constraints
    # E_{bat_h} / E_{bat_{max}
    dsoc_de = np.zeros((PARAMS["N_HOURS"], PARAMS["N_HOURS"]))
    for i in range(PARAMS["N_HOURS"]):
        dsoc_de[i, i] = 1 / PARAMS["MAX_BAT_CAPACITY"]

    # grid power constraints
    # -P_{gen_h} + P_{bat_h}
    dgrid_dp = np.zeros((PARAMS["N_HOURS"], PARAMS["N_HOURS"]))
    for i in range(PARAMS["N_HOURS"]):
        dgrid_dp[i, i] = 1

    # battery energy constraints
    # E_0 - (SOC_MIN * MAX_BAT_CAPACITY) = 0
    # E_h - E_{h-1} - P_h = 0
    denergy_dp = np.zeros((PARAMS["N_HOURS"], PARAMS["N_HOURS"]))
    denergy_de = np.zeros((PARAMS["N_HOURS"], PARAMS["N_HOURS"]))
    denergy_dp[0, 0] = 0
    denergy_de[0, 0] = 1
    for i in range(1, PARAMS["N_HOURS"]):
        denergy_dp[i, i] = -1
        denergy_de[i, i] = 1
        denergy_de[i, i - 1] = -1

    return dsoc_de, dgrid_dp, denergy_dp, denergy_de


def sens(opt, design_variables: DesignVariables, func_values):
    grid_prices_kwh = PARAMS["grid_prices_kwh"]
    # dsoc_de, dgrid_dp, denergy_dp, denergy_de = get_constraint_jacs()
    dsoc_de, dgrid_dp, denergy_dp, denergy_de = get_constraint_sparse_jacs()

    return {
        "obj": {
            "p_bat": grid_prices_kwh / 1e3,
        },
        "battery_soc": {
            "e_bat": dsoc_de,
        },
        "grid_power": {
            "p_bat": dgrid_dp,
        },
        "battery_energy": {
            "p_bat": denergy_dp,
            "e_bat": denergy_de,
        },
    }


def run_optimization(plot=True):
    opt = Opt("refactor_name", obj)

    # Parameters
    hours, grid_prices_kwh, p_gen = get_input_data()
    PARAMS["hours"] = hours
    PARAMS["grid_prices_kwh"] = grid_prices_kwh
    PARAMS["p_gen"] = p_gen
    opt.add_parameters(PARAMS)

    dsoc_de, dgrid_dp, denergy_dp, denergy_de = get_constraint_sparse_jacs()

    # Design Variables
    p_bat: DesignVariableInfo = {
        "name": "p_bat",
        "n_params": PARAMS["N_HOURS"],
        "type": "c",
        "lower": -PARAMS["P_BAT_MAX"],
        "upper": PARAMS["P_BAT_MAX"],
        "initial_value": 0,
        "scale": 1 / PARAMS["P_BAT_MAX"],
    }
    opt.add_design_variables_info(p_bat)

    e_bat: DesignVariableInfo = {
        "name": "e_bat",
        "n_params": PARAMS["N_HOURS"],
        "type": "c",
        "lower": -PARAMS["MAX_BAT_CAPACITY"] * PARAMS["SOC_MIN"],
        "upper": PARAMS["MAX_BAT_CAPACITY"] * PARAMS["SOC_MAX"],
        "initial_value": PARAMS["MAX_BAT_CAPACITY"] / 2,
        "scale": 1 / PARAMS["MAX_BAT_CAPACITY"],
    }
    opt.add_design_variables_info(e_bat)

    # Constraints
    battery_energy_constraint: ConstraintInfo = {
        "name": "battery_energy",
        "n_constraints": PARAMS["N_HOURS"],
        "lower": 0,
        "upper": 0,
        "function": battery_energy_constraint_fun,
        "scale": 1 / PARAMS["MAX_BAT_CAPACITY"],
        "wrt": ["e_bat", "p_bat"],
        "jac": {"e_bat": denergy_de, "p_bat": denergy_dp},
    }
    opt.add_constraint_info(battery_energy_constraint)

    battery_soc_constraint: ConstraintInfo = {
        "name": "battery_soc",
        "n_constraints": PARAMS["N_HOURS"],
        "lower": PARAMS["SOC_MIN"],
        "upper": PARAMS["SOC_MAX"],
        "function": battery_soc_constraint_fun,
        "scale": 1,
        "wrt": ["e_bat"],
        "jac": {"e_bat": dsoc_de},
    }
    opt.add_constraint_info(battery_soc_constraint)

    grid_power_constraint: ConstraintInfo = {
        "name": "grid_power",
        "n_constraints": PARAMS["N_HOURS"],
        "lower": -PARAMS["P_GRID_MAX"],
        "upper": PARAMS["P_GRID_MAX"],
        "function": grid_power_constraint_fun,
        "scale": 1 / PARAMS["P_GRID_MAX"],
        "wrt": ["p_bat"],
        "jac": {"p_bat": dgrid_dp},
    }
    opt.add_constraint_info(grid_power_constraint)

    # Optimizer
    slsqpoptOptions = {"IPRINT": -1}
    ipoptOptions = {
        "print_level": 5,
        "tol": 1e-2,
        # "mu_strategy": "adaptive",
        # "alpha_red_factor": 0.2
    }
    opt.add_optimizer("ipopt", ipoptOptions)

    # Setup and check optimization problem
    opt.setup()
    opt.optProb.printSparsity()
    if plot:
        opt.print()
    # exit(0)

    # Run
    sol = opt.optimize(sens=sens)
    # sol = opt.optimize(sens="FD", sensStep=1e-6)
    p_bat_star = sol.xStar["p_bat"]
    e_bat_star = sol.xStar["e_bat"]

    # Check Solution
    if plot:
        # print(sol)

        battery_soc = []
        for h in range(PARAMS["N_HOURS"]):
            battery_soc.append(e_bat_star[h] / PARAMS["MAX_BAT_CAPACITY"])

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
                            "y": (-p_gen + p_bat_star),
                            "label": "From grid",
                        },
                        {
                            "x": hours,
                            "y": p_bat_star,
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
