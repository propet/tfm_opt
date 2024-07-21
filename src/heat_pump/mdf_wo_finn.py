import numpy as np
from pyoptsparse import OPT
from opt import Opt
from utils import get_dynamic_parameters, ks_max
from parameters import PARAMS
from typing import List, Dict
from custom_types import DesignVariables, PlotData, DesignVariableInfo, ConstraintInfo, Parameters
import scipy.sparse as sp
from heat_pump.simulate_wo_finn import dae_forward, dae_adjoints, get_p_heat


def obj(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    # DAE model
    p_compressor = design_variables["p_compressor"]
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_load = design_variables["m_dot_load"]
    u = [p_compressor, m_dot_cond, m_dot_load]

    h = parameters["H"]
    cost_grid = parameters["cost_grid"]
    q_dot_required = parameters["q_dot_required"]
    y0 = parameters["y0"]

    dae_p = [
        parameters["CP_WATER"],
        parameters["TANK_VOLUME"] * parameters["RHO_WATER"],  # tank mass
        parameters["U"],
        6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3),  # tank surface area (m2)
        parameters["T_AMB"],
        parameters["LOAD_HX_EFF"]
    ]

    n_steps = len(cost_grid)
    t_tank = dae_forward(y0, u, dae_p, n_steps)
    p_heat = get_p_heat(q_dot_required, t_tank, m_dot_load)
    p_heat = ks_max(np.zeros(p_heat.shape), p_heat)  # don't get paid by wasting heat

    # Objective function
    return np.sum(cost_grid * (p_compressor + p_heat) * h)


def battery_soc_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
    # Inequality constraint
    # E_{bat_h} / E_{bat_{max}
    e_bat = design_variables["e_bat"]
    battery_soc = []

    for h in range(parameters["N_HOURS"]):
        battery_soc.append(e_bat[h] / parameters["MAX_BAT_CAPACITY"])

    return np.array(battery_soc)


def battery_energy_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
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


def grid_power_constraint_fun(design_variables: DesignVariables, parameters: Parameters) -> np.ndarray:
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
    dsoc_de = sp.eye(N_HOURS, format='csr') / MAX_BAT_CAPACITY

    # grid power constraints
    dgrid_dp = sp.eye(N_HOURS, format='csr')

    # battery energy constraints
    denergy_dp = sp.eye(N_HOURS, format='csr') * -1
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
        return {
            'csr': [mat.indptr, mat.indices, mat.data],
            'shape': list(mat.shape)
        }

    return (
        to_required_format(dsoc_de),
        to_required_format(dgrid_dp),
        to_required_format(denergy_dp),
        to_required_format(denergy_de)
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
        denergy_de[i, i-1] = -1

    return dsoc_de, dgrid_dp, denergy_dp, denergy_de


def sens(design_variables: DesignVariables, func_values):
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


def run_optimization(parameters, plot=True):
    opt = Opt("mdf_wo_finn", obj)
    opt.add_parameters(parameters)

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

    opt.setup()

    opt.optProb.printSparsity()

    # Check optimization problem
    if plot:
        opt.print()
    # exit(0)

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
    # sol = optimizer(opt.optProb, sens="FD", sensStep=1e-6)
    sol = optimizer(opt.optProb, sens=sens)
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

    return sol.xStar, sol.fStar


if __name__ == "__main__":
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    t0 = 0
    y0 = [298.34089176, 309.70395426]  # T_tank, T_cond

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["cost_grid"] = dynamic_parameters["cost_grid"]
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["p_solar_gen"] = dynamic_parameters["p_solar_gen"]
    parameters["p_solar_gen"] = dynamic_parameters["p_solar_gen"]
    parameters["y0"] = y0

    run_optimization(parameters, plot=True)
