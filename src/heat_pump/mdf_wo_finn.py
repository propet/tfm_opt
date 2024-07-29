import numpy as np
from pyoptsparse import OPT, History
from opt import Opt
from utils import get_dynamic_parameters, plot_film, save_dict_to_file
from parameters import PARAMS
from typing import List, Dict
from custom_types import DesignVariables, PlotData, DesignVariableInfo, ConstraintInfo, Parameters
import scipy.sparse as sp
from heat_pump.simulate_wo_finn import cost_function, dae_forward, dae_adjoints, plot


def obj(opt, design_variables: DesignVariables) -> np.ndarray:
    parameters = opt.parameters
    y0 = parameters["y0"]
    n_steps = parameters["n_steps"]

    # DAE model
    p_compressor = design_variables["p_compressor"]
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_load = design_variables["m_dot_load"]
    u = np.array([p_compressor, m_dot_cond, m_dot_load])
    opt.parameters["u"] = u

    # May have some design variables in here as well
    dae_p = np.array(
        [
            parameters["CP_WATER"],
            parameters["TANK_VOLUME"] * parameters["RHO_WATER"],  # tank mass
            parameters["U"],
            6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3),  # tank surface area (m2)
            parameters["T_AMB"],
            parameters["LOAD_HX_EFF"],
        ]
    )
    opt.parameters["dae_p"] = dae_p

    y = dae_forward(y0, u, dae_p, n_steps)

    plot(y, u, n_steps, parameters, block=False)

    opt.parameters["y"] = y
    cost = cost_function(y, u, parameters)

    return np.array(cost)


def sens(opt, design_variables: DesignVariables, func_values):
    parameters = opt.parameters
    dCdy_0_adj, dCdp_adj, dCdu_adj = dae_adjoints(
        parameters["y"], parameters["u"], parameters["dae_p"], parameters["n_steps"], parameters
    )

    return {
        "obj": {
            "p_compressor": dCdu_adj[0],
            "m_dot_cond": dCdu_adj[1],
            "m_dot_load": dCdu_adj[2],
        },
    }


def run_optimization(parameters, plot=True):
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    n_steps = int(horizon / h)

    opt = Opt("mdf_wo_finn", obj, historyFileName="saves/mdf_wo_finn.hst")

    parameters["n_steps"] = n_steps
    opt.add_parameters(parameters)

    # Design Variables
    p_compressor: DesignVariableInfo = {
        "name": "p_compressor",
        "n_params": n_steps,
        "type": "c",
        "lower": 0,
        "upper": PARAMS["P_COMPRESSOR_MAX"],
        "initial_value": PARAMS["P_COMPRESSOR_MAX"] / 2,
        "scale": 1 / PARAMS["P_COMPRESSOR_MAX"],
    }
    opt.add_design_variables_info(p_compressor)

    m_dot_cond: DesignVariableInfo = {
        "name": "m_dot_cond",
        "n_params": n_steps,
        "type": "c",
        "lower": 0,
        "upper": PARAMS["M_DOT_COND_MAX"],
        "initial_value": PARAMS["M_DOT_COND_MAX"] / 2,
        "scale": 1 / PARAMS["M_DOT_COND_MAX"],
    }
    opt.add_design_variables_info(m_dot_cond)

    m_dot_load: DesignVariableInfo = {
        "name": "m_dot_load",
        "n_params": n_steps,
        "type": "c",
        "lower": 0,
        "upper": PARAMS["M_DOT_LOAD_MAX"],
        "initial_value": PARAMS["M_DOT_LOAD_MAX"] / 2,
        "scale": 1 / PARAMS["M_DOT_LOAD_MAX"],
    }
    opt.add_design_variables_info(m_dot_load)

    # Optimizer
    slsqpoptOptions = {"IPRINT": -1}
    ipoptOptions = {
        "print_level": 5,
        "max_iter": 50,
        "tol": 1e-2,
        "obj_scaling_factor": 1e-9,  # tells IPOPT how to internally handle the scaling without distorting the gradients
        "acceptable_tol": 1e-3,
        "acceptable_obj_change_tol": 1e-3,
        # "mu_strategy": "adaptive",
        # "alpha_red_factor": 0.2
    }
    opt.add_optimizer("ipopt", ipoptOptions)

    # Setup and check optimization problem
    opt.setup()
    opt.print()

    # Run
    sol = opt.optimize(sens=sens)  # adjoint gradients
    # sol = opt.optimize(sens="FD", sensStep=1e-6)
    plot_film("saves/mdf_wo_finn.gif")  # create animation with pictures from tmp folder

    p_compressor = sol.xStar["p_compressor"]
    m_dot_cond = sol.xStar["m_dot_cond"]
    m_dot_load = sol.xStar["m_dot_load"]

    # Check Solution
    if plot:
        # print(sol)
        exit(0)

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
    y0 = np.array([298.34089176, 309.70395426])  # T_tank, T_cond

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["cost_grid"] = dynamic_parameters["cost_grid"]
    # print(parameters["cost_grid"])
    # exit(0)
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["y0"] = y0

    run_optimization(parameters, plot=True)
