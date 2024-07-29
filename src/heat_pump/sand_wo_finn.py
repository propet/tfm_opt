import numpy as np
from pyoptsparse import OPT, History
import matplotlib.pyplot as plt
from parameters import PARAMS
from opt import Opt
from utils import get_dynamic_parameters, plot_film, jax_to_numpy
from typing import List, Dict
from custom_types import DesignVariables, PlotData, DesignVariableInfo, ConstraintInfo, Parameters
import scipy.sparse as sp
from heat_pump.simulate_wo_finn import dae_forward, dae_adjoints, plot, cop, get_dcopdT
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def obj(opt, design_variables: DesignVariables) -> np.ndarray:
    # Parameters
    parameters = opt.parameters
    h = parameters["H"]
    cost_grid = parameters["cost_grid"]

    # Design variables
    p_compressor = design_variables["p_compressor"]

    cost = cost_function(p_compressor, h, cost_grid)
    return np.array(cost)


def cost_function(p_compressor, h, cost_grid):
    cost = jnp.sum(h * cost_grid * p_compressor)
    return cost


get_dcostdp_compressor_jax = jax.jit(jax.jacobian(cost_function, argnums=0))
get_dcostdp_compressor = jax_to_numpy(get_dcostdp_compressor_jax)


def dae1_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # m_tank * cp_water * ((T_tank - T_tank_prev)/h)
    #   - m_dot_cond * cp_water * T_cond
    #     - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_amb))
    #   + (m_dot_cond + m_dot_load) * cp_water * T_tank
    #   + U * A * (T_tank - T_amb)
    #   = 0

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    load_hx_eff = parameters["LOAD_HX_EFF"]
    cp_water = parameters["CP_WATER"]
    h = parameters["H"]
    t_amb = parameters["t_amb"]
    m_tank = parameters["TANK_VOLUME"] * parameters["RHO_WATER"]
    U = parameters["U"]
    A = 6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3)  # tank surface area (m2)

    # Design variables
    t_tank = design_variables["t_tank"]
    t_cond = design_variables["t_cond"]
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_load = design_variables["m_dot_load"]

    dae1 = []
    for i in range(1, n_steps):
        dae1.append(
            m_tank * cp_water * ((t_tank[i] - t_tank[i - 1]) / h)
            - m_dot_cond[i] * cp_water * t_cond[i]
            - m_dot_load[i] * cp_water * (t_tank[i] - load_hx_eff * (t_tank[i] - t_amb[i]))
            + (m_dot_cond[i] + m_dot_load[i]) * cp_water * t_tank[i]
            + U * A * (t_tank[i] - t_amb[i])
        )

    return np.array(dae1)


def dae2_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # cop(T_cond) * P_comp - m_dot_cond * cp_water * (T_cond - T_tank) = 0

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    cp_water = parameters["CP_WATER"]

    # Design variables
    t_tank = design_variables["t_tank"]
    p_compressor = design_variables["p_compressor"]
    t_cond = design_variables["t_cond"]
    m_dot_cond = design_variables["m_dot_cond"]

    dae2 = cop(t_cond) * p_compressor - m_dot_cond * cp_water * (t_cond - t_tank)
    return dae2

    # dae2 = []
    # for i in range(1, n_steps):
    #     dae2.append(cop(t_cond[i]) * p_compressor[i] - m_dot_cond[i] * cp_water * (t_cond[i] - t_tank[i]))
    #
    # return np.array(dae2)


def q_required_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Q_dot_load should be +-5% of q_dot_required
    # Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)
    # 0.95 < Q_dot_load / Q_dot_required < 1.05

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    q_dot_required = parameters["q_dot_required"]
    t_amb = parameters["t_amb"]
    cp_water = parameters["CP_WATER"]
    load_hx_eff = parameters["LOAD_HX_EFF"]

    # Design variables
    t_tank = design_variables["t_tank"]
    m_dot_load = design_variables["m_dot_load"]

    # Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)
    q_dot_load = load_hx_eff * m_dot_load * cp_water * (t_tank - t_amb)
    return q_dot_load / (q_dot_required + 1e-6)

    # q_dot_required_con = []
    # for i in range(1, n_steps):
    #     q_dot_load_i = load_hx_eff * m_dot_load[i] * cp_water * (t_tank[i] - t_amb[i])
    #     q_dot_required_con.append(q_dot_load_i / (q_dot_required[i] + 1e-6))
    #
    # return np.array(q_dot_required_con)


def get_constraint_sparse_jacs(parameters, design_variables):
    def to_required_format(mat):
        return {"csr": [mat.indptr, mat.indices, mat.data], "shape": list(mat.shape)}

    # Parameters
    n_steps = parameters["n_steps"]
    load_hx_eff = parameters["LOAD_HX_EFF"]
    cp_water = parameters["CP_WATER"]
    h = parameters["H"]
    m_tank = parameters["TANK_VOLUME"] * parameters["RHO_WATER"]
    U = parameters["U"]
    A = 6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3)  # tank surface area (m2)
    cost_grid = parameters["cost_grid"]
    t_amb = parameters["t_amb"]
    q_dot_required = parameters["q_dot_required"]
    t_tank = design_variables["t_tank"]
    m_dot_load = design_variables["m_dot_load"]

    # Design variables
    p_compressor = design_variables["p_compressor"]
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_load = design_variables["m_dot_load"]
    t_tank = design_variables["t_tank"]
    t_cond = design_variables["t_cond"]

    # dae1:
    # m_tank * cp_water * ((T_tank - T_tank_prev)/h)
    #     - m_dot_cond * cp_water * T_cond
    #     - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_amb))
    #     + (m_dot_cond + m_dot_load) * cp_water * T_tank
    #     + U * A * (T_tank - T_amb)
    #     = 0
    ddae1_dm_dot_cond = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        ddae1_dm_dot_cond[i - 1, i] = -cp_water * t_cond[i] + cp_water * t_tank[i] + 1e-20
    ddae1_dm_dot_cond = ddae1_dm_dot_cond.tocsr()
    ddae1_dm_dot_cond = to_required_format(ddae1_dm_dot_cond)

    ddae1_dm_dot_load = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        ddae1_dm_dot_load[i - 1, i] = (
            -cp_water * (t_tank[i] - load_hx_eff * (t_tank[i] - t_amb[i])) + cp_water * t_tank[i] + 1e-20
        )
    ddae1_dm_dot_load = ddae1_dm_dot_load.tocsr()
    ddae1_dm_dot_load = to_required_format(ddae1_dm_dot_load)

    ddae1_dt_tank = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        ddae1_dt_tank[i - 1, i] = (
            m_tank * cp_water / h
            - m_dot_load[i] * cp_water
            + m_dot_load[i] * cp_water * load_hx_eff
            + (m_dot_cond[i] + m_dot_load[i]) * cp_water
            + U * A
            + 1e-20
        )
        ddae1_dt_tank[i - 1, i - 1] = -m_tank * cp_water / h + 1e-20
    ddae1_dt_tank = ddae1_dt_tank.tocsr()
    ddae1_dt_tank = to_required_format(ddae1_dt_tank)
    # plt.spy(ddae1_dt_tank)
    # plt.show()

    ddae1_dt_cond = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        ddae1_dt_cond[i - 1, i] = -m_dot_cond[i] * cp_water + 1e-20
    ddae1_dt_cond = ddae1_dt_cond.tocsr()
    ddae1_dt_cond = to_required_format(ddae1_dt_cond)

    dae1_jac = {
        "m_dot_cond": ddae1_dm_dot_cond,
        "m_dot_load": ddae1_dm_dot_load,
        "t_tank": ddae1_dt_tank,
        "t_cond": ddae1_dt_cond,
    }
    dae1_wrt = [
        "m_dot_cond",
        "m_dot_load",
        "t_tank",
        "t_cond",
    ]

    # dae2:
    # cop(T_cond) * P_comp - m_dot_cond * cp_water * (T_cond - T_tank) = 0

    # ddae2_dp_compressor = sp.lil_matrix((n_steps - 1, n_steps))
    # for i in range(1, n_steps):
    #     ddae2_dp_compressor[i - 1, i] = cop(t_cond[i]) + 1e-20
    # ddae2_dp_compressor = ddae2_dp_compressor.tocsr()
    # ddae2_dp_compressor = to_required_format(ddae2_dp_compressor)
    #
    # ddae2_dm_dot_cond = sp.lil_matrix((n_steps - 1, n_steps))
    # for i in range(1, n_steps):
    #     ddae2_dm_dot_cond[i - 1, i] = -cp_water * (t_cond[i] - t_tank[i]) + 1e-20
    # ddae2_dm_dot_cond = ddae2_dm_dot_cond.tocsr()
    # ddae2_dm_dot_cond = to_required_format(ddae2_dm_dot_cond)
    #
    # ddae2_dt_tank = sp.lil_matrix((n_steps - 1, n_steps))
    # for i in range(1, n_steps):
    #     ddae2_dt_tank[i - 1, i] = m_dot_cond[i] * cp_water + 1e-20
    # ddae2_dt_tank = ddae2_dt_tank.tocsr()
    # ddae2_dt_tank = to_required_format(ddae2_dt_tank)
    #
    # ddae2_dt_cond = sp.lil_matrix((n_steps - 1, n_steps))
    # for i in range(1, n_steps):
    #     ddae2_dt_cond[i - 1, i] = get_dcopdT(t_cond[i]) - m_dot_cond[i] * cp_water + 1e-20
    # ddae2_dt_cond = ddae2_dt_cond.tocsr()
    # ddae2_dt_cond = to_required_format(ddae2_dt_cond)

    ddae2_dp_compressor = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        ddae2_dp_compressor[i, i] = cop(t_cond[i]) + 1e-20
    ddae2_dp_compressor = ddae2_dp_compressor.tocsr()
    ddae2_dp_compressor = to_required_format(ddae2_dp_compressor)

    ddae2_dm_dot_cond = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        ddae2_dm_dot_cond[i, i] = -cp_water * (t_cond[i] - t_tank[i]) + 1e-20
    ddae2_dm_dot_cond = ddae2_dm_dot_cond.tocsr()
    ddae2_dm_dot_cond = to_required_format(ddae2_dm_dot_cond)

    ddae2_dt_tank = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        ddae2_dt_tank[i, i] = m_dot_cond[i] * cp_water + 1e-20
    ddae2_dt_tank = ddae2_dt_tank.tocsr()
    ddae2_dt_tank = to_required_format(ddae2_dt_tank)

    ddae2_dt_cond = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        ddae2_dt_cond[i, i] = get_dcopdT(t_cond[i]) - m_dot_cond[i] * cp_water + 1e-20
    ddae2_dt_cond = ddae2_dt_cond.tocsr()
    ddae2_dt_cond = to_required_format(ddae2_dt_cond)

    dae2_jac = {
        "p_compressor": ddae2_dp_compressor,
        "m_dot_cond": ddae2_dm_dot_cond,
        "t_tank": ddae2_dt_tank,
        "t_cond": ddae2_dt_cond,
    }
    dae2_wrt = [
        "p_compressor",
        "m_dot_cond",
        "t_tank",
        "t_cond",
    ]

    # q_required:
    # Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)
    # 0.95 < Q_dot_load / Q_dot_required < 1.05
    # ->
    # (1 / q_dot_required) * (load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb))

    # dq_required_dm_dot_load = sp.lil_matrix((n_steps - 1, n_steps))
    # for i in range(1, n_steps):
    #     dq_required_dm_dot_load[i - 1, i] = (1 / (q_dot_required[i] + 1e-6)) * (load_hx_eff * cp_water * (t_tank[i] - t_amb[i])) + 1e-20
    # dq_required_dm_dot_load = dq_required_dm_dot_load.tocsr()
    # dq_required_dm_dot_load = to_required_format(dq_required_dm_dot_load)
    #
    # dq_required_dt_tank = sp.lil_matrix((n_steps - 1, n_steps))
    # # (1 / q_dot_required) * (load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb))
    # for i in range(1, n_steps):
    #     dq_required_dt_tank[i - 1, i] = (1 / (q_dot_required[i] + 1e-6)) * (load_hx_eff * m_dot_load[i] * cp_water) + 1e-20
    # dq_required_dt_tank = dq_required_dt_tank.tocsr()
    # dq_required_dt_tank = to_required_format(dq_required_dt_tank)

    dq_required_dm_dot_load = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        dq_required_dm_dot_load[i, i] = (1 / (q_dot_required[i] + 1e-6)) * (
            load_hx_eff * cp_water * (t_tank[i] - t_amb[i])
        ) + 1e-20
    dq_required_dm_dot_load = dq_required_dm_dot_load.tocsr()
    dq_required_dm_dot_load = to_required_format(dq_required_dm_dot_load)

    dq_required_dt_tank = sp.lil_matrix((n_steps, n_steps))
    # (1 / q_dot_required) * (load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb))
    for i in range(n_steps):
        dq_required_dt_tank[i, i] = (1 / (q_dot_required[i] + 1e-6)) * (load_hx_eff * m_dot_load[i] * cp_water) + 1e-20
    dq_required_dt_tank = dq_required_dt_tank.tocsr()
    dq_required_dt_tank = to_required_format(dq_required_dt_tank)

    q_required_jac = {
        "m_dot_load": dq_required_dm_dot_load,
        "t_tank": dq_required_dt_tank,
    }
    q_required_wrt = [
        "m_dot_load",
        "t_tank",
    ]

    # Initial condition constraints
    dp_t_tank_0_dm_dot_con = sp.lil_matrix((1, n_steps))
    dp_t_tank_0_dm_dot_con[0, 0] = 1
    dp_t_tank_0_dm_dot_con = dp_t_tank_0_dm_dot_con.tocsr()
    dp_t_tank_0_dm_dot_con = to_required_format(dp_t_tank_0_dm_dot_con)
    t_tank_0_jac = {"t_tank": dp_t_tank_0_dm_dot_con}
    t_tank_0_wrt = ["t_tank"]

    dp_t_cond_0_dm_dot_con = sp.lil_matrix((1, n_steps))
    dp_t_cond_0_dm_dot_con[0, 0] = 1
    dp_t_cond_0_dm_dot_con = dp_t_cond_0_dm_dot_con.tocsr()
    dp_t_cond_0_dm_dot_con = to_required_format(dp_t_cond_0_dm_dot_con)
    t_cond_0_jac = {"t_cond": dp_t_cond_0_dm_dot_con}
    t_cond_0_wrt = ["t_cond"]

    return (
        dae1_jac,
        dae1_wrt,
        dae2_jac,
        dae2_wrt,
        q_required_wrt,
        q_required_jac,
        t_tank_0_jac,
        t_tank_0_wrt,
        t_cond_0_jac,
        t_cond_0_wrt,
    )


def sens(opt, design_variables: DesignVariables, func_values):
    parameters = opt.parameters
    load_hx_eff = parameters["LOAD_HX_EFF"]
    cp_water = parameters["CP_WATER"]
    h = parameters["H"]
    cost_grid = parameters["cost_grid"]
    t_amb = parameters["t_amb"]
    t_tank = design_variables["t_tank"]
    p_compressor = design_variables["p_compressor"]
    m_dot_load = design_variables["m_dot_load"]

    (
        dae1_jac,
        dae1_wrt,
        dae2_jac,
        dae2_wrt,
        q_required_wrt,
        q_required_jac,
        t_tank_0_jac,
        t_tank_0_wrt,
        t_cond_0_jac,
        t_cond_0_wrt,
    ) = get_constraint_sparse_jacs(parameters, design_variables)

    # def cost_function(p_compressor, h, cost_grid):
    dcostdp_compressor = get_dcostdp_compressor(p_compressor, h, cost_grid)

    return {
        # "obj": {
        #     "p_compressor": h * cost_grid,
        #     # "m_dot_cond": 0,
        #     "m_dot_load": -h * cost_grid * load_hx_eff * cp_water * (t_tank - t_amb),
        #     "t_tank": -h * cost_grid * load_hx_eff * m_dot_load * cp_water,
        #     # "t_cond": 0,
        # },
        "obj": {
            "p_compressor": dcostdp_compressor,
        },
        "dae1_constraint": dae1_jac,
        "dae2_constraint": dae2_jac,
        "q_required_constraint": q_required_jac,
        "t_tank_0": t_tank_0_jac,
        "t_cond_0": t_cond_0_jac,
    }


def run_optimization(parameters, plot=True):
    opt = Opt("sand_wo_finn", obj, historyFileName="saves/sand_wo_finn.hst")

    # Parameters
    h = parameters["H"]
    horizon = parameters["HORIZON"]
    n_steps = int(horizon / h)
    parameters["n_steps"] = n_steps
    opt.add_parameters(parameters)

    # Design Variables
    p_compressor: DesignVariableInfo = {
        "name": "p_compressor",
        "n_params": n_steps,
        "type": "c",
        "lower": 1e-3,
        "upper": parameters["P_COMPRESSOR_MAX"],
        "initial_value": parameters["P_COMPRESSOR_MAX"] / 2,
        "scale": 1 / parameters["P_COMPRESSOR_MAX"],
    }
    opt.add_design_variables_info(p_compressor)

    m_dot_cond: DesignVariableInfo = {
        "name": "m_dot_cond",
        "n_params": n_steps,
        "type": "c",
        "lower": 1e-3,
        "upper": parameters["M_DOT_COND_MAX"],
        "initial_value": parameters["M_DOT_COND_MAX"] / 2,
        "scale": 1 / parameters["M_DOT_COND_MAX"],
    }
    opt.add_design_variables_info(m_dot_cond)

    m_dot_load: DesignVariableInfo = {
        "name": "m_dot_load",
        "n_params": n_steps,
        "type": "c",
        "lower": 1e-4,
        "upper": parameters["M_DOT_LOAD_MAX"],
        "initial_value": parameters["M_DOT_LOAD_MAX"] / 2,
        "scale": 1 / parameters["M_DOT_LOAD_MAX"],
    }
    opt.add_design_variables_info(m_dot_load)

    t_tank: DesignVariableInfo = {
        "name": "t_tank",
        "n_params": n_steps,
        "type": "c",
        "lower": 273,
        "upper": 500,
        "initial_value": 300,
        "scale": 1 / 300,
    }
    opt.add_design_variables_info(t_tank)

    t_cond: DesignVariableInfo = {
        "name": "t_cond",
        "n_params": n_steps,
        "type": "c",
        "lower": 273,
        "upper": 500,
        "initial_value": 300,
        "scale": 1 / 300,
    }
    opt.add_design_variables_info(t_cond)

    # Constraints

    # For non-linear constraints, only the sparsity structure
    # (i.e. which entries are nonzero) is significant.
    # The values themselves will be determined by a call to the sens() function.
    # So we get the sparsity of the jacobians, by evaluating the jacobians
    # with some dummy design variables
    dummy_design_variables = {
        "p_compressor": np.ones(n_steps),
        "m_dot_cond": np.ones(n_steps),
        "m_dot_load": np.ones(n_steps),
        "t_tank": np.ones(n_steps),
        "t_cond": np.ones(n_steps),
    }

    (
        dae1_jac,
        dae1_wrt,
        dae2_jac,
        dae2_wrt,
        q_required_wrt,
        q_required_jac,
        t_tank_0_jac,
        t_tank_0_wrt,
        t_cond_0_jac,
        t_cond_0_wrt,
    ) = get_constraint_sparse_jacs(parameters, dummy_design_variables)

    dae1_constraint: ConstraintInfo = {
        "name": "dae1_constraint",
        "n_constraints": n_steps - 1,
        "lower": 0,
        "upper": 0,
        "function": dae1_constraint_fun,
        # "scale": 1,
        "scale": 1 / (parameters["CP_WATER"] * 300),
        "wrt": dae1_wrt,
        "jac": dae1_jac,
    }
    opt.add_constraint_info(dae1_constraint)

    dae2_constraint: ConstraintInfo = {
        "name": "dae2_constraint",
        "n_constraints": n_steps,
        "lower": 0,
        "upper": 0,
        "function": dae2_constraint_fun,
        # "scale": 1,
        "scale": 1 / parameters["P_COMPRESSOR_MAX"],
        "wrt": dae2_wrt,
        "jac": dae2_jac,
    }
    opt.add_constraint_info(dae2_constraint)

    q_required_constraint: ConstraintInfo = {
        "name": "q_required_constraint",
        "n_constraints": n_steps,
        "lower": 0.95,
        "upper": 1.05,
        "function": q_required_constraint_fun,
        "scale": 1,
        "wrt": q_required_wrt,
        "jac": q_required_jac,
    }
    opt.add_constraint_info(q_required_constraint)

    # Initial value constraints
    t_tank_0: ConstraintInfo = {
        "name": "t_tank_0",
        "n_constraints": 1,
        "lower": parameters["y0"]["t_tank"],
        "upper": parameters["y0"]["t_tank"],
        "function": lambda _, design_variables: design_variables["t_tank"][0],
        "scale": 1 / parameters["y0"]["t_tank"],
        "wrt": t_tank_0_wrt,
        "jac": t_tank_0_jac,
    }
    opt.add_constraint_info(t_tank_0)

    t_cond_0: ConstraintInfo = {
        "name": "t_cond_0",
        "n_constraints": 1,
        "lower": parameters["y0"]["t_cond"],
        "upper": parameters["y0"]["t_cond"],
        "function": lambda _, design_variables: design_variables["t_cond"][0],
        "scale": 1 / parameters["y0"]["t_cond"],
        "wrt": t_cond_0_wrt,
        "jac": t_cond_0_jac,
    }
    opt.add_constraint_info(t_cond_0)

    # Optimizer
    slsqpoptOptions = {"IPRINT": -1}
    ipoptOptions = {
        "print_level": 5,
        "max_iter": 200,
        # "tol": 1e-3,
        # "obj_scaling_factor": 1e7,  # tells IPOPT how to internally handle the scaling without distorting the gradients
        # "nlp_scaling_method": "gradient-based",
        # "acceptable_tol": 1e-4,
        # "acceptable_obj_change_tol": 1e-4,
        # "mu_strategy": "adaptive",
        # "alpha_red_factor": 0.2
    }
    opt.add_optimizer("ipopt", ipoptOptions)

    # Setup and check optimization problem
    opt.setup()
    opt.print()
    opt.optProb.printSparsity()
    # exit(0)

    # Run
    sol = opt.optimize(sens=sens)
    # sol = opt.optimize(sens="FD", sensStep=1e-6)

    p_compressor = sol.xStar["p_compressor"]
    m_dot_cond = sol.xStar["m_dot_cond"]
    m_dot_load = sol.xStar["m_dot_load"]

    # Check Solution
    if plot:
        # print(sol)
        exit(0)
    return sol.xStar, sol.fStar


if __name__ == "__main__":
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    t0 = 0

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon, year=2022)
    parameters = PARAMS
    parameters["cost_grid"] = dynamic_parameters["cost_grid"]
    # print(parameters["cost_grid"])
    # exit(0)
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    # print("q_dot_required: ", parameters["q_dot_required"].shape)
    # print(parameters["q_dot_required"])
    # exit(0)
    parameters["p_required"] = dynamic_parameters["p_required"]
    print("p_required: ", parameters["p_required"].shape)
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    print("t_amb: ", parameters["t_amb"].shape)
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    print("w_solar_per_w_installed: ", parameters["w_solar_per_w_installed"].shape)

    y0 = {
        "p_compressor": 900,  # up to P_COMPRESSOR_MAX
        "m_dot_cond": 4.98853131e-01,
        "m_dot_load": 0.1,
        "t_tank": 298.3,
        "t_cond": 298.5,
    }
    parameters["y0"] = y0

    run_optimization(parameters, plot=True)
