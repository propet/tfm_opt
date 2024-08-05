import numpy as np
from pyoptsparse import OPT, History
import matplotlib.pyplot as plt
from parameters import PARAMS
from opt import Opt
from utils import (
    get_dynamic_parameters,
    cop,
    get_dcopdT,
    get_solar_panels_depreciation_by_second,
    jax_to_numpy,
    get_battery_depreciation_by_joule,
    get_battery_depreciation_by_second,
    get_solar_panels_depreciation_by_second,
    get_tank_depreciation_by_second,
    get_hp_depreciation_by_joule,
    get_fixed_energy_cost_by_second,
)
from typing import List, Dict
from custom_types import DesignVariables, PlotData, DesignVariableInfo, ConstraintInfo, Parameters
import scipy.sparse as sp
from full_system.simulate_simple_hp import dae_forward, dae_adjoints, plot
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def obj(opt, design_variables: DesignVariables) -> np.ndarray:
    # Design variables
    p_compressor = design_variables["p_compressor"]
    p_bat = design_variables["p_bat"]

    # Sizing
    parameters = opt.parameters
    e_bat_max = parameters["E_BAT_MAX"]
    solar_size = parameters["SOLAR_SIZE"]
    p_compressor_max = parameters["P_COMPRESSOR_MAX"]
    tank_volume = parameters["TANK_VOLUME"]
    p_grid_max = parameters["P_GRID_MAX"]

    # Parameters
    h = np.ones(p_compressor.shape[0]) * parameters["H"]
    p_required = parameters["p_required"]
    p_solar = parameters["w_solar_per_w_installed"] * solar_size
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]

    cost = cost_function(
        h,
        p_bat,
        p_compressor,
        p_required,
        p_solar,
        e_bat_max,
        solar_size,
        p_compressor_max,
        p_grid_max,
        tank_volume,
        pvpc_prices,
        excess_prices,
    )
    return np.array(cost)


def cost_function(
    h,
    p_bat,
    p_compressor,
    p_required,
    p_solar,
    e_bat_max,
    solar_size,
    p_compressor_max,
    p_grid_max,
    tank_volume,
    pvpc_prices,
    excess_prices,
):
    p_grid = -p_solar + p_compressor + p_bat + p_required
    cost = (
        # Variable energy cost
        # buy at pvpc price, sell at excess price
        jnp.sum(h * jnp.maximum(pvpc_prices * p_grid, excess_prices * p_grid))
        # buy and sell at the same daily_price
        # jnp.sum(h * excess_prices * p_grid)
        # Fixed energy cost
        + jnp.sum(h * get_fixed_energy_cost_by_second(p_grid_max))
        # depreciate battery by usage
        + jnp.sum(h * jnp.abs(p_bat) * get_battery_depreciation_by_joule(e_bat_max))
        # depreciate battery by time
        # + jnp.sum(h * get_battery_depreciation_by_second(e_bat_max))
        # depreciate solar panels by time
        + jnp.sum(h * get_solar_panels_depreciation_by_second(solar_size))
        # depreciate heat pump by usage
        + jnp.sum(h * jnp.abs(p_compressor) * get_hp_depreciation_by_joule(p_compressor_max))
        # depreciate water tank by time
        + jnp.sum(h * get_tank_depreciation_by_second(tank_volume))
    )
    return cost


get_dcostdp_bat_jax = jax.jit(jax.jacobian(cost_function, argnums=1))
get_dcostdp_compressor_jax = jax.jit(jax.jacobian(cost_function, argnums=2))
get_dcostdp_bat = jax_to_numpy(get_dcostdp_bat_jax)
get_dcostdp_compressor = jax_to_numpy(get_dcostdp_compressor_jax)


##############
# Constraints
##############
def dae1_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # m_tank * cp_water * ((T_tank - T_tank_prev)/h)
    #   - m_dot_cond * cp_water * T_cond
    #     - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_target))
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
    t_target = parameters["T_TARGET"]
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
            - m_dot_load[i] * cp_water * (t_tank[i] - load_hx_eff * (t_tank[i] - t_target))
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


def q_required_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Q_dot_load should track Q_dot_required with some tolerance
    # Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_target)
    # Q_dot_required - tol < Q_dot_load < Q_dot_required + tol
    # ->
    # -tol < Q_dot_load - Q_dot_required < tol

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    q_dot_required = parameters["q_dot_required"]
    t_target = parameters["T_TARGET"]
    cp_water = parameters["CP_WATER"]
    load_hx_eff = parameters["LOAD_HX_EFF"]

    # Design variables
    t_tank = design_variables["t_tank"]
    m_dot_load = design_variables["m_dot_load"]

    # Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_target)
    q_dot_load = load_hx_eff * m_dot_load * cp_water * (t_tank - t_target)
    return q_dot_load - q_dot_required


def battery_soc_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Inequality constraint
    # E_{bat_h} / E_{bat_{max}

    # Parameters
    parameters = opt.parameters
    e_bat_max = parameters["E_BAT_MAX"]  # or as a design variable

    # Design variables
    e_bat = design_variables["e_bat"]

    return e_bat / e_bat_max


def battery_energy_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Equality constraint
    # e_bat_h - e_bat_{h-1} - bat_eta * p_bat_h * h = 0

    # Parameters
    parameters = opt.parameters
    h = parameters["H"]
    n_steps = parameters["n_steps"]
    bat_eta = parameters["BAT_ETA"]

    # Design variables
    p_bat = design_variables["p_bat"]
    e_bat = design_variables["e_bat"]

    energy_constraints = []
    for i in range(1, n_steps):
        constraint_value = e_bat[i] - e_bat[i - 1] - bat_eta * p_bat[i] * h
        energy_constraints.append(constraint_value)

    return np.array(energy_constraints)


def p_grid_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Inequality constraints
    # p_grid = - p_solar + p_compressor + p_bat + p_required
    # -P_GRID_MAX < p_grid < P_GRID_MAX

    # Design variables
    p_bat = design_variables["p_bat"]
    p_compressor = design_variables["p_compressor"]

    # Parameters
    parameters = opt.parameters
    solar_size = parameters["SOLAR_SIZE"]
    p_required = parameters["p_required"]
    p_solar = parameters["w_solar_per_w_installed"] * solar_size

    p_grid = -p_solar + p_compressor + p_bat + p_required
    return p_grid


############
# Gradients
############
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
    t_target = parameters["T_TARGET"]
    q_dot_required = parameters["q_dot_required"]
    e_bat_max = parameters["E_BAT_MAX"]
    bat_eta = parameters["BAT_ETA"]

    # Design variables
    t_tank = design_variables["t_tank"]
    m_dot_load = design_variables["m_dot_load"]
    p_bat = design_variables["p_bat"]
    e_bat = design_variables["e_bat"]
    p_compressor = design_variables["p_compressor"]
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_load = design_variables["m_dot_load"]
    t_cond = design_variables["t_cond"]

    # dae1:
    # m_tank * cp_water * ((T_tank - T_tank_prev)/h)
    #     - m_dot_cond * cp_water * T_cond
    #     - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_target))
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
            -cp_water * (t_tank[i] - load_hx_eff * (t_tank[i] - t_target)) + cp_water * t_tank[i] + 1e-20
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
        ddae2_dt_cond[i, i] = get_dcopdT(t_cond[i]) * p_compressor[i] - m_dot_cond[i] * cp_water + 1e-20
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

    # q_required
    # Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_target)
    # -tol < Q_dot_load - Q_dot_required < tol
    dq_required_dm_dot_load = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        dq_required_dm_dot_load[i, i] = load_hx_eff * cp_water * (t_tank[i] - t_target) + 1e-20
    dq_required_dm_dot_load = dq_required_dm_dot_load.tocsr()
    dq_required_dm_dot_load = to_required_format(dq_required_dm_dot_load)

    dq_required_dt_tank = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        dq_required_dt_tank[i, i] = load_hx_eff * m_dot_load[i] * cp_water + 1e-20
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

    # battery_soc_jac
    # e_bat / e_bat_max
    dbattery_soc_de_bat = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        dbattery_soc_de_bat[i, i] = 1 / e_bat_max + 1e-20
    dbattery_soc_de_bat = dbattery_soc_de_bat.tocsr()
    dbattery_soc_de_bat = to_required_format(dbattery_soc_de_bat)

    battery_soc_jac = {
        "e_bat": dbattery_soc_de_bat,
    }
    battery_soc_wrt = [
        "e_bat",
    ]

    # battery_energy_jac
    # e_bat_h - e_bat_{h-1} - bat_eta * p_bat_h * h = 0
    dbattery_energy_de_bat = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        dbattery_energy_de_bat[i - 1, i] = 1
        dbattery_energy_de_bat[i - 1, i - 1] = -1
    dbattery_energy_de_bat = dbattery_energy_de_bat.tocsr()
    dbattery_energy_de_bat = to_required_format(dbattery_energy_de_bat)

    dbattery_energy_dp_bat = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        dbattery_energy_dp_bat[i - 1, i] = - bat_eta * h
    dbattery_energy_dp_bat = dbattery_energy_dp_bat.tocsr()
    dbattery_energy_dp_bat = to_required_format(dbattery_energy_dp_bat)

    battery_energy_jac = {
        "e_bat": dbattery_energy_de_bat,
        "p_bat": dbattery_energy_dp_bat,
    }
    battery_energy_wrt = [
        "e_bat",
        "p_bat",
    ]

    # p_grid_jac
    # p_grid = - p_solar + p_compressor + p_bat + p_required
    # -P_GRID_MAX < p_grid < P_GRID_MAX
    dp_grid_dp_bat = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        dp_grid_dp_bat[i, i] = 1
    dp_grid_dp_bat = dp_grid_dp_bat.tocsr()
    dp_grid_dp_bat = to_required_format(dp_grid_dp_bat)

    dp_grid_dp_compressor = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        dp_grid_dp_compressor[i, i] = 1
    dp_grid_dp_compressor = dp_grid_dp_compressor.tocsr()
    dp_grid_dp_compressor = to_required_format(dp_grid_dp_compressor)

    p_grid_jac = {
        "p_bat": dp_grid_dp_bat,
        "p_compressor": dp_grid_dp_compressor,
    }
    p_grid_wrt = [
        "p_bat",
        "p_compressor",
    ]

    # Initial condition constraints
    dt_tank_0_dt_tank = sp.lil_matrix((1, n_steps))
    dt_tank_0_dt_tank[0, 0] = 1
    dt_tank_0_dt_tank = dt_tank_0_dt_tank.tocsr()
    dt_tank_0_dt_tank = to_required_format(dt_tank_0_dt_tank)
    t_tank_0_jac = {"t_tank": dt_tank_0_dt_tank}
    t_tank_0_wrt = ["t_tank"]

    dt_cond_0_dt_cond = sp.lil_matrix((1, n_steps))
    dt_cond_0_dt_cond[0, 0] = 1
    dt_cond_0_dt_cond = dt_cond_0_dt_cond.tocsr()
    dt_cond_0_dt_cond = to_required_format(dt_cond_0_dt_cond)
    t_cond_0_jac = {"t_cond": dt_cond_0_dt_cond}
    t_cond_0_wrt = ["t_cond"]

    de_bat_0_de_bat = sp.lil_matrix((1, n_steps))
    de_bat_0_de_bat[0, 0] = 1
    de_bat_0_de_bat = de_bat_0_de_bat.tocsr()
    de_bat_0_de_bat = to_required_format(de_bat_0_de_bat)
    e_bat_0_jac = {"e_bat": de_bat_0_de_bat}
    e_bat_0_wrt = ["e_bat"]

    dp_bat_0_dp_bat = sp.lil_matrix((1, n_steps))
    dp_bat_0_dp_bat[0, 0] = 1
    dp_bat_0_dp_bat = dp_bat_0_dp_bat.tocsr()
    dp_bat_0_dp_bat = to_required_format(dp_bat_0_dp_bat)
    p_bat_0_jac = {"p_bat": dp_bat_0_dp_bat}
    p_bat_0_wrt = ["p_bat"]

    return (
        dae1_jac,
        dae1_wrt,
        dae2_jac,
        dae2_wrt,
        q_required_jac,
        q_required_wrt,
        battery_soc_jac,
        battery_soc_wrt,
        battery_energy_jac,
        battery_energy_wrt,
        p_grid_jac,
        p_grid_wrt,
        t_tank_0_jac,
        t_tank_0_wrt,
        t_cond_0_jac,
        t_cond_0_wrt,
        e_bat_0_jac,
        e_bat_0_wrt,
        p_bat_0_jac,
        p_bat_0_wrt,
    )


def sens(opt, design_variables: DesignVariables, func_values):
    # Parameters
    parameters = opt.parameters
    h = parameters["H"]
    p_required = parameters["p_required"]
    e_bat_max = parameters["E_BAT_MAX"]
    p_compressor_max = parameters["P_COMPRESSOR_MAX"]
    p_grid_max = parameters["P_GRID_MAX"]
    solar_size = parameters["SOLAR_SIZE"]
    p_solar = parameters["w_solar_per_w_installed"] * solar_size
    tank_volume = parameters["TANK_VOLUME"]
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]

    # Design variables
    p_compressor = design_variables["p_compressor"]
    p_bat = design_variables["p_bat"]
    e_bat = design_variables["e_bat"]

    (
        dae1_jac,
        dae1_wrt,
        dae2_jac,
        dae2_wrt,
        q_required_jac,
        q_required_wrt,
        battery_soc_jac,
        battery_soc_wrt,
        battery_energy_jac,
        battery_energy_wrt,
        p_grid_jac,
        p_grid_wrt,
        t_tank_0_jac,
        t_tank_0_wrt,
        t_cond_0_jac,
        t_cond_0_wrt,
        e_bat_0_jac,
        e_bat_0_wrt,
        p_bat_0_jac,
        p_bat_0_wrt,
    ) = get_constraint_sparse_jacs(parameters, design_variables)

    dcostdp_bat = get_dcostdp_bat(
        h,
        p_bat,
        p_compressor,
        p_required,
        p_solar,
        e_bat_max,
        solar_size,
        p_compressor_max,
        p_grid_max,
        tank_volume,
        pvpc_prices,
        excess_prices,
    )
    dcostdp_compressor = get_dcostdp_compressor(
        h,
        p_bat,
        p_compressor,
        p_required,
        p_solar,
        e_bat_max,
        solar_size,
        p_compressor_max,
        p_grid_max,
        tank_volume,
        pvpc_prices,
        excess_prices,
    )

    return {
        "obj": {
            "p_bat": dcostdp_bat,
            "p_compressor": dcostdp_compressor,
        },
        "dae1_constraint": dae1_jac,
        "dae2_constraint": dae2_jac,
        "q_required_constraint": q_required_jac,
        "battery_soc_constraint": battery_soc_jac,
        "battery_energy_constraint": battery_energy_jac,
        "p_grid_constraint": p_grid_jac,
        "t_tank_0_constraint": t_tank_0_jac,
        "t_cond_0_constraint": t_cond_0_jac,
        "e_bat_0_constraint": e_bat_0_jac,
        "p_bat_0_constraint": p_bat_0_jac,
    }


def run_optimization(parameters, plot=True):
    opt = Opt("full_sand_wo_heating", obj, historyFileName="saves/full_sand_wo_heating.hst")

    # Parameters
    h = parameters["H"]
    # horizon = parameters["HORIZON"]
    # n_steps = int(horizon / h)
    n_steps = parameters["t_amb"].shape[0]
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

    p_bat: DesignVariableInfo = {
        "name": "p_bat",
        "n_params": n_steps,
        "type": "c",
        "lower": -parameters["P_BAT_MAX"],
        "upper": parameters["P_BAT_MAX"],
        "initial_value": 0,
        "scale": 1 / parameters["P_BAT_MAX"],
    }
    opt.add_design_variables_info(p_bat)

    e_bat: DesignVariableInfo = {
        "name": "e_bat",
        "n_params": n_steps,
        "type": "c",
        "lower": parameters["E_BAT_MAX"] * parameters["SOC_MIN"],
        "upper": parameters["E_BAT_MAX"] * parameters["SOC_MAX"],
        "initial_value": parameters["E_BAT_MAX"] / 2,
        "scale": 1 / parameters["E_BAT_MAX"],
    }
    opt.add_design_variables_info(e_bat)

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
        "p_bat": np.ones(n_steps),
        "e_bat": np.ones(n_steps),
    }

    (
        dae1_jac,
        dae1_wrt,
        dae2_jac,
        dae2_wrt,
        q_required_jac,
        q_required_wrt,
        battery_soc_jac,
        battery_soc_wrt,
        battery_energy_jac,
        battery_energy_wrt,
        p_grid_jac,
        p_grid_wrt,
        t_tank_0_jac,
        t_tank_0_wrt,
        t_cond_0_jac,
        t_cond_0_wrt,
        e_bat_0_jac,
        e_bat_0_wrt,
        p_bat_0_jac,
        p_bat_0_wrt,
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
        "lower": -parameters["q_required_max"] * 1e-2,
        "upper": parameters["q_required_max"] * 1e-2,
        "function": q_required_constraint_fun,
        "scale": 1 / parameters["q_required_max"],
        "wrt": q_required_wrt,
        "jac": q_required_jac,
    }
    opt.add_constraint_info(q_required_constraint)

    battery_soc_constraint: ConstraintInfo = {
        "name": "battery_soc_constraint",
        "n_constraints": n_steps,
        "lower": parameters["SOC_MIN"],
        "upper": parameters["SOC_MAX"],
        "function": battery_soc_constraint_fun,
        "scale": 1,
        "wrt": battery_soc_wrt,
        "jac": battery_soc_jac,
    }
    opt.add_constraint_info(battery_soc_constraint)

    battery_energy_constraint: ConstraintInfo = {
        "name": "battery_energy_constraint",
        "n_constraints": n_steps - 1,
        "lower": 0,
        "upper": 0,
        "function": battery_energy_constraint_fun,
        "scale": 1 / parameters["E_BAT_MAX"],
        "wrt": battery_energy_wrt,
        "jac": battery_energy_jac,
    }
    opt.add_constraint_info(battery_energy_constraint)

    p_grid_constraint: ConstraintInfo = {
        "name": "p_grid_constraint",
        "n_constraints": n_steps,
        "lower": -parameters["P_GRID_MAX"],
        # "lower": 0,  # off-grid
        "upper": parameters["P_GRID_MAX"],
        "function": p_grid_constraint_fun,
        "scale": 1 / parameters["P_GRID_MAX"],
        "wrt": p_grid_wrt,
        "jac": p_grid_jac,
    }
    opt.add_constraint_info(p_grid_constraint)

    # Initial value constraints
    t_tank_0_constraint: ConstraintInfo = {
        "name": "t_tank_0_constraint",
        "n_constraints": 1,
        "lower": parameters["y0"]["t_tank"],
        "upper": parameters["y0"]["t_tank"],
        "function": lambda _, design_variables: design_variables["t_tank"][0],
        "scale": 1 / parameters["y0"]["t_tank"],
        "wrt": t_tank_0_wrt,
        "jac": t_tank_0_jac,
    }
    opt.add_constraint_info(t_tank_0_constraint)

    t_cond_0_constraint: ConstraintInfo = {
        "name": "t_cond_0_constraint",
        "n_constraints": 1,
        "lower": parameters["y0"]["t_cond"],
        "upper": parameters["y0"]["t_cond"],
        "function": lambda _, design_variables: design_variables["t_cond"][0],
        "scale": 1 / parameters["y0"]["t_cond"],
        "wrt": t_cond_0_wrt,
        "jac": t_cond_0_jac,
    }
    opt.add_constraint_info(t_cond_0_constraint)

    e_bat_0_constraint: ConstraintInfo = {
        "name": "e_bat_0_constraint",
        "n_constraints": 1,
        "lower": parameters["y0"]["e_bat"],
        "upper": parameters["y0"]["e_bat"],
        "function": lambda _, design_variables: design_variables["e_bat"][0],
        "scale": 1 / parameters["E_BAT_MAX"],
        "wrt": e_bat_0_wrt,
        "jac": e_bat_0_jac,
    }
    opt.add_constraint_info(e_bat_0_constraint)

    p_bat_0_constraint: ConstraintInfo = {
        "name": "p_bat_0_constraint",
        "n_constraints": 1,
        "lower": parameters["y0"]["p_bat"],
        "upper": parameters["y0"]["p_bat"],
        "function": lambda _, design_variables: design_variables["p_bat"][0],
        "scale": 1 / parameters["P_BAT_MAX"],
        "wrt": p_bat_0_wrt,
        "jac": p_bat_0_jac,
    }
    opt.add_constraint_info(p_bat_0_constraint)

    # Optimizer
    slsqpoptOptions = {"IPRINT": -1}
    ipoptOptions = {
        "print_level": 5,
        "max_iter": 100,
        # "tol": 1e-3,
        "obj_scaling_factor": 1e1,  # tells IPOPT how to internally handle the scaling without distorting the gradients
        # "nlp_scaling_method": "gradient-based",
        # "acceptable_tol": 1e-4,
        # "acceptable_obj_change_tol": 1e-4,
        "mu_strategy": "adaptive",
        # "alpha_red_factor": 0.2
        "alpha_for_y": "safer-min-dual-infeas"
        # "alpha_for_y": "primal-and-full"
        # "alpha_for_y": "dual-and-full"
        # "alpha_for_y": "full"
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
    t0 = PARAMS["T0"]

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon, year=2022)
    parameters = PARAMS
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["q_required_max"] = np.max(parameters["q_dot_required"])
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["pvpc_prices"] = dynamic_parameters["pvpc_prices"]
    parameters["excess_prices"] = dynamic_parameters["excess_prices"]

    y0 = {
        "t_tank": 306.65207722,
        "t_cond": 317.87997975,
        "e_bat": PARAMS["E_BAT_MAX"] * PARAMS["SOC_MAX"],
        "p_bat": 0,
    }
    parameters["y0"] = y0

    run_optimization(parameters, plot=True)
