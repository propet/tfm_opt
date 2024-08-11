import numpy as np
from pyoptsparse import History
from parameters import PARAMS
from opt import Opt
from utils import (
    get_dynamic_parameters,
    cop,
    get_dcopdT,
    get_solar_panels_depreciation_by_second,
    jax_to_numpy,
    get_battery_depreciation_by_joule,
    get_solar_panels_depreciation_by_second,
    get_tank_depreciation_by_second,
    get_hp_depreciation_by_joule,
    get_fixed_energy_cost_by_second,
    sparse_to_required_format,
)
from custom_types import DesignVariables, DesignVariableInfo, ConstraintInfo, Parameters
import scipy.sparse as sp
from control_with_adjoints.simulate import dae_forward, dae_adjoints, j_t_room_min
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


"""
Dimensiones de equipos fijas
Optimizacion del control para minimizar el coste en tarifa regulada
"""


def obj_fun(
    p_bat,
    p_compressor,
    e_bat_max,
    solar_size,
    p_compressor_max,
    p_grid_max,
    tank_volume,
    pvpc_prices,
    excess_prices,
    p_required,
    w_solar_per_w_installed,
    h,
):
    p_solar = w_solar_per_w_installed * solar_size
    p_grid = -p_solar + p_compressor + p_bat + p_required
    cost = (
        # Variable energy cost
        # ∘ buy at pvpc price, sell at excess price, but can't earn money at the end
        jnp.maximum(0, jnp.sum(h * jnp.maximum(pvpc_prices * p_grid, excess_prices * p_grid)))
        # ∘ Fixed energy cost
        + jnp.sum(h * get_fixed_energy_cost_by_second(p_grid_max))
        # ∘ depreciate battery by usage
        + jnp.sum(h * jnp.abs(p_bat) * get_battery_depreciation_by_joule(e_bat_max))
        # ∘ depreciate solar panels by time
        + jnp.sum(h * get_solar_panels_depreciation_by_second(solar_size))
        # ∘ depreciate heat pump by usage
        + jnp.sum(h * jnp.abs(p_compressor) * get_hp_depreciation_by_joule(p_compressor_max))
        # ∘ depreciate water tank by time
        + jnp.sum(h * get_tank_depreciation_by_second(tank_volume))
    )
    return cost


get_dobj_dp_bat = jax_to_numpy(jax.jit(jax.jacobian(obj_fun, argnums=0)))
get_dobj_dp_compressor = jax_to_numpy(jax.jit(jax.jacobian(obj_fun, argnums=1)))


def obj(opt, design_variables: DesignVariables) -> np.ndarray:
    # Design variables
    p_compressor = design_variables["p_compressor"]
    p_bat = design_variables["p_bat"]

    # Parameters
    parameters = opt.parameters
    h = np.ones(p_compressor.shape[0]) * parameters["H"]
    p_required = parameters["p_required"]
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]
    w_solar_per_w_installed = parameters["w_solar_per_w_installed"]
    e_bat_max = parameters["E_BAT_MAX"]
    solar_size = parameters["SOLAR_SIZE"]
    p_compressor_max = parameters["P_COMPRESSOR_MAX"]
    p_grid_max = parameters["P_GRID_MAX"]
    tank_volume = parameters["TANK_VOLUME"]

    objective = obj_fun(
        p_bat,
        p_compressor,
        e_bat_max,
        solar_size,
        p_compressor_max,
        p_grid_max,
        tank_volume,
        pvpc_prices,
        excess_prices,
        p_required,
        w_solar_per_w_installed,
        h,
    )
    return np.array(objective)


def obj_sens(opt, design_variables: DesignVariables):
    # Design variables
    p_compressor = design_variables["p_compressor"]
    p_bat = design_variables["p_bat"]

    # Parameters
    parameters = opt.parameters
    h = np.ones(p_compressor.shape[0]) * parameters["H"]
    p_required = parameters["p_required"]
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]
    w_solar_per_w_installed = parameters["w_solar_per_w_installed"]
    e_bat_max = parameters["E_BAT_MAX"]
    solar_size = parameters["SOLAR_SIZE"]
    p_compressor_max = parameters["P_COMPRESSOR_MAX"]
    p_grid_max = parameters["P_GRID_MAX"]
    tank_volume = parameters["TANK_VOLUME"]

    fun_inputs = (
        p_bat,
        p_compressor,
        e_bat_max,
        solar_size,
        p_compressor_max,
        p_grid_max,
        tank_volume,
        pvpc_prices,
        excess_prices,
        p_required,
        w_solar_per_w_installed,
        h,
    )
    dobj_dp_bat = get_dobj_dp_bat(*fun_inputs)
    dobj_dp_compressor = get_dobj_dp_compressor(*fun_inputs)

    obj_jac = {
        "p_bat": dobj_dp_bat,
        "p_compressor": dobj_dp_compressor,
    }
    obj_wrt = [
        "p_bat",
        "p_compressor",
    ]
    return (obj_jac, obj_wrt)


##############
# Constraints
##############
def battery_soc_fun(
    e_bat,
    e_bat_max,
):
    # SOC_MIN < e_bat / e_bat_max < SOC_MAX
    return e_bat / e_bat_max


get_dbattery_soc_de_bat = jax_to_numpy(jax.jit(jax.jacobian(battery_soc_fun, argnums=0)))


def battery_soc_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Design variables
    e_bat = design_variables["e_bat"]

    # Parameters
    parameters = opt.parameters
    e_bat_max = parameters["E_BAT_MAX"]

    return battery_soc_fun(e_bat, e_bat_max)


def battery_soc_constraint_sens(opt, design_variables: DesignVariables):
    # Design variables
    e_bat = design_variables["e_bat"]

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    e_bat_max = parameters["E_BAT_MAX"]

    dbattery_soc_de_bat = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        fun_inputs = (
            e_bat[i],
            e_bat_max,
        )
        dbattery_soc_de_bat[i, i] = get_dbattery_soc_de_bat(*fun_inputs) + 1e-20

    dbattery_soc_de_bat = sparse_to_required_format(dbattery_soc_de_bat.tocsr())

    battery_soc_jac = {
        "e_bat": dbattery_soc_de_bat,
    }
    battery_soc_wrt = [
        "e_bat",
    ]
    return (battery_soc_jac, battery_soc_wrt)


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


def battery_energy_constraint_sens(opt, design_variables: DesignVariables):
    # battery_energy_jac
    # e_bat_h - e_bat_{h-1} - bat_eta * p_bat_h * h = 0

    # Parameters
    parameters = opt.parameters
    h = parameters["H"]
    n_steps = parameters["n_steps"]
    bat_eta = parameters["BAT_ETA"]

    dbattery_energy_de_bat = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        dbattery_energy_de_bat[i - 1, i] = 1
        dbattery_energy_de_bat[i - 1, i - 1] = -1
    dbattery_energy_de_bat = dbattery_energy_de_bat.tocsr()
    dbattery_energy_de_bat = sparse_to_required_format(dbattery_energy_de_bat)

    dbattery_energy_dp_bat = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        dbattery_energy_dp_bat[i - 1, i] = -bat_eta * h
    dbattery_energy_dp_bat = dbattery_energy_dp_bat.tocsr()
    dbattery_energy_dp_bat = sparse_to_required_format(dbattery_energy_dp_bat)

    battery_energy_jac = {
        "e_bat": dbattery_energy_de_bat,
        "p_bat": dbattery_energy_dp_bat,
    }
    battery_energy_wrt = [
        "e_bat",
        "p_bat",
    ]
    return (battery_energy_jac, battery_energy_wrt)


def p_grid_fun(
    p_bat,
    p_compressor,
    solar_size,
    p_required,
    w_solar_per_w_installed,
):
    # p_grid = - p_solar + p_compressor + p_bat + p_required
    p_solar = w_solar_per_w_installed * solar_size
    p_grid = -p_solar + p_compressor + p_bat + p_required
    return p_grid


get_dp_grid_dp_bat = jax_to_numpy(jax.jit(jax.jacobian(p_grid_fun, argnums=0)))
get_dp_grid_dp_compressor = jax_to_numpy(jax.jit(jax.jacobian(p_grid_fun, argnums=1)))


def p_grid_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Design variables
    p_bat = design_variables["p_bat"]
    p_compressor = design_variables["p_compressor"]

    # Parameters
    parameters = opt.parameters
    p_required = parameters["p_required"]
    w_solar_per_w_installed = parameters["w_solar_per_w_installed"]
    solar_size = parameters["SOLAR_SIZE"]

    return p_grid_fun(
        p_bat,
        p_compressor,
        solar_size,
        p_required,
        w_solar_per_w_installed,
    )


def p_grid_constraint_sens(opt, design_variables: DesignVariables):
    # Design variables
    p_bat = design_variables["p_bat"]
    p_compressor = design_variables["p_compressor"]

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    p_required = parameters["p_required"]
    w_solar_per_w_installed = parameters["w_solar_per_w_installed"]
    solar_size = parameters["SOLAR_SIZE"]

    dp_grid_dp_bat = sp.lil_matrix((n_steps, n_steps))
    dp_grid_dp_compressor = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        fun_inputs = (
            p_bat[i],
            p_compressor[i],
            solar_size,
            p_required[i],
            w_solar_per_w_installed[i],
        )
        dp_grid_dp_bat[i, i] = get_dp_grid_dp_bat(*fun_inputs) + 1e-20
        dp_grid_dp_compressor[i, i] = get_dp_grid_dp_compressor(*fun_inputs) + 1e-20

    dp_grid_dp_bat = sparse_to_required_format(dp_grid_dp_bat.tocsr())
    dp_grid_dp_compressor = sparse_to_required_format(dp_grid_dp_compressor.tocsr())

    p_grid_jac = {
        "p_bat": dp_grid_dp_bat,
        "p_compressor": dp_grid_dp_compressor,
    }
    p_grid_wrt = [
        "p_bat",
        "p_compressor",
    ]
    return (p_grid_jac, p_grid_wrt)


def t_room_min_fun(
    y0_arr,
    m_dot_cond,
    m_dot_heating,
    p_compressor,
    t_amb,
    dae_p,
    h,
    n_steps,
):
    # t_target < t_room_min
    # where t_room_min is result of solving the thermal system DAE
    u = np.zeros((4, n_steps))
    u[0, :] = m_dot_cond
    u[1, :] = m_dot_heating
    u[2, :] = p_compressor
    u[3, :] = t_amb
    y = dae_forward(y0_arr, u, dae_p, h, n_steps)
    t_room_min = jnp.min(y[4])
    return np.array(t_room_min)


get_dt_room_min_dm_dot_cond = jax_to_numpy(jax.jit(jax.jacobian(t_room_min_fun, argnums=1)))
get_dt_room_min_dm_dot_heating = jax_to_numpy(jax.jit(jax.jacobian(t_room_min_fun, argnums=2)))
get_dt_room_min_dp_compressor = jax_to_numpy(jax.jit(jax.jacobian(t_room_min_fun, argnums=3)))


def t_room_min_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Design variables
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_heating = design_variables["m_dot_heating"]
    p_compressor = design_variables["p_compressor"]

    # Parameters
    parameters = opt.parameters
    h = parameters["H"]
    y0_arr = parameters["y0_arr"]
    t_amb = parameters["t_amb"]
    tank_volume = parameters["TANK_VOLUME"]
    n_steps = parameters["n_steps"]
    dae_p = np.array(
        [
            parameters["FLOOR_MASS"],
            parameters["CP_CONCRETE"],
            parameters["GRAVITY_ACCELERATION"],
            parameters["AIR_VOLUMETRIC_EXPANSION_COEFF"],
            parameters["FLOOR_WIDTH"],
            parameters["NU_AIR"],
            parameters["PR_AIR"],
            parameters["K_AIR"],
            parameters["TUBE_INNER_DIAMETER"],
            parameters["FLOOR_AREA"],
            parameters["STEFAN_BOLTZMANN_CONSTANT"],
            parameters["EPSILON_CONCRETE"],
            parameters["CP_WATER"],
            parameters["MU_WATER_AT_320K"],
            parameters["PR_WATER"],
            parameters["K_WATER"],
            parameters["K_PEX"],
            parameters["TUBE_THICKNESS"],
            parameters["A_TUBES"],
            parameters["ROOM_AIR_MASS"],
            parameters["CP_AIR"],
            parameters["A_WALLS"],
            parameters["A_ROOF"],
            parameters["WINDOWS_AREA"],
            parameters["U_WALLS"],
            parameters["U_ROOF"],
            parameters["WINDOWS_U"],
            tank_volume * parameters["RHO_WATER"],  # tank mass [kg]
            parameters["U_TANK"],
            6 * np.pi * (tank_volume / (2 * np.pi)) ** (2 / 3),  # tank area [m2]
        ]
    )
    return t_room_min_fun(
        y0_arr,
        m_dot_cond,
        m_dot_heating,
        p_compressor,
        t_amb,
        dae_p,
        h,
        n_steps,
    )


def t_room_min_constraint_sens(opt, design_variables: DesignVariables):
    # Design variables
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_heating = design_variables["m_dot_heating"]
    p_compressor = design_variables["p_compressor"]

    # Parameters
    parameters = opt.parameters
    h = parameters["H"]
    y0_arr = parameters["y0_arr"]
    t_amb = parameters["t_amb"]
    tank_volume = parameters["TANK_VOLUME"]
    n_steps = parameters["n_steps"]
    dae_p = np.array(
        [
            parameters["FLOOR_MASS"],
            parameters["CP_CONCRETE"],
            parameters["GRAVITY_ACCELERATION"],
            parameters["AIR_VOLUMETRIC_EXPANSION_COEFF"],
            parameters["FLOOR_WIDTH"],
            parameters["NU_AIR"],
            parameters["PR_AIR"],
            parameters["K_AIR"],
            parameters["TUBE_INNER_DIAMETER"],
            parameters["FLOOR_AREA"],
            parameters["STEFAN_BOLTZMANN_CONSTANT"],
            parameters["EPSILON_CONCRETE"],
            parameters["CP_WATER"],
            parameters["MU_WATER_AT_320K"],
            parameters["PR_WATER"],
            parameters["K_WATER"],
            parameters["K_PEX"],
            parameters["TUBE_THICKNESS"],
            parameters["A_TUBES"],
            parameters["ROOM_AIR_MASS"],
            parameters["CP_AIR"],
            parameters["A_WALLS"],
            parameters["A_ROOF"],
            parameters["WINDOWS_AREA"],
            parameters["U_WALLS"],
            parameters["U_ROOF"],
            parameters["WINDOWS_U"],
            tank_volume * parameters["RHO_WATER"],  # tank mass [kg]
            parameters["U_TANK"],
            6 * np.pi * (tank_volume / (2 * np.pi)) ** (2 / 3),  # tank area [m2]
        ]
    )

    dt_room_min_dm_dot_cond = sp.lil_matrix((1, n_steps))
    dt_room_min_dm_dot_heating = sp.lil_matrix((1, n_steps))
    dt_room_min_dp_compressor = sp.lil_matrix((1, n_steps))

    u = np.zeros((4, n_steps))
    u[0, :] = m_dot_cond
    u[1, :] = m_dot_heating
    u[2, :] = p_compressor
    u[3, :] = t_amb
    y = dae_forward(y0_arr, u, dae_p, h, n_steps)
    dj_dy0, dj_dp, dj_du = dae_adjoints(
        y,
        u,
        dae_p,
        h,
        n_steps,
        parameters,
        j_t_room_min,
    )

    for i in range(n_steps):
        dt_room_min_dm_dot_cond[0, i] = dj_du[0, i] + 1e-20
        dt_room_min_dm_dot_heating[0, i] = dj_du[1, i] + 1e-20
        dt_room_min_dp_compressor[0, i] = dj_du[2, i] + 1e-20

    dt_room_min_dm_dot_cond = sparse_to_required_format(dt_room_min_dm_dot_cond.tocsr())
    dt_room_min_dm_dot_heating = sparse_to_required_format(dt_room_min_dm_dot_heating.tocsr())
    dt_room_min_dp_compressor = sparse_to_required_format(dt_room_min_dp_compressor.tocsr())

    t_room_min_jac = {
        "m_dot_cond": dt_room_min_dm_dot_cond,
        "m_dot_heating": dt_room_min_dm_dot_heating,
        "p_compressor": dt_room_min_dp_compressor,
    }
    t_room_min_wrt = [
        "m_dot_cond",
        "m_dot_heating",
        "p_compressor",
    ]
    return (t_room_min_jac, t_room_min_wrt)


def e_bat_0_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]

    de_bat_0_de_bat = sp.lil_matrix((1, n_steps))
    de_bat_0_de_bat[0, 0] = 1
    de_bat_0_de_bat = de_bat_0_de_bat.tocsr()
    de_bat_0_de_bat = sparse_to_required_format(de_bat_0_de_bat)
    e_bat_0_jac = {"e_bat": de_bat_0_de_bat}
    e_bat_0_wrt = ["e_bat"]
    return (e_bat_0_jac, e_bat_0_wrt)


def p_bat_0_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]

    dp_bat_0_dp_bat = sp.lil_matrix((1, n_steps))
    dp_bat_0_dp_bat[0, 0] = 1
    dp_bat_0_dp_bat = dp_bat_0_dp_bat.tocsr()
    dp_bat_0_dp_bat = sparse_to_required_format(dp_bat_0_dp_bat)
    p_bat_0_jac = {"p_bat": dp_bat_0_dp_bat}
    p_bat_0_wrt = ["p_bat"]
    return (p_bat_0_jac, p_bat_0_wrt)


def sens(opt, design_variables: DesignVariables, func_values):
    (obj_jac, obj_wrt) = obj_sens(opt, design_variables)
    (battery_soc_jac, battery_soc_wrt) = battery_soc_constraint_sens(opt, design_variables)
    (battery_energy_jac, battery_energy_wrt) = battery_energy_constraint_sens(opt, design_variables)
    (p_grid_jac, p_grid_wrt) = p_grid_constraint_sens(opt, design_variables)
    (t_room_min_jac, t_room_min_wrt) = t_room_min_constraint_sens(opt, design_variables)
    (e_bat_0_jac, e_bat_0_wrt) = e_bat_0_constraint_sens(opt, design_variables)
    (p_bat_0_jac, p_bat_0_wrt) = p_bat_0_constraint_sens(opt, design_variables)

    return {
        "obj": obj_jac,
        "battery_soc_constraint": battery_soc_jac,
        "battery_energy_constraint": battery_energy_jac,
        "p_grid_constraint": p_grid_jac,
        "t_room_min_constraint": t_room_min_jac,
        # "t_out_heating_0_constraint": t_out_heating_0_jac,
        "e_bat_0_constraint": e_bat_0_jac,
        "p_bat_0_constraint": p_bat_0_jac,
    }


def run_optimization(parameters, plot=True):
    optName = "control_adjoints_regulated"
    historyFileName = f"saves/{optName}.hst"
    opt = Opt(optName, obj, historyFileName=historyFileName)

    # If you wanna run again with the last found
    # design variables as initial values
    dvs_from_last = False
    history = None
    if dvs_from_last:
        history = History(historyFileName)
        history = history.getValues()

    # Parameters
    n_steps = parameters["t_amb"].shape[0]
    print("n_steps: ", n_steps)
    parameters["n_steps"] = n_steps
    opt.add_parameters(parameters)

    # Controls
    # m_dot_cond    = u[0]
    # m_dot_heating = u[1]
    # p_compressor  = u[2]
    m_dot_cond: DesignVariableInfo = {
        "name": "m_dot_cond",
        "n_params": n_steps,
        "type": "c",
        "lower": 1e-3,
        "upper": parameters["M_DOT_COND_MAX"],
        "initial_value": history["m_dot_cond"][-1] if history else parameters["M_DOT_COND_MAX"] / 2,
        "scale": 1 / parameters["M_DOT_COND_MAX"],
    }
    opt.add_design_variables_info(m_dot_cond)

    m_dot_heating: DesignVariableInfo = {
        "name": "m_dot_heating",
        "n_params": n_steps,
        "type": "c",
        "lower": 1e-3,
        "upper": parameters["M_DOT_HEATING_MAX"],
        "initial_value": history["m_dot_heating"][-1] if history else parameters["M_DOT_HEATING_MAX"] / 2,
        "scale": 1 / parameters["M_DOT_HEATING_MAX"],
    }
    opt.add_design_variables_info(m_dot_heating)

    p_compressor: DesignVariableInfo = {
        "name": "p_compressor",
        "n_params": n_steps,
        "type": "c",
        "lower": 0,
        "upper": parameters["P_COMPRESSOR_MAX"],
        "initial_value": history["p_compressor"][-1] if history else 0,
        "scale": 1 / parameters["P_COMPRESSOR_MAX"],
    }
    opt.add_design_variables_info(p_compressor)

    # Electrical state
    p_bat: DesignVariableInfo = {
        "name": "p_bat",
        "n_params": n_steps,
        "type": "c",
        "lower": -parameters["P_BAT_MAX"],
        "upper": parameters["P_BAT_MAX"],
        "initial_value": history["p_bat"][-1] if history else 0,
        "scale": 1 / parameters["P_BAT_MAX"],
    }
    opt.add_design_variables_info(p_bat)

    e_bat: DesignVariableInfo = {
        "name": "e_bat",
        "n_params": n_steps,
        "type": "c",
        "lower": parameters["SOC_MIN"] * parameters["E_BAT_MAX"],
        "upper": parameters["SOC_MAX"] * parameters["E_BAT_MAX"],
        "initial_value": history["e_bat"][-1] if history else parameters["E_BAT_MAX"],
        "scale": 1 / parameters["E_BAT_MAX_LIMIT"],
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
        "m_dot_heating": np.ones(n_steps),
        "p_bat": np.ones(n_steps),
        "e_bat": np.ones(n_steps),
    }

    (battery_soc_jac, battery_soc_wrt) = battery_soc_constraint_sens(opt, dummy_design_variables)
    (battery_energy_jac, battery_energy_wrt) = battery_energy_constraint_sens(opt, dummy_design_variables)
    (p_grid_jac, p_grid_wrt) = p_grid_constraint_sens(opt, dummy_design_variables)
    (t_room_min_jac, t_room_min_wrt) = t_room_min_constraint_sens(opt, dummy_design_variables)
    (e_bat_0_jac, e_bat_0_wrt) = e_bat_0_constraint_sens(opt, dummy_design_variables)
    (p_bat_0_jac, p_bat_0_wrt) = p_bat_0_constraint_sens(opt, dummy_design_variables)

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
        "scale": 1 / parameters["E_BAT_MAX_LIMIT"],
        "wrt": battery_energy_wrt,
        "jac": battery_energy_jac,
    }
    opt.add_constraint_info(battery_energy_constraint)

    p_grid_constraint: ConstraintInfo = {
        "name": "p_grid_constraint",
        "n_constraints": n_steps,
        "lower": -parameters["P_GRID_MAX"],
        "upper": parameters["P_GRID_MAX"],
        "function": p_grid_constraint_fun,
        "scale": 1 / parameters["P_GRID_MAX"],
        "wrt": p_grid_wrt,
        "jac": p_grid_jac,
    }
    opt.add_constraint_info(p_grid_constraint)

    t_room_min_constraint: ConstraintInfo = {
        "name": "t_room_min_constraint",
        "n_constraints": 1,
        "lower": parameters["T_TARGET"],
        "upper": None,
        "function": t_room_min_constraint_fun,
        "scale": 1 / 300,
        "wrt": t_room_min_wrt,
        "jac": t_room_min_jac,
    }
    opt.add_constraint_info(t_room_min_constraint)

    # Initial value constraints
    p_bat_0_constraint: ConstraintInfo = {
        "name": "p_bat_0_constraint",
        "n_constraints": 1,
        "lower": parameters["y0"]["p_bat"],
        "upper": parameters["y0"]["p_bat"],
        "function": lambda _, design_variables: design_variables["p_bat"][0],
        "scale": 1 / parameters["y0"]["p_bat"],
        "wrt": p_bat_0_wrt,
        "jac": p_bat_0_jac,
    }
    opt.add_constraint_info(p_bat_0_constraint)

    e_bat_0_constraint: ConstraintInfo = {
        "name": "e_bat_0_constraint",
        "n_constraints": 1,
        "lower": parameters["y0"]["e_bat"],
        "upper": parameters["y0"]["e_bat"],
        "function": lambda _, design_variables: design_variables["e_bat"][0],
        "scale": 1 / parameters["y0"]["e_bat"],
        "wrt": e_bat_0_wrt,
        "jac": e_bat_0_jac,
    }
    opt.add_constraint_info(e_bat_0_constraint)

    # Optimizer
    ipoptOptions = {
        "print_level": 5,  # up to 12
        "max_iter": 400,
        # "obj_scaling_factor": 1e-1,
        # "mu_strategy": "adaptive",
        # "alpha_for_y": "safer-min-dual-infeas",
        "mumps_mem_percent": 4000,
    }
    opt.add_optimizer("ipopt", ipoptOptions)

    # Setup and check optimization problem
    opt.setup()
    # opt.print()
    # opt.optProb.printSparsity()
    # exit(0)

    # Run
    sol = opt.optimize(sens=sens)

    # Check Solution
    if plot:
        pass
        # print(sol)

    return sol.xStar, sol.fStar


if __name__ == "__main__":
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    t0 = PARAMS["T0"]

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon, year=2022)
    parameters = PARAMS
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["pvpc_prices"] = dynamic_parameters["pvpc_prices"]
    parameters["excess_prices"] = dynamic_parameters["excess_prices"]

    y0 = {
        "t_cond": 308.7801,
        "t_tank": 307.646,
        "t_out_heating": 304.54,
        "t_floor": 295,
        "t_room": 293.79,
        "e_bat": parameters["SOC_MIN"] * parameters["E_BAT_MAX"] + 10000,
        "p_bat": 1e-2,
    }
    y0_arr = np.array(list(y0.values())[:5])  # get only state temperatures for dae
    parameters["y0"] = y0
    parameters["y0_arr"] = y0_arr

    run_optimization(parameters, plot=True)
