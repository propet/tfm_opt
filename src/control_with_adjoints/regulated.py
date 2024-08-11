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
from full_system.simulate_full import get_h_floor_air, get_h_tube_water
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
def dae1_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # ∘ cop(t_cond) * p_compressor - m_dot_cond * cp_water * (t_cond - t_tank) = 0

    # Parameters
    parameters = opt.parameters
    cp_water = parameters["CP_WATER"]

    # Design variables
    p_compressor = design_variables["p_compressor"]
    t_tank = design_variables["t_tank"]
    t_cond = design_variables["t_cond"]
    m_dot_cond = design_variables["m_dot_cond"]

    dae1 = cop(t_cond) * p_compressor - m_dot_cond * cp_water * (t_cond - t_tank)
    return dae1


def dae1_constraint_sens(opt, design_variables: DesignVariables):
    # ∘ cop(t_cond) * p_compressor - m_dot_cond * cp_water * (t_cond - t_tank) = 0

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    cp_water = parameters["CP_WATER"]

    # Design variables
    p_compressor = design_variables["p_compressor"]
    t_tank = design_variables["t_tank"]
    t_cond = design_variables["t_cond"]
    m_dot_cond = design_variables["m_dot_cond"]

    ddae1_dt_tank = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        ddae1_dt_tank[i, i] = m_dot_cond[i] * cp_water + 1e-20
    ddae1_dt_tank = ddae1_dt_tank.tocsr()
    ddae1_dt_tank = sparse_to_required_format(ddae1_dt_tank)

    ddae1_dt_cond = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        ddae1_dt_cond[i, i] = get_dcopdT(t_cond[i]) * p_compressor[i] - m_dot_cond[i] * cp_water + 1e-20
    ddae1_dt_cond = ddae1_dt_cond.tocsr()
    ddae1_dt_cond = sparse_to_required_format(ddae1_dt_cond)

    ddae1_dp_compressor = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        ddae1_dp_compressor[i, i] = cop(t_cond[i]) + 1e-20
    ddae1_dp_compressor = ddae1_dp_compressor.tocsr()
    ddae1_dp_compressor = sparse_to_required_format(ddae1_dp_compressor)

    ddae1_dm_dot_cond = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        ddae1_dm_dot_cond[i, i] = -cp_water * (t_cond[i] - t_tank[i]) + 1e-20
    ddae1_dm_dot_cond = ddae1_dm_dot_cond.tocsr()
    ddae1_dm_dot_cond = sparse_to_required_format(ddae1_dm_dot_cond)

    dae1_jac = {
        "t_tank": ddae1_dt_tank,
        "t_cond": ddae1_dt_cond,
        "p_compressor": ddae1_dp_compressor,
        "m_dot_cond": ddae1_dm_dot_cond,
    }
    dae1_wrt = [
        "t_cond",
        "t_tank",
        "p_compressor",
        "m_dot_cond",
    ]
    return (dae1_jac, dae1_wrt)


def dae2_fun(
    t_cond,
    t_tank,
    t_tank_prev,
    t_out_heating,
    m_dot_cond,
    m_dot_heating,
    tank_volume,
    cp_water,
    rho_water,
    h,
    t_amb,
    U_tank,
):
    # ∘ m_tank * cp_water * ((t_tank - t_tank_prev)/h)
    #     - m_dot_cond * cp_water * t_cond
    #     - m_dot_heating * cp_water * t_out_heating
    #     + (m_dot_cond + m_dot_heating) * cp_water * t_tank
    #     + U_tank * A_tank * (t_tank - t_amb)
    #     = 0

    A_tank = 6 * np.pi * (tank_volume / (2 * jnp.pi)) ** (2 / 3)  # tank surface area (m2)
    m_tank = tank_volume * rho_water

    return (
        m_tank * cp_water * ((t_tank - t_tank_prev) / h)
        - m_dot_cond * cp_water * t_cond
        - m_dot_heating * cp_water * t_out_heating
        + (m_dot_cond + m_dot_heating) * cp_water * t_tank
        + U_tank * A_tank * (t_tank - t_amb)
    )


get_ddae2_dt_cond = jax_to_numpy(jax.jit(jax.jacobian(dae2_fun, argnums=0)))
get_ddae2_dt_tank = jax_to_numpy(jax.jit(jax.jacobian(dae2_fun, argnums=1)))
get_ddae2_dt_tank_prev = jax_to_numpy(jax.jit(jax.jacobian(dae2_fun, argnums=2)))
get_ddae2_dt_out_heating = jax_to_numpy(jax.jit(jax.jacobian(dae2_fun, argnums=3)))
get_ddae2_dm_dot_cond = jax_to_numpy(jax.jit(jax.jacobian(dae2_fun, argnums=4)))
get_ddae2_dm_dot_heating = jax_to_numpy(jax.jit(jax.jacobian(dae2_fun, argnums=5)))


def dae2_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Design variables
    t_tank = design_variables["t_tank"]
    t_out_heating = design_variables["t_out_heating"]
    t_cond = design_variables["t_cond"]
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_heating = design_variables["m_dot_heating"]

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    cp_water = parameters["CP_WATER"]
    rho_water = parameters["RHO_WATER"]
    h = parameters["H"]
    t_amb = parameters["t_amb"]
    U_tank = parameters["U_TANK"]
    tank_volume = parameters["TANK_VOLUME"]

    dae2 = []
    for i in range(1, n_steps):
        dae2.append(
            dae2_fun(
                t_cond[i],
                t_tank[i],
                t_tank[i - 1],
                t_out_heating[i],
                m_dot_cond[i],
                m_dot_heating[i],
                tank_volume,
                cp_water,
                rho_water,
                h,
                t_amb[i],
                U_tank,
            )
        )
    return np.array(dae2)


def dae2_constraint_sens(opt, design_variables: DesignVariables):
    # Design variables
    t_tank = design_variables["t_tank"]
    t_out_heating = design_variables["t_out_heating"]
    t_cond = design_variables["t_cond"]
    m_dot_cond = design_variables["m_dot_cond"]
    m_dot_heating = design_variables["m_dot_heating"]

    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    cp_water = parameters["CP_WATER"]
    rho_water = parameters["RHO_WATER"]
    h = parameters["H"]
    t_amb = parameters["t_amb"]
    U_tank = parameters["U_TANK"]
    tank_volume = parameters["TANK_VOLUME"]

    ddae2_dt_cond = sp.lil_matrix((n_steps - 1, n_steps))
    ddae2_dt_tank = sp.lil_matrix((n_steps - 1, n_steps))
    ddae2_dt_out_heating = sp.lil_matrix((n_steps - 1, n_steps))
    ddae2_dm_dot_cond = sp.lil_matrix((n_steps - 1, n_steps))
    ddae2_dm_dot_heating = sp.lil_matrix((n_steps - 1, n_steps))
    ddae2_dtank_volume = sp.lil_matrix((n_steps - 1, 1))
    for i in range(1, n_steps):
        fun_inputs = (
            t_cond[i],
            t_tank[i],
            t_tank[i - 1],
            t_out_heating[i],
            m_dot_cond[i],
            m_dot_heating[i],
            tank_volume,
            cp_water,
            rho_water,
            h,
            t_amb[i],
            U_tank,
        )
        ddae2_dt_cond[i - 1, i] = get_ddae2_dt_cond(*fun_inputs) + 1e-20
        ddae2_dt_tank[i - 1, i] = get_ddae2_dt_tank(*fun_inputs) + 1e-20
        ddae2_dt_tank[i - 1, i - 1] = get_ddae2_dt_tank_prev(*fun_inputs) + 1e-20
        ddae2_dt_out_heating[i - 1, i] = get_ddae2_dt_out_heating(*fun_inputs) + 1e-20
        ddae2_dm_dot_cond[i - 1, i] = get_ddae2_dm_dot_cond(*fun_inputs) + 1e-20
        ddae2_dm_dot_heating[i - 1, i] = get_ddae2_dm_dot_heating(*fun_inputs) + 1e-20

    ddae2_dt_cond = sparse_to_required_format(ddae2_dt_cond.tocsr())
    ddae2_dt_tank = sparse_to_required_format(ddae2_dt_tank.tocsr())
    ddae2_dt_out_heating = sparse_to_required_format(ddae2_dt_out_heating.tocsr())
    ddae2_dm_dot_cond = sparse_to_required_format(ddae2_dm_dot_cond.tocsr())
    ddae2_dm_dot_heating = sparse_to_required_format(ddae2_dm_dot_heating.tocsr())

    dae2_jac = {
        "t_cond": ddae2_dt_cond,
        "t_tank": ddae2_dt_tank,
        "t_out_heating": ddae2_dt_out_heating,
        "m_dot_cond": ddae2_dm_dot_cond,
        "m_dot_heating": ddae2_dm_dot_heating,
    }
    dae2_wrt = [
        "t_cond",
        "t_tank",
        "t_out_heating",
        "m_dot_cond",
        "m_dot_heating",
    ]
    return (dae2_jac, dae2_wrt)


def dae3_fun(
    t_tank,
    t_out_heating,
    t_floor,
    m_dot_heating,
    cp_water,
    A_tubes,
    tube_inner_diameter,
    mu_water_at_320K,
    Pr_water,
    k_water,
    k_pex,
    tube_thickness,
):
    # ∘ m_dot_heating * cp_water * (t_tank - t_out_heating) - U_tubes * A_tubes * DeltaT_tubes = 0
    h_tube_water = get_h_tube_water(
        tube_inner_diameter,
        mu_water_at_320K,
        Pr_water,
        k_water,
        m_dot_heating,
    )
    U_tubes = 1 / ((1 / h_tube_water) + (1 / (k_pex / tube_thickness)))

    # Mean deltaT with absolute differences
    DeltaT_tubes = ((t_tank - t_floor) + jnp.abs(t_out_heating - t_floor)) / 2

    dae3 = m_dot_heating * cp_water * (t_tank - t_out_heating) - U_tubes * A_tubes * DeltaT_tubes
    return dae3


get_ddae3_dt_tank = jax_to_numpy(jax.jit(jax.jacobian(dae3_fun, argnums=0)))
get_ddae3_dt_out_heating = jax_to_numpy(jax.jit(jax.jacobian(dae3_fun, argnums=1)))
get_ddae3_dt_floor = jax_to_numpy(jax.jit(jax.jacobian(dae3_fun, argnums=2)))
get_ddae3_dm_dot_heating = jax_to_numpy(jax.jit(jax.jacobian(dae3_fun, argnums=3)))


def dae3_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Parameters
    parameters = opt.parameters
    cp_water = parameters["CP_WATER"]
    A_tubes = parameters["A_TUBES"]
    tube_inner_diameter = parameters["TUBE_INNER_DIAMETER"]
    mu_water_at_320K = parameters["MU_WATER_AT_320K"]
    Pr_water = parameters["PR_WATER"]
    k_water = parameters["K_WATER"]
    k_pex = parameters["K_PEX"]
    tube_thickness = parameters["TUBE_THICKNESS"]

    # Design variables
    t_tank = design_variables["t_tank"]
    t_floor = design_variables["t_floor"]
    t_out_heating = design_variables["t_out_heating"]
    m_dot_heating = design_variables["m_dot_heating"]

    dae3 = dae3_fun(
        t_tank,
        t_out_heating,
        t_floor,
        m_dot_heating,
        cp_water,
        A_tubes,
        tube_inner_diameter,
        mu_water_at_320K,
        Pr_water,
        k_water,
        k_pex,
        tube_thickness,
    )
    return dae3


def dae3_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    cp_water = parameters["CP_WATER"]
    A_tubes = parameters["A_TUBES"]
    tube_inner_diameter = parameters["TUBE_INNER_DIAMETER"]
    mu_water_at_320K = parameters["MU_WATER_AT_320K"]
    Pr_water = parameters["PR_WATER"]
    k_water = parameters["K_WATER"]
    k_pex = parameters["K_PEX"]
    tube_thickness = parameters["TUBE_THICKNESS"]

    # Design variables
    t_tank = design_variables["t_tank"]
    t_floor = design_variables["t_floor"]
    t_out_heating = design_variables["t_out_heating"]
    m_dot_heating = design_variables["m_dot_heating"]

    ddae3_dt_tank = sp.lil_matrix((n_steps, n_steps))
    ddae3_dt_out_heating = sp.lil_matrix((n_steps, n_steps))
    ddae3_dt_floor = sp.lil_matrix((n_steps, n_steps))
    ddae3_dm_dot_heating = sp.lil_matrix((n_steps, n_steps))
    for i in range(n_steps):
        fun_inputs = (
            t_tank[i],
            t_out_heating[i],
            t_floor[i],
            m_dot_heating[i],
            cp_water,
            A_tubes,
            tube_inner_diameter,
            mu_water_at_320K,
            Pr_water,
            k_water,
            k_pex,
            tube_thickness,
        )
        ddae3_dt_tank[i, i] = get_ddae3_dt_tank(*fun_inputs) + 1e-20
        ddae3_dt_out_heating[i, i] = get_ddae3_dt_out_heating(*fun_inputs) + 1e-20
        ddae3_dt_floor[i, i] = get_ddae3_dt_floor(*fun_inputs) + 1e-20
        ddae3_dm_dot_heating[i, i] = get_ddae3_dm_dot_heating(*fun_inputs) + 1e-20

    ddae3_dt_tank = sparse_to_required_format(ddae3_dt_tank.tocsr())
    ddae3_dt_out_heating = sparse_to_required_format(ddae3_dt_out_heating.tocsr())
    ddae3_dt_floor = sparse_to_required_format(ddae3_dt_floor.tocsr())
    ddae3_dm_dot_heating = sparse_to_required_format(ddae3_dm_dot_heating.tocsr())

    dae3_jac = {
        "t_tank": ddae3_dt_tank,
        "t_out_heating": ddae3_dt_out_heating,
        "t_floor": ddae3_dt_floor,
        "m_dot_heating": ddae3_dm_dot_heating,
    }
    dae3_wrt = [
        "t_tank",
        "t_out_heating",
        "t_floor",
        "m_dot_heating",
    ]
    return (dae3_jac, dae3_wrt)


def dae4_fun(
    t_tank,
    t_out_heating,
    t_floor,
    t_floor_prev,
    t_room,
    m_dot_heating,
    cp_water,
    h,
    floor_mass,
    cp_concrete,
    floor_area,
    stefan_boltzmann_constant,
    epsilon_concrete,
    gravity_acceleration,
    air_volumetric_expansion_coeff,
    floor_width,
    nu_air,
    Pr_air,
    k_air,
    A_roof,
):
    # ∘ floor_mass * cp_concrete * ((t_floor - t_floor_prev)/h)
    #     - m_dot_heating * cp_water * (t_tank - t_out_heating)
    #     + h_floor_air * floor_area * (t_floor - t_room)
    #     + stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
    #     = 0
    h_floor_air = get_h_floor_air(
        t_floor,
        t_room,
        gravity_acceleration,
        air_volumetric_expansion_coeff,
        floor_width,
        nu_air,
        Pr_air,
        k_air,
        A_roof,
    )
    dae4 = (
        floor_mass * cp_concrete * ((t_floor - t_floor_prev) / h)
        - m_dot_heating * cp_water * (t_tank - t_out_heating)
        + h_floor_air * floor_area * (t_floor - t_room)
        + stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
    )
    return dae4


get_ddae4_dt_tank = jax_to_numpy(jax.jit(jax.jacobian(dae4_fun, argnums=0)))
get_ddae4_dt_out_heating = jax_to_numpy(jax.jit(jax.jacobian(dae4_fun, argnums=1)))
get_ddae4_dt_floor = jax_to_numpy(jax.jit(jax.jacobian(dae4_fun, argnums=2)))
get_ddae4_dt_floor_prev = jax_to_numpy(jax.jit(jax.jacobian(dae4_fun, argnums=3)))
get_ddae4_dt_room = jax_to_numpy(jax.jit(jax.jacobian(dae4_fun, argnums=4)))
get_ddae4_dm_dot_heating = jax_to_numpy(jax.jit(jax.jacobian(dae4_fun, argnums=5)))


def dae4_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    cp_water = parameters["CP_WATER"]
    h = parameters["H"]
    floor_mass = parameters["FLOOR_MASS"]
    cp_concrete = parameters["CP_CONCRETE"]
    floor_area = parameters["FLOOR_AREA"]
    stefan_boltzmann_constant = parameters["STEFAN_BOLTZMANN_CONSTANT"]
    epsilon_concrete = parameters["EPSILON_CONCRETE"]
    gravity_acceleration = parameters["GRAVITY_ACCELERATION"]
    air_volumetric_expansion_coeff = parameters["AIR_VOLUMETRIC_EXPANSION_COEFF"]
    floor_width = parameters["FLOOR_WIDTH"]
    nu_air = parameters["NU_AIR"]
    Pr_air = parameters["PR_AIR"]
    k_air = parameters["K_AIR"]
    A_roof = parameters["A_ROOF"]

    # Design variables
    t_tank = design_variables["t_tank"]
    t_floor = design_variables["t_floor"]
    t_room = design_variables["t_room"]
    t_out_heating = design_variables["t_out_heating"]
    m_dot_heating = design_variables["m_dot_heating"]

    dae4 = []
    for i in range(1, n_steps):
        dae4.append(
            dae4_fun(
                t_tank[i],
                t_out_heating[i],
                t_floor[i],
                t_floor[i - 1],
                t_room[i],
                m_dot_heating[i],
                cp_water,
                h,
                floor_mass,
                cp_concrete,
                floor_area,
                stefan_boltzmann_constant,
                epsilon_concrete,
                gravity_acceleration,
                air_volumetric_expansion_coeff,
                floor_width,
                nu_air,
                Pr_air,
                k_air,
                A_roof,
            )
        )
    return np.array(dae4)


def dae4_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    cp_water = parameters["CP_WATER"]
    h = parameters["H"]
    floor_mass = parameters["FLOOR_MASS"]
    cp_concrete = parameters["CP_CONCRETE"]
    floor_area = parameters["FLOOR_AREA"]
    stefan_boltzmann_constant = parameters["STEFAN_BOLTZMANN_CONSTANT"]
    epsilon_concrete = parameters["EPSILON_CONCRETE"]
    gravity_acceleration = parameters["GRAVITY_ACCELERATION"]
    air_volumetric_expansion_coeff = parameters["AIR_VOLUMETRIC_EXPANSION_COEFF"]
    floor_width = parameters["FLOOR_WIDTH"]
    nu_air = parameters["NU_AIR"]
    Pr_air = parameters["PR_AIR"]
    k_air = parameters["K_AIR"]
    A_roof = parameters["A_ROOF"]

    # Design variables
    t_tank = design_variables["t_tank"]
    t_floor = design_variables["t_floor"]
    t_room = design_variables["t_room"]
    t_out_heating = design_variables["t_out_heating"]
    m_dot_heating = design_variables["m_dot_heating"]

    ddae4_dt_tank = sp.lil_matrix((n_steps - 1, n_steps))
    ddae4_dt_out_heating = sp.lil_matrix((n_steps - 1, n_steps))
    ddae4_dt_floor = sp.lil_matrix((n_steps - 1, n_steps))
    ddae4_dt_room = sp.lil_matrix((n_steps - 1, n_steps))
    ddae4_dm_dot_heating = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        fun_inputs = (
            t_tank[i],
            t_out_heating[i],
            t_floor[i],
            t_floor[i - 1],
            t_room[i],
            m_dot_heating[i],
            cp_water,
            h,
            floor_mass,
            cp_concrete,
            floor_area,
            stefan_boltzmann_constant,
            epsilon_concrete,
            gravity_acceleration,
            air_volumetric_expansion_coeff,
            floor_width,
            nu_air,
            Pr_air,
            k_air,
            A_roof,
        )
        ddae4_dt_tank[i - 1, i] = get_ddae4_dt_tank(*fun_inputs) + 1e-20
        ddae4_dt_out_heating[i - 1, i] = get_ddae4_dt_out_heating(*fun_inputs) + 1e-20
        ddae4_dt_floor[i - 1, i] = get_ddae4_dt_floor(*fun_inputs) + 1e-20
        ddae4_dt_floor[i - 1, i - 1] = get_ddae4_dt_floor_prev(*fun_inputs) + 1e-20
        ddae4_dt_room[i - 1, i] = get_ddae4_dt_room(*fun_inputs) + 1e-20
        ddae4_dm_dot_heating[i - 1, i] = get_ddae4_dm_dot_heating(*fun_inputs) + 1e-20

    ddae4_dt_tank = sparse_to_required_format(ddae4_dt_tank.tocsr())
    ddae4_dt_out_heating = sparse_to_required_format(ddae4_dt_out_heating.tocsr())
    ddae4_dt_floor = sparse_to_required_format(ddae4_dt_floor.tocsr())
    ddae4_dt_room = sparse_to_required_format(ddae4_dt_room.tocsr())
    ddae4_dm_dot_heating = sparse_to_required_format(ddae4_dm_dot_heating.tocsr())

    dae4_jac = {
        "t_tank": ddae4_dt_tank,
        "t_out_heating": ddae4_dt_out_heating,
        "t_floor": ddae4_dt_floor,
        "t_room": ddae4_dt_room,
        "m_dot_heating": ddae4_dm_dot_heating,
    }
    dae4_wrt = [
        "t_tank",
        "t_out_heating",
        "t_floor",
        "t_room",
        "m_dot_heating",
    ]
    return (dae4_jac, dae4_wrt)


def dae5_fun(
    t_floor,
    t_room,
    t_room_prev,
    t_amb,
    room_air_mass,
    cp_air,
    h,
    floor_area,
    stefan_boltzmann_constant,
    epsilon_concrete,
    gravity_acceleration,
    air_volumetric_expansion_coeff,
    floor_width,
    nu_air,
    Pr_air,
    k_air,
    A_roof,
    A_windows,
    A_walls,
    U_walls,
    U_roof,
    U_windows,
):
    # ∘ room_air_mass * cp_air * ((t_room - t_room_prev)/h)
    #     - h_floor_air * floor_area * (t_floor - t_room)
    #     - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
    #     + U_walls * A_walls * (t_room - t_amb)
    #     + U_roof * A_roof * (t_room - t_amb)
    #     + U_windows * A_windows * (t_room - t_amb)
    #     = 0
    h_floor_air = get_h_floor_air(
        t_floor,
        t_room,
        gravity_acceleration,
        air_volumetric_expansion_coeff,
        floor_width,
        nu_air,
        Pr_air,
        k_air,
        A_roof,
    )
    dae4 = (
        room_air_mass * cp_air * ((t_room - t_room_prev) / h)
        - h_floor_air * floor_area * (t_floor - t_room)
        - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
        + U_walls * A_walls * (t_room - t_amb)
        + U_roof * A_roof * (t_room - t_amb)
        + U_windows * A_windows * (t_room - t_amb)
    )
    return dae4


get_ddae5_dt_floor = jax_to_numpy(jax.jit(jax.jacobian(dae5_fun, argnums=0)))
get_ddae5_dt_room = jax_to_numpy(jax.jit(jax.jacobian(dae5_fun, argnums=1)))
get_ddae5_dt_room_prev = jax_to_numpy(jax.jit(jax.jacobian(dae5_fun, argnums=2)))


def dae5_constraint_fun(opt, design_variables: DesignVariables) -> np.ndarray:
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    t_amb = parameters["t_amb"]
    room_air_mass = parameters["ROOM_AIR_MASS"]
    cp_air = parameters["CP_AIR"]
    h = parameters["H"]
    floor_area = parameters["FLOOR_AREA"]
    stefan_boltzmann_constant = parameters["STEFAN_BOLTZMANN_CONSTANT"]
    epsilon_concrete = parameters["EPSILON_CONCRETE"]
    gravity_acceleration = parameters["GRAVITY_ACCELERATION"]
    air_volumetric_expansion_coeff = parameters["AIR_VOLUMETRIC_EXPANSION_COEFF"]
    floor_width = parameters["FLOOR_WIDTH"]
    nu_air = parameters["NU_AIR"]
    Pr_air = parameters["PR_AIR"]
    k_air = parameters["K_AIR"]
    A_roof = parameters["A_ROOF"]
    A_windows = parameters["WINDOWS_AREA"]
    A_walls = parameters["A_WALLS"]
    U_walls = parameters["U_WALLS"]
    U_roof = parameters["U_ROOF"]
    U_windows = parameters["WINDOWS_U"]

    # Design variables
    t_floor = design_variables["t_floor"]
    t_room = design_variables["t_room"]

    dae5 = []
    for i in range(1, n_steps):
        dae5.append(
            dae5_fun(
                t_floor[i],
                t_room[i],
                t_room[i - 1],
                t_amb[i],
                room_air_mass,
                cp_air,
                h,
                floor_area,
                stefan_boltzmann_constant,
                epsilon_concrete,
                gravity_acceleration,
                air_volumetric_expansion_coeff,
                floor_width,
                nu_air,
                Pr_air,
                k_air,
                A_roof,
                A_windows,
                A_walls,
                U_walls,
                U_roof,
                U_windows,
            )
        )
    return np.array(dae5)


def dae5_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]
    t_amb = parameters["t_amb"]
    room_air_mass = parameters["ROOM_AIR_MASS"]
    cp_air = parameters["CP_AIR"]
    h = parameters["H"]
    floor_area = parameters["FLOOR_AREA"]
    stefan_boltzmann_constant = parameters["STEFAN_BOLTZMANN_CONSTANT"]
    epsilon_concrete = parameters["EPSILON_CONCRETE"]
    gravity_acceleration = parameters["GRAVITY_ACCELERATION"]
    air_volumetric_expansion_coeff = parameters["AIR_VOLUMETRIC_EXPANSION_COEFF"]
    floor_width = parameters["FLOOR_WIDTH"]
    nu_air = parameters["NU_AIR"]
    Pr_air = parameters["PR_AIR"]
    k_air = parameters["K_AIR"]
    A_roof = parameters["A_ROOF"]
    A_windows = parameters["WINDOWS_AREA"]
    A_walls = parameters["A_WALLS"]
    U_walls = parameters["U_WALLS"]
    U_roof = parameters["U_ROOF"]
    U_windows = parameters["WINDOWS_U"]

    # Design variables
    t_floor = design_variables["t_floor"]
    t_room = design_variables["t_room"]

    ddae5_dt_floor = sp.lil_matrix((n_steps - 1, n_steps))
    ddae5_dt_room = sp.lil_matrix((n_steps - 1, n_steps))
    for i in range(1, n_steps):
        fun_inputs = (
            t_floor[i],
            t_room[i],
            t_room[i - 1],
            t_amb[i],
            room_air_mass,
            cp_air,
            h,
            floor_area,
            stefan_boltzmann_constant,
            epsilon_concrete,
            gravity_acceleration,
            air_volumetric_expansion_coeff,
            floor_width,
            nu_air,
            Pr_air,
            k_air,
            A_roof,
            A_windows,
            A_walls,
            U_walls,
            U_roof,
            U_windows,
        )
        ddae5_dt_floor[i - 1, i] = get_ddae5_dt_floor(*fun_inputs) + 1e-20
        ddae5_dt_room[i - 1, i] = get_ddae5_dt_room(*fun_inputs) + 1e-20
        ddae5_dt_room[i - 1, i - 1] = get_ddae5_dt_room_prev(*fun_inputs) + 1e-20

    ddae5_dt_floor = sparse_to_required_format(ddae5_dt_floor.tocsr())
    ddae5_dt_room = sparse_to_required_format(ddae5_dt_room.tocsr())

    dae5_jac = {
        "t_floor": ddae5_dt_floor,
        "t_room": ddae5_dt_room,
    }
    dae5_wrt = [
        "t_floor",
        "t_room",
    ]
    return (dae5_jac, dae5_wrt)


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


def t_cond_0_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]

    dt_cond_0_dt_cond = sp.lil_matrix((1, n_steps))
    dt_cond_0_dt_cond[0, 0] = 1
    dt_cond_0_dt_cond = dt_cond_0_dt_cond.tocsr()
    dt_cond_0_dt_cond = sparse_to_required_format(dt_cond_0_dt_cond)
    t_cond_0_jac = {"t_cond": dt_cond_0_dt_cond}
    t_cond_0_wrt = ["t_cond"]
    return (t_cond_0_jac, t_cond_0_wrt)


def t_tank_0_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]

    dt_tank_0_dt_tank = sp.lil_matrix((1, n_steps))
    dt_tank_0_dt_tank[0, 0] = 1
    dt_tank_0_dt_tank = dt_tank_0_dt_tank.tocsr()
    dt_tank_0_dt_tank = sparse_to_required_format(dt_tank_0_dt_tank)
    t_tank_0_jac = {"t_tank": dt_tank_0_dt_tank}
    t_tank_0_wrt = ["t_tank"]
    return (t_tank_0_jac, t_tank_0_wrt)


def t_out_heating_0_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]

    dt_out_heating_0_dt_out_heating = sp.lil_matrix((1, n_steps))
    dt_out_heating_0_dt_out_heating[0, 0] = 1
    dt_out_heating_0_dt_out_heating = dt_out_heating_0_dt_out_heating.tocsr()
    dt_out_heating_0_dt_out_heating = sparse_to_required_format(dt_out_heating_0_dt_out_heating)
    t_out_heating_0_jac = {"t_out_heating": dt_out_heating_0_dt_out_heating}
    t_out_heating_0_wrt = ["t_out_heating"]
    return (t_out_heating_0_jac, t_out_heating_0_wrt)


def t_floor_0_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]

    dt_floor_0_dt_floor = sp.lil_matrix((1, n_steps))
    dt_floor_0_dt_floor[0, 0] = 1
    dt_floor_0_dt_floor = dt_floor_0_dt_floor.tocsr()
    dt_floor_0_dt_floor = sparse_to_required_format(dt_floor_0_dt_floor)
    t_floor_0_jac = {"t_floor": dt_floor_0_dt_floor}
    t_floor_0_wrt = ["t_floor"]
    return (t_floor_0_jac, t_floor_0_wrt)


def t_room_0_constraint_sens(opt, design_variables: DesignVariables):
    # Parameters
    parameters = opt.parameters
    n_steps = parameters["n_steps"]

    dt_room_0_dt_room = sp.lil_matrix((1, n_steps))
    dt_room_0_dt_room[0, 0] = 1
    dt_room_0_dt_room = dt_room_0_dt_room.tocsr()
    dt_room_0_dt_room = sparse_to_required_format(dt_room_0_dt_room)
    t_room_0_jac = {"t_room": dt_room_0_dt_room}
    t_room_0_wrt = ["t_room"]
    return (t_room_0_jac, t_room_0_wrt)


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
    (dae1_jac, dae1_wrt) = dae1_constraint_sens(opt, design_variables)
    (dae2_jac, dae2_wrt) = dae2_constraint_sens(opt, design_variables)
    (dae3_jac, dae3_wrt) = dae3_constraint_sens(opt, design_variables)
    (dae4_jac, dae4_wrt) = dae4_constraint_sens(opt, design_variables)
    (dae5_jac, dae5_wrt) = dae5_constraint_sens(opt, design_variables)
    (battery_soc_jac, battery_soc_wrt) = battery_soc_constraint_sens(opt, design_variables)
    (battery_energy_jac, battery_energy_wrt) = battery_energy_constraint_sens(opt, design_variables)
    (p_grid_jac, p_grid_wrt) = p_grid_constraint_sens(opt, design_variables)
    (t_cond_0_jac, t_cond_0_wrt) = t_cond_0_constraint_sens(opt, design_variables)
    (t_tank_0_jac, t_tank_0_wrt) = t_tank_0_constraint_sens(opt, design_variables)
    (t_out_heating_0_jac, t_out_heating_0_wrt) = t_out_heating_0_constraint_sens(opt, design_variables)
    (t_floor_0_jac, t_floor_0_wrt) = t_floor_0_constraint_sens(opt, design_variables)
    (t_room_0_jac, t_room_0_wrt) = t_room_0_constraint_sens(opt, design_variables)
    (e_bat_0_jac, e_bat_0_wrt) = e_bat_0_constraint_sens(opt, design_variables)
    (p_bat_0_jac, p_bat_0_wrt) = p_bat_0_constraint_sens(opt, design_variables)

    return {
        "obj": obj_jac,
        "dae1_constraint": dae1_jac,
        "dae2_constraint": dae2_jac,
        "dae3_constraint": dae3_jac,
        "dae4_constraint": dae4_jac,
        "dae5_constraint": dae5_jac,
        "battery_soc_constraint": battery_soc_jac,
        "battery_energy_constraint": battery_energy_jac,
        "p_grid_constraint": p_grid_jac,
        "t_cond_0_constraint": t_cond_0_jac,
        "t_tank_0_constraint": t_tank_0_jac,
        # "t_out_heating_0_constraint": t_out_heating_0_jac,
        "t_floor_0_constraint": t_floor_0_jac,
        "t_room_0_constraint": t_room_0_jac,
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

    # States
    # t_cond        = y[0]
    # t_tank        = y[1]
    # t_out_heating = y[2]
    # t_floor       = y[3]
    # t_room        = y[4]
    t_cond: DesignVariableInfo = {
        "name": "t_cond",
        "n_params": n_steps,
        "type": "c",
        "lower": 273,
        "upper": 500,
        "initial_value": history["t_cond"][-1] if history else 300,
        "scale": 1 / 300,
    }
    opt.add_design_variables_info(t_cond)

    t_tank: DesignVariableInfo = {
        "name": "t_tank",
        "n_params": n_steps,
        "type": "c",
        "lower": 273,
        "upper": 500,
        "initial_value": history["t_tank"][-1] if history else 300,
        "scale": 1 / 300,
    }
    opt.add_design_variables_info(t_tank)

    t_out_heating: DesignVariableInfo = {
        "name": "t_out_heating",
        "n_params": n_steps,
        "type": "c",
        "lower": 273,
        "upper": 500,
        "initial_value": history["t_out_heating"][-1] if history else 300,
        "scale": 1 / 300,
    }
    opt.add_design_variables_info(t_out_heating)

    t_floor: DesignVariableInfo = {
        "name": "t_floor",
        "n_params": n_steps,
        "type": "c",
        "lower": 253,
        "upper": 500,
        "initial_value": history["t_floor"][-1] if history else 300,
        "scale": 1 / 300,
    }
    opt.add_design_variables_info(t_floor)

    t_room: DesignVariableInfo = {
        "name": "t_room",
        "n_params": n_steps,
        "type": "c",
        "lower": parameters["T_TARGET"],
        "upper": 500,
        "initial_value": history["t_room"][-1] if history else 300,
        "scale": 1 / 300,
    }
    opt.add_design_variables_info(t_room)

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
        "t_cond": np.ones(n_steps),
        "t_tank": np.ones(n_steps),
        "t_out_heating": np.ones(n_steps),
        "t_floor": np.ones(n_steps),
        "t_room": np.ones(n_steps),
        "p_compressor": np.ones(n_steps),
        "m_dot_cond": np.ones(n_steps),
        "m_dot_heating": np.ones(n_steps),
        "p_bat": np.ones(n_steps),
        "e_bat": np.ones(n_steps),
    }

    (dae1_jac, dae1_wrt) = dae1_constraint_sens(opt, dummy_design_variables)
    (dae2_jac, dae2_wrt) = dae2_constraint_sens(opt, dummy_design_variables)
    (dae3_jac, dae3_wrt) = dae3_constraint_sens(opt, dummy_design_variables)
    (dae4_jac, dae4_wrt) = dae4_constraint_sens(opt, dummy_design_variables)
    (dae5_jac, dae5_wrt) = dae5_constraint_sens(opt, dummy_design_variables)
    (battery_soc_jac, battery_soc_wrt) = battery_soc_constraint_sens(opt, dummy_design_variables)
    (battery_energy_jac, battery_energy_wrt) = battery_energy_constraint_sens(opt, dummy_design_variables)
    (p_grid_jac, p_grid_wrt) = p_grid_constraint_sens(opt, dummy_design_variables)
    (t_cond_0_jac, t_cond_0_wrt) = t_cond_0_constraint_sens(opt, dummy_design_variables)
    (t_tank_0_jac, t_tank_0_wrt) = t_tank_0_constraint_sens(opt, dummy_design_variables)
    (t_out_heating_0_jac, t_out_heating_0_wrt) = t_out_heating_0_constraint_sens(opt, dummy_design_variables)
    (t_floor_0_jac, t_floor_0_wrt) = t_floor_0_constraint_sens(opt, dummy_design_variables)
    (t_room_0_jac, t_room_0_wrt) = t_room_0_constraint_sens(opt, dummy_design_variables)
    (e_bat_0_jac, e_bat_0_wrt) = e_bat_0_constraint_sens(opt, dummy_design_variables)
    (p_bat_0_jac, p_bat_0_wrt) = p_bat_0_constraint_sens(opt, dummy_design_variables)

    # ∘ cop(t_cond) * p_compressor - m_dot_cond * cp_water * (t_cond - t_tank) = 0
    dae1_constraint: ConstraintInfo = {
        "name": "dae1_constraint",
        "n_constraints": n_steps,
        "lower": 0,
        "upper": 0,
        "function": dae1_constraint_fun,
        "scale": 1 / parameters["P_COMPRESSOR_MAX_LIMIT"],
        "wrt": dae1_wrt,
        "jac": dae1_jac,
    }
    opt.add_constraint_info(dae1_constraint)

    dae2_constraint: ConstraintInfo = {
        "name": "dae2_constraint",
        "n_constraints": n_steps - 1,
        "lower": 0,
        "upper": 0,
        "function": dae2_constraint_fun,
        "scale": 1 / (parameters["CP_WATER"] * 300),
        "wrt": dae2_wrt,
        "jac": dae2_jac,
    }
    opt.add_constraint_info(dae2_constraint)

    dae3_constraint: ConstraintInfo = {
        "name": "dae3_constraint",
        "n_constraints": n_steps,
        "lower": 0,
        "upper": 0,
        "function": dae3_constraint_fun,
        "scale": 1 / (parameters["CP_WATER"] * 5),  # deltaT: ~5K, m_dot_heating: ~1
        "wrt": dae3_wrt,
        "jac": dae3_jac,
    }
    opt.add_constraint_info(dae3_constraint)

    dae4_constraint: ConstraintInfo = {
        "name": "dae4_constraint",
        "n_constraints": n_steps - 1,
        "lower": 0,
        "upper": 0,
        "function": dae4_constraint_fun,
        "scale": 1 / (parameters["CP_WATER"] * 5),  # deltaT: ~5K, m_dot_heating: ~1
        "wrt": dae4_wrt,
        "jac": dae4_jac,
    }
    opt.add_constraint_info(dae4_constraint)

    dae5_constraint: ConstraintInfo = {
        "name": "dae5_constraint",
        "n_constraints": n_steps - 1,
        "lower": 0,
        "upper": 0,
        "function": dae5_constraint_fun,
        # "scale": 1,
        "scale": 1 / (parameters["U_ROOF"] * parameters["A_ROOF"] * 5),  # deltaT: ~5K,
        "wrt": dae5_wrt,
        "jac": dae5_jac,
    }
    opt.add_constraint_info(dae5_constraint)

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

    # Initial value constraints
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

    # t_out_heating_0_constraint: ConstraintInfo = {
    #     "name": "t_out_heating_0_constraint",
    #     "n_constraints": 1,
    #     "lower": parameters["y0"]["t_out_heating"],
    #     "upper": parameters["y0"]["t_out_heating"],
    #     "function": lambda _, design_variables: design_variables["t_out_heating"][0],
    #     "scale": 1 / parameters["y0"]["t_out_heating"],
    #     "wrt": t_out_heating_0_wrt,
    #     "jac": t_out_heating_0_jac,
    # }
    # opt.add_constraint_info(t_out_heating_0_constraint)

    t_floor_0_constraint: ConstraintInfo = {
        "name": "t_floor_0_constraint",
        "n_constraints": 1,
        "lower": parameters["y0"]["t_floor"],
        "upper": parameters["y0"]["t_floor"],
        "function": lambda _, design_variables: design_variables["t_floor"][0],
        "scale": 1 / parameters["y0"]["t_floor"],
        "wrt": t_floor_0_wrt,
        "jac": t_floor_0_jac,
    }
    opt.add_constraint_info(t_floor_0_constraint)

    t_room_0_constraint: ConstraintInfo = {
        "name": "t_room_0_constraint",
        "n_constraints": 1,
        "lower": parameters["y0"]["t_room"],
        "upper": parameters["y0"]["t_room"],
        "function": lambda _, design_variables: design_variables["t_room"][0],
        "scale": 1 / parameters["y0"]["t_room"],
        "wrt": t_room_0_wrt,
        "jac": t_room_0_jac,
    }
    opt.add_constraint_info(t_room_0_constraint)

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
        "max_iter": 500,
        # "obj_scaling_factor": 1e-1,
        "mu_strategy": "adaptive",
        "alpha_for_y": "safer-min-dual-infeas",
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
        print(sol)

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
    parameters["y0"] = y0

    run_optimization(parameters, plot=True)
