import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from parameters import PARAMS, Y0
from utils import (
    get_dynamic_parameters,
    plot_styles,
    jax_to_numpy,
    plot_film,
    cop,
    get_fixed_energy_cost_by_second,
    get_battery_depreciation_by_joule,
    get_solar_panels_depreciation_by_second,
    get_hp_depreciation_by_joule,
    get_tank_depreciation_by_second,
)
from pyoptsparse import History
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def dae_system(y, y_prev, u, p, h):
    r"""
    Solve the following system of non-linear equations

    1) ∘ COP = cop(t_cond)

    2) ∘ q_dot_cond = COP * p_comp
    3) ∘ q_dot_cond = m_dot_cond * cp_water * (t_cond - t_tank)

    4) ∘ m_tank * cp_water * ((t_tank - t_tank_prev)/h)
        = m_dot_cond * cp_water * t_cond
        + m_dot_heating * cp_water * t_out_heating
        - m_dot_tank * cp_water * t_tank
        - q_dot_loss

    5) ∘ m_dot_tank = m_dot_cond + m_dot_heating

    6) ∘ q_dot_loss = U_tank * A_tank * (t_tank - t_amb)

    7) ∘ floor_mass * cp_concrete * ((t_floor - t_floor_prev)/h)
        - q_conduction_floor
        + q_convection_floor
        + q_radiation_floor
        = 0

    8) ∘ room_air_mass * cp_air * ((t_room - t_room_prev)/h)
        - q_convection_floor
        - q_radiation_floor
        + U_walls * A_walls * (t_room - t_amb)
        + U_roof * A_roof * (t_room - t_amb)
        + U_windows * A_windows * (t_room - t_amb)
        = 0

    9) ∘ q_convection_floor - h_floor_air * floor_area * (t_floor - t_room) = 0
        --------------------------------------
        ∘ Nu_floor_air = (h_floor_air * floor_width) / k_air
        ∘ Nu_floor_air = 0.15 * Ra_floor_air**(1/3)
        ∘ Ra_floor_air = Gr_air * Pr_air
        ∘ Gr_floor_air = (gravity_acceleration * air_volumetric_expansion_coeff * np.abs(t_floor - t_room) * L**3) / nu_air**2

    10) ∘ q_radiation_floor - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4) = 0

    11) ∘ q_conduction_floor - m_dot_heating * cp_water * (t_tank - t_out_heating) = 0
    12) ∘ q_conduction_floor - U_tubes * A_tubes * DeltaT_tubes = 0
        --------------------------------------
        ∘ DeltaT_tubes = ((t_tank - t_floor) + (t_out_heating - t_floor)) / 2
        ∘ U_tubes = 1 / ((1 / h_tube_water) + (1 / (k_pex / tube_thickness)))
        ∘ Nu_tube_water = (h_tube_water * tube_inner_diameter) / k_water
        ∘ Nu_tube_water = 0.023 * Re_tube_water**0.8 * Pr_water**(1/3)
        ∘ Re_tube_water = (tube_inner_diameter * v * rho_water) / mu_water_at_320K = (4 * m_dot_heating) / (pi * mu_water_at_320K * tube_inner_diameter)

    substituting equations 1) and 2) into 3)
    13) ∘ cop(t_cond) * p_comp - m_dot_cond * cp_water * (t_cond - t_tank) = 0

    substituting equations 5) and 6) into 4)
    14) ∘ m_tank * cp_water * ((t_tank - t_tank_prev)/h)
        - m_dot_cond * cp_water * t_cond
        - m_dot_heating * cp_water * t_out_heating
        + (m_dot_cond + m_dot_heating) * cp_water * t_tank
        + U_tank * A_tank * (t_tank - t_amb)
        = 0


    substitute the values for q_conduction, q_radiation and q_convection
    into the equations 7), 8) and 12):

    q_conduction_floor = m_dot_heating * cp_water * (t_tank - t_out_heating)
    q_convection_floor = h_floor_air * floor_area * (t_floor - t_room)
    q_radiation_floor = stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)

    15) ∘ m_dot_heating * cp_water * (t_tank - t_out_heating) - U_tubes * A_tubes * DeltaT_tubes = 0

    16) ∘ floor_mass * cp_concrete * ((t_floor - t_floor_prev)/h)
        - m_dot_heating * cp_water * (t_tank - t_out_heating)
        + h_floor_air * floor_area * (t_floor - t_room)
        + stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
        = 0

    17) ∘ room_air_mass * cp_air * ((t_room - t_room_prev)/h)
        - h_floor_air * floor_area * (t_floor - t_room)
        - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
        + U_walls * A_walls * (t_room - t_amb)
        + U_roof * A_roof * (t_room - t_amb)
        + U_windows * A_windows * (t_room - t_amb)
        = 0

    We can also get T_out_heating from 15)
    ∘ m_dot_heating * cp_water * (t_tank - t_out_heating) - U_tubes * A_tubes * DeltaT_tubes = 0
    and
    ∘ DeltaT_tubes = ((t_tank - t_floor) + (t_out_heating - t_floor)) / 2 = (t_tank + t_out_heating - 2 * t_floor) / 2

    giving:
    18) t_out_heating = ((2 * A_tubes * U_tubes * t_floor - A_tubes * U_tubes * t_tank + 2 * cp_water * m_dot_heating * t_tank) / (A_tubes * U_tubes + 2 * cp_water * m_dot_heating))

    Final system consist of equations 13), 19), 20), 17):

    ∘ cop(t_cond) * p_compressor - m_dot_cond * cp_water * (t_cond - t_tank) = 0

    ∘ m_tank * cp_water * ((t_tank - t_tank_prev)/h)
        - m_dot_cond * cp_water * t_cond
        - m_dot_heating * cp_water * t_out_heating
        + (m_dot_cond + m_dot_heating) * cp_water * t_tank
        + U_tank * A_tank * (t_tank - t_amb)
        = 0

    ∘ floor_mass * cp_concrete * ((t_floor - t_floor_prev)/h)
        - m_dot_heating * cp_water * (t_tank - t_out_heating)
        + h_floor_air * floor_area * (t_floor - t_room)
        + stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
        = 0

    ∘ room_air_mass * cp_air * ((t_room - t_room_prev)/h)
        - h_floor_air * floor_area * (t_floor - t_room)
        - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
        + U_walls * A_walls * (t_room - t_amb)
        + U_roof * A_roof * (t_room - t_amb)
        + U_windows * A_windows * (t_room - t_amb)
        = 0


    4 equations for 4 unknowns:
    t_cond, t_tank, t_floor, t_room

    Making:
    t_cond        = y[0]
    t_tank        = y[1]
    t_floor       = y[2]
    t_room        = y[3]

    m_dot_cond    = u[0]
    m_dot_heating = u[1]
    p_compressor  = u[2]
    t_amb         = u[3]

    floor_mass                      =         p[0]
    cp_concrete                     =         p[1]
    gravity_acceleration            =         p[2]
    air_volumetric_expansion_coeff  =         p[3]
    floor_width                     =         p[4]
    nu_air                          =         p[5]
    Pr_air                          =         p[6]
    k_air                           =         p[7]
    tube_inner_diameter             =         p[8]
    floor_area                      =         p[9]
    stefan_boltzmann_constant       =         p[10]
    epsilon_concrete                =         p[11]
    cp_water                        =         p[12]
    mu_water_at_320K                =         p[13]
    Pr_water                        =         p[14]
    k_water                         =         p[15]
    k_pex                           =         p[16]
    tube_thickness                  =         p[17]
    A_tubes                         =         p[18]
    room_air_mass                   =         p[19]
    cp_air                          =         p[20]
    A_walls                         =         p[21]
    A_roof                          =         p[22]
    A_windows                       =         p[23]
    U_walls                         =         p[24]
    U_roof                          =         p[25]
    U_windows                       =         p[26]
    m_tank                          =         p[27]
    U_tank                          =         p[28]
    A_tank                          =         p[29]
    """
    t_cond = y[0]
    t_tank = y[1]
    t_tank_prev = y_prev[1]
    t_floor = y[2]
    t_floor_prev = y_prev[2]
    t_room = y[3]
    t_room_prev = y_prev[3]

    m_dot_cond = u[0]
    m_dot_heating = u[1]
    p_compressor = u[2]
    t_amb = u[3]

    floor_mass = p[0]
    cp_concrete = p[1]
    gravity_acceleration = p[2]
    air_volumetric_expansion_coeff = p[3]
    floor_width = p[4]
    nu_air = p[5]
    Pr_air = p[6]
    k_air = p[7]
    tube_inner_diameter = p[8]
    floor_area = p[9]
    stefan_boltzmann_constant = p[10]
    epsilon_concrete = p[11]
    cp_water = p[12]
    mu_water_at_320K = p[13]
    Pr_water = p[14]
    k_water = p[15]
    k_pex = p[16]
    tube_thickness = p[17]
    A_tubes = p[18]
    room_air_mass = p[19]
    cp_air = p[20]
    A_walls = p[21]
    A_roof = p[22]
    A_windows = p[23]
    U_walls = p[24]
    U_roof = p[25]
    U_windows = p[26]
    m_tank = p[27]
    U_tank = p[28]
    A_tank = p[29]

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

    h_tube_water = get_h_tube_water(
        tube_inner_diameter,
        mu_water_at_320K,
        Pr_water,
        k_water,
        m_dot_heating,
    )

    U_tubes = 1 / ((1 / h_tube_water) + (1 / (k_pex / tube_thickness)))

    t_out_heating = (
        2 * A_tubes * U_tubes * t_floor - A_tubes * U_tubes * t_tank + 2 * cp_water * m_dot_heating * t_tank
    ) / (A_tubes * U_tubes + 2 * cp_water * m_dot_heating)

    return [
        # 1
        cop(t_cond) * p_compressor - m_dot_cond * cp_water * (t_cond - t_tank),
        # 2
        m_tank * cp_water * ((t_tank - t_tank_prev) / h)
        - m_dot_cond * cp_water * t_cond
        - m_dot_heating * cp_water * t_out_heating
        + (m_dot_cond + m_dot_heating) * cp_water * t_tank
        + U_tank * A_tank * (t_tank - t_amb),
        # 3
        floor_mass * cp_concrete * ((t_floor - t_floor_prev) / h)
        - m_dot_heating * cp_water * (t_tank - t_out_heating)
        + h_floor_air * floor_area * (t_floor - t_room)
        + stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4),
        # 4
        room_air_mass * cp_air * ((t_room - t_room_prev) / h)
        - h_floor_air * floor_area * (t_floor - t_room)
        - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
        + U_walls * A_walls * (t_room - t_amb)
        + U_roof * A_roof * (t_room - t_amb)
        + U_windows * A_windows * (t_room - t_amb),
    ]


def get_h_floor_air(
    t_floor, t_room, gravity_acceleration, air_volumetric_expansion_coeff, floor_width, nu_air, Pr_air, k_air, A_roof
):
    """
    Convective heat transfer coefficient for natural convection over a radiant floor
    """
    # Gr_air = (gravity_acceleration * air_volumetric_expansion_coeff * jnp.abs(T_floor - T_room) * L**3) / nu_air**2
    # Ra_air = Gr_air * Pr_air
    # Nu_air = 0.15 * Ra_air**(1/3)
    # h_air = Nu_air * k_air / floor_width

    floor_perimeter = 4 * floor_width
    floor_area = A_roof
    L = floor_area / floor_perimeter  # characteristic_length

    # Absolute value for the difference between t_floor and t_room
    # since if negative, would imply a negative h, which doesn't have physical meaning
    Gr_floor_air = (
        jnp.abs(gravity_acceleration * air_volumetric_expansion_coeff * (t_floor - t_room) * L**3 / nu_air**2) + 1e-6
    )
    Ra_floor_air = Gr_floor_air * Pr_air
    Nu_floor_air = 0.15 * Ra_floor_air ** (1 / 3)
    h_floor_air = Nu_floor_air * k_air / L
    return h_floor_air


def get_h_tube_water(tube_inner_diameter, mu_water_at_320K, Pr_water, k_water, m_dot_heating):
    """
    Assuming that the flow is always in turbulent region
    Which is not the case
    So we are overestimating the convective coefficient here
    For laminar flow: Nu_tube_water = 3.66

    Using Dittus-Boelter correlation for the Nusselt number
    """
    # Re_water = (4 * m_dot_heating) / (pi * mu_water_at_320K * tube_inner_diameter)
    # Nu_water = 0.023 * Re_water**0.8 * Pr_water**0.3
    # h_tube_water = Nu_water * k_water / tube_inner_diameter
    pi = 3.141592

    Re_tube_water = (4 * m_dot_heating) / (pi * mu_water_at_320K * tube_inner_diameter)
    Nu_tube_water = 0.023 * Re_tube_water**0.8 * Pr_water ** (1 / 3)
    h_tube_water = Nu_tube_water * k_water / tube_inner_diameter
    return h_tube_water


def solve(y_0, u, dae_p, h, n_steps):
    """
    Solve the differential-algebraic equation (DAE) system using implicit time-stepping.

    Parameters:
    y_0: Initial conditions of the states.
    u: Control inputs over time.
    dae_p: Parameters for the DAE system.
    h: Time step size.
    n_steps: Number of time steps to perform.

    Returns:
    State evolution over time.
    """
    # Initialize solution
    y = np.zeros((len(y_0), n_steps))

    # Initial conditions
    y[:, 0] = y_0

    # Time-stepping loop
    for n in range(1, n_steps):
        # Use fsolve to solve the nonlinear system
        y_n = fsolve(dae_system, y[:, n - 1], args=(y[:, n - 1], u[:, n], dae_p, h))
        y[:, n] = y_n

    return y


def j_compressor_cost(y, u, p, h, excess_prices):
    p_compressor = u[2]
    return jnp.sum(h * excess_prices * p_compressor)


def j_t_room_min(y, u, p, h):
    t_room = y[3]
    return jnp.min(t_room)


def adjoint_gradients(y, u, p, h, n_steps, f, j_fun, j_extra_args):
    r"""
    Adjoint gradients of the cost function with respect to parameters

    j(y, u, p) = Σr(y_n, p, u_n)
    ó
    j(y, u, p) = min(y)
    min_p C = j(y, u, p)
    s.t.
        f(y_n, y_(n-1), p, u_n) = 0
        g(y_n, p, u_n) = 0

    where f are the differential constraints
    and g are the algebraic constraints

    by augmenting the system
    p' = 0

    min_{p_n} C = j(y, u, p)
        f(y_n, y_(n-1), p_n, u_n) = 0
        (p_n - p_(n-1)) / h = 0

    Lagrangian:
    L = j(y, u, p)
        - Σ λ_n f(y_n, y_(n-1), p_n, u_n)
        - Σ μ_n (p_n - p_(n-1))

    ∂L/∂y_n = 0 = ∂j(y, u, p)/∂y_n
        - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂y_n
        - λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1), u_(n+1))/∂y_n
    ∂L/∂p_n = 0 = ∂j(y, u, p)/∂p_n
        - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂p_n
        - μ_n + μ_(n+1)
    ∂L/∂u_n = ∂C/∂u_n = ∂j(y, u, p)/∂u_n
        - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂u_n

    Solve for λ_n and μ_n at each step:
    λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂y_n = ∂j(y, u, p)/∂y_n
        - λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1), u_(n+1))/∂y_n

    μ_n = ∂j(y, u, p)/∂p_n
        - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂p_n
        + μ_(n+1)

    Terminal conditions:
    λ_N ∂f(y_N, y_(N-1), p_N, u_N)/∂y_N = ∂j(y, u, p)/∂y_N
    μ_N = ∂j(y, u, p)/∂p_N - λ_N ∂f(y_N, y_(N-1), p_N, u_N)/∂p_N

    Solve for initial timestep:
    ∂L/∂y_0 = ∂C/∂y_0 = (∂j/∂y_0 - λ_1 ∂f(y_1, y_0, p_1, u_1)/∂y_0)^T
    ∂L/∂p_0 = ∂C/∂p_0 = (∂j/∂p_0 + μ_1)^T
    """
    # Obtain shapes of jacobians
    # JACOBIANS
    # jax.jacobian will automatically
    # use jax.jacrev or jax.jacfwd
    # based on the number of inputs and outputs
    # and convert JAX jacobian functions to NumPy functions
    get_djdy = jax_to_numpy(jax.jit(jax.jacobian(j_fun, argnums=0)))
    get_djdu = jax_to_numpy(jax.jit(jax.jacobian(j_fun, argnums=1)))
    get_djdp = jax_to_numpy(jax.jit(jax.jacobian(j_fun, argnums=2)))
    get_dfdy = jax_to_numpy(jax.jit(jax.jacobian(f, argnums=0)))
    get_dfdy_prev = jax_to_numpy(jax.jit(jax.jacobian(f, argnums=1)))
    get_dfdu = jax_to_numpy(jax.jit(jax.jacobian(f, argnums=2)))
    get_dfdp = jax_to_numpy(jax.jit(jax.jacobian(f, argnums=3)))

    # Obtain shapes of jacobians
    dfdy = get_dfdy(y[:, 1], y[:, 0], u[:, 0], p, h)
    dfdp = get_dfdp(y[:, 1], y[:, 0], u[:, 0], p, h)
    n_eqs = dfdy.shape[0]
    n_states = dfdy.shape[1]
    n_params = dfdp.shape[1]

    # Initialize adjoint variables
    adj_lambda = np.zeros((n_eqs, n_steps + 1))
    adj_mu = np.zeros((n_params, n_steps + 1))
    dCdy_0 = np.zeros(n_states)
    dCdp_0 = np.zeros(n_params)
    dCdu = np.zeros(u.shape)

    # Backward propagation of adjoint variables
    for n in range(n_steps - 1, -1, -1):
        # tic = time.time()
        y_current = y[:, n]
        y_prev = y[:, n - 1]
        u_current = u[:, n]
        dfdy_n = get_dfdy(y_current, y_prev, u_current, p, h)
        dfdu_n = get_dfdu(y_current, y_prev, u_current, p, h)
        dfdp_n = get_dfdp(y_current, y_prev, u_current, p, h)
        djdy_n = get_djdy(y, u, p, h, *j_extra_args)[:, n]
        djdu_n = get_djdu(y, u, p, h, *j_extra_args)[:, n]
        djdp_n = get_djdp(y, u, p, h, *j_extra_args)

        if n == n_steps - 1:
            # Terminal condition
            # λ_N ∂f(y_N, y_(N-1), p_N, u_N)/∂y_N = ∂j(y, u, p)/∂y_N
            # μ_N = ∂j(y, u, p)/∂p_N - λ_N ∂f(y_N, y_(N-1), p_N, u_N)/∂p_N
            adj_lambda[:, n] = np.linalg.solve(dfdy_n.T, djdy_n)
            adj_mu[:, n] = djdp_n - np.dot(adj_lambda[:, n].T, dfdp_n)
        elif n == 0:
            # Initial timestep
            # ∂L/∂y_0 = ∂C/∂y_0 = (∂j/∂y_0 - λ_1 ∂f(y_1, y_0, p_1, u_1)/∂y_0)^T
            # ∂L/∂p_0 = ∂C/∂p_0 = (∂j/∂p_0 + μ_1)^T
            dfdy_prev = get_dfdy_prev(y[:, 1], y[:, 0], u[:, 1], p, h)
            dCdy_0 = djdy_n - np.dot(adj_lambda[:, 1].T, dfdy_prev)
            dCdp_0 = djdp_n + adj_mu[:, 1]
        else:
            # λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂y_n = ∂j(y, u, p)/∂y_n
            #     - λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1), u_(n+1))/∂y_n
            #
            # μ_n = ∂j(y, u, p)/∂p_n
            #     - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂p_n
            #     + μ_(n+1)
            dfdy_prev = get_dfdy_prev(y[:, n + 1], y[:, n], u[:, n + 1], p, h)
            adj_lambda[:, n] = np.linalg.solve(dfdy_n.T, (djdy_n - np.dot(adj_lambda[:, n + 1].T, dfdy_prev)))
            adj_mu[:, n] = djdp_n - np.dot(adj_lambda[:, n].T, dfdp_n) + adj_mu[:, n + 1]

        # ∂L/∂u_n = ∂C/∂u_n = ∂r(y_n, p_n, u_n)/∂u_n - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂u_n
        dCdu[:, n] = djdu_n - adj_lambda[:, n].T @ dfdu_n

    return dCdy_0, dCdp_0, dCdu


def dae_forward(y0, u, dae_p, h, n_steps):
    y = solve(y0, u, dae_p, h, n_steps)
    return y


def dae_adjoints(y, u, dae_p, h, n_steps, f, j_fun, j_extra_args):
    dCdy_0_adj, dCdp_adj, dCdu_adj = adjoint_gradients(y, u, dae_p, h, n_steps, f, j_fun, j_extra_args)
    return dCdy_0_adj, dCdp_adj, dCdu_adj


def fd_gradients(y0, u, dae_p, h, n_steps, j_fun, j_extra_args):
    """
    Finite difference of function j_fun
    with respect
        initial conditions y0,
        parameters p,
        and control signals u at timestep 100

    f'(x) = (f(y+h) - f(y)) / h
    """
    delta = 1e-5

    # Initial solution
    y = solve(y0, u, dae_p, h, n_steps)
    # print("fd solve: ", y)

    dfdy0 = []
    for i in range(len(y0)):
        y0_perturbed = y0.copy()  # Create a new copy of y0
        y0_perturbed[i] += delta
        y_perturbed = solve(y0_perturbed, u, dae_p, h, n_steps)
        dfdy0.append((j_fun(y_perturbed, u, dae_p, h, *j_extra_args) - j_fun(y, u, dae_p, h, *j_extra_args)) / delta)

    dfdp = []
    for i in range(len(dae_p)):
        p_perturbed = dae_p.copy()  # Create a new copy of p
        p_perturbed[i] += delta
        y_perturbed = solve(y0, u, p_perturbed, h, n_steps)
        dfdp.append((j_fun(y_perturbed, u, dae_p, h, *j_extra_args) - j_fun(y, u, dae_p, h, *j_extra_args)) / delta)

    dfdu_3 = []
    for i in range(len(u[:, 0])):
        u_perturbed = u.copy()  # Create a new copy of u
        u_perturbed[i, 3] += delta
        y_perturbed = solve(y0, u_perturbed, dae_p, h, n_steps)
        dfdu_3.append(
            (j_fun(y_perturbed, u_perturbed, dae_p, h, *j_extra_args) - j_fun(y, u, dae_p, h, *j_extra_args)) / delta
        )

    return dfdy0, dfdp, dfdu_3


fig = None


def plot_thermals(y, u, n_steps, dae_p, parameters, title=None, show=True, block=True, save=True):
    global fig
    # Close the previous figure if it exists
    if fig:
        plt.close(fig)

    h = parameters["H"]
    t = np.linspace(0, n_steps * h, n_steps) / 3600  # seconds to hours

    t_cond = y[0]
    t_tank = y[1]
    t_floor = y[2]
    t_room = y[3]
    m_dot_cond = u[0]
    m_dot_heating = u[1]
    p_compressor = u[2]
    t_amb = u[3]

    gravity_acceleration = dae_p[2]
    air_volumetric_expansion_coeff = dae_p[3]
    floor_width = dae_p[4]
    nu_air = dae_p[5]
    Pr_air = dae_p[6]
    k_air = dae_p[7]
    tube_inner_diameter = dae_p[8]
    floor_area = dae_p[9]
    stefan_boltzmann_constant = dae_p[10]
    epsilon_concrete = dae_p[11]
    cp_water = dae_p[12]
    mu_water_at_320K = dae_p[13]
    Pr_water = dae_p[14]
    k_water = dae_p[15]
    k_pex = dae_p[16]
    tube_thickness = dae_p[17]
    A_tubes = dae_p[18]
    A_roof = dae_p[22]

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

    h_tube_water = get_h_tube_water(
        tube_inner_diameter,
        mu_water_at_320K,
        Pr_water,
        k_water,
        m_dot_heating,
    )

    U_tubes = 1 / ((1 / h_tube_water) + (1 / (k_pex / tube_thickness)))
    t_out_heating = (
        2 * A_tubes * U_tubes * t_floor - A_tubes * U_tubes * t_tank + 2 * cp_water * m_dot_heating * t_tank
    ) / (A_tubes * U_tubes + 2 * cp_water * m_dot_heating)

    q_conduction_floor = m_dot_heating * cp_water * (t_tank - t_out_heating)
    q_convection_floor = h_floor_air * floor_area * (t_floor - t_room)
    q_radiation_floor = stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].plot(t, t_cond, label="t_cond", **plot_styles[0])
    axes[0].plot(t, t_tank, label="t_tank", **plot_styles[1])
    axes[0].plot(t, t_floor, label="t_floor", **plot_styles[2])
    axes[0].plot(t, t_room, label="t_room", **plot_styles[3])
    axes[0].legend(loc="upper left")
    ax0 = axes[0].twinx()
    ax0.plot(t, t_amb, label="t_amb", **plot_styles[4])
    ax0.legend(loc="upper right")

    axes[1].plot(t, q_conduction_floor, label="q_conduction_floor", **plot_styles[0])
    axes[1].plot(t, q_convection_floor + q_radiation_floor, label="q_rad+conv,", **plot_styles[1])
    axes[1].plot(t, q_radiation_floor, label="q_radiation_floor", **plot_styles[2])
    axes[1].plot(t, q_convection_floor, label="q_convection_floor", **plot_styles[3])
    axes[1].legend(loc="upper left")

    axes[2].plot(t, m_dot_heating, label="m_dot_heating", **plot_styles[0])
    axes[2].plot(t, m_dot_cond, label="m_dot_cond", **plot_styles[1])
    axes[2].legend(loc="upper left")
    ax2 = axes[2].twinx()
    ax2.plot(t, p_compressor, label="p_compressor", **plot_styles[2])
    ax2.legend(loc="upper right")

    # Show the plots
    if show:
        # Save and close plot
        # plt.ion()  # Turn on the interactive mode
        plt.show(block=block)  # Draw the figure
        plt.pause(0.3)  # Time for the figure to load

    if save:
        plt.savefig(f"tmp/frame_{time.time()}.png")


def save_simulation_plots(y, u, n_steps, dae_p, parameters):
    h = parameters["H"]
    t = np.linspace(0, n_steps * h, n_steps) / 3600  # seconds to hours

    # States
    t_cond = y[0]
    t_tank = y[1]
    t_floor = y[2]
    t_room = y[3]

    # Controls
    m_dot_cond = u[0]
    m_dot_heating = u[1]
    p_compressor = u[2]
    t_amb = u[3]

    # Parameters
    gravity_acceleration = dae_p[2]
    air_volumetric_expansion_coeff = dae_p[3]
    floor_width = dae_p[4]
    nu_air = dae_p[5]
    Pr_air = dae_p[6]
    k_air = dae_p[7]
    tube_inner_diameter = dae_p[8]
    floor_area = dae_p[9]
    stefan_boltzmann_constant = dae_p[10]
    epsilon_concrete = dae_p[11]
    cp_water = dae_p[12]
    mu_water_at_320K = dae_p[13]
    Pr_water = dae_p[14]
    k_water = dae_p[15]
    k_pex = dae_p[16]
    tube_thickness = dae_p[17]
    A_tubes = dae_p[18]
    A_roof = dae_p[22]

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

    h_tube_water = get_h_tube_water(
        tube_inner_diameter,
        mu_water_at_320K,
        Pr_water,
        k_water,
        m_dot_heating,
    )

    U_tubes = 1 / ((1 / h_tube_water) + (1 / (k_pex / tube_thickness)))
    t_out_heating = (
        2 * A_tubes * U_tubes * t_floor - A_tubes * U_tubes * t_tank + 2 * cp_water * m_dot_heating * t_tank
    ) / (A_tubes * U_tubes + 2 * cp_water * m_dot_heating)

    q_conduction_floor = m_dot_heating * cp_water * (t_tank - t_out_heating)
    q_convection_floor = h_floor_air * floor_area * (t_floor - t_room)
    q_radiation_floor = stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)

    # K to ºC
    t_cond = t_cond - 273
    t_tank = t_tank - 273
    t_out_heating = t_out_heating - 273
    t_floor = t_floor - 273
    t_room = t_room - 273
    t_amb = t_amb - 273

    # Plot temperatures
    fig, ax = plt.subplots(figsize=(8.27, 2.4))  # A4: (8.27, 11.69)
    ax.plot(t, t_cond, label="Salida bomba calor", **plot_styles[0])
    ax.plot(t, t_tank, label="Tanque", **plot_styles[1])
    ax.plot(t, t_floor, label="Suelo", **plot_styles[2])
    ax.plot(t, t_room, label="Habitación", **plot_styles[3])
    ax.plot(t, t_amb, label="Ambiente", **plot_styles[4])
    ax.set_ylabel("Temperatura [ºC]")
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    save_plot(fig, ax, f"saves/plot_simulation_temperatures.svg")

    # Plot heats
    fig, ax = plt.subplots(figsize=(8.27, 2.4))  # A4: (8.27, 11.69)
    ax.plot(t, q_conduction_floor, label="Conducción a suelo", **plot_styles[0])
    ax.plot(t, q_convection_floor + q_radiation_floor, label="Calor total desde suelo", **plot_styles[1])
    ax.plot(t, q_radiation_floor, label="Radiación desde suelo", **plot_styles[2])
    ax.plot(t, q_convection_floor, label="Convección desde suelo", **plot_styles[3])
    ax.set_ylabel("Calor [W]")
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    save_plot(fig, ax, f"saves/plot_simulation_heat.svg")

    # Plot valve controls
    fig, ax = plt.subplots(figsize=(8.27, 2.4))  # A4: (8.27, 11.69)
    ax.plot(t, m_dot_heating, label="Suelo radiante", **plot_styles[0])
    ax.plot(t, m_dot_cond, label="Bomba de calor", **plot_styles[1])
    ax.set_ylabel(r"Caudal de agua $[kg \cdot s^{-1}]$")
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    save_plot(fig, ax, f"saves/plot_simulation_m_dot.svg")

    # Plot compressor control
    fig, ax = plt.subplots(figsize=(8.27, 2.4))  # A4: (8.27, 11.69)
    ax.plot(t, p_compressor, **plot_styles[0])
    ax.set_ylabel("Potencia Compresor [W]")
    ax.grid(True)
    ax.set_xlabel("Tiempo [h]")
    save_plot(fig, ax, f"saves/plot_simulation_compressor.svg")


def plot_full(y, u, i, histories, parameters, title=None, show=True, block=True, save=True):
    print(f"plotting...{title}")
    global fig
    # Close the previous figure if it exists
    if fig:
        plt.close(fig)

    # States
    t_cond = y[0]
    t_tank = y[1]
    t_floor = y[2]
    t_room = y[3]

    # Controls
    m_dot_cond = u[0]
    m_dot_heating = u[1]
    p_compressor = u[2]
    t_amb = u[3]

    # Design variables
    e_bat = histories["e_bat"][i]
    p_bat = histories["p_bat"][i]

    # Parameters
    solar_size = parameters["SOLAR_SIZE"]
    e_bat_max = parameters["E_BAT_MAX"]
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]
    p_required = parameters["p_required"]
    h = parameters["H"]
    t_amb = parameters["t_amb"]
    n_steps = parameters["t_amb"].shape[0]
    t = np.linspace(0, n_steps * h, n_steps)
    t = t / 3600
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]
    p_required = parameters["p_required"]
    p_solar = parameters["w_solar_per_w_installed"] * solar_size
    p_grid = -p_solar + p_compressor + p_bat + p_required

    # Create time array and set figsize to A4 dimensions (8.27 x 11.69 inches)
    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), sharex=True)
    fig.subplots_adjust(hspace=0.5)  # # Adjust the spacing between subplots

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # First subplot for inputs
    axes[0].plot(t, pvpc_prices, label="pvpc_price", **plot_styles[0])
    axes[0].plot(t, excess_prices, label="excess_price", **plot_styles[1])
    axes[0].set_ylabel(r"€$/(kW \cdot h)$")
    axes[0].legend()
    axes[0].grid(True)
    ax0 = axes[0].twinx()
    ax0.plot(t, p_required, label="p_required", **plot_styles[2])
    ax0.plot(t, -p_solar, label="p_solar", **plot_styles[3])
    ax0.set_ylabel("K")
    ax0.legend()
    if title:
        axes[0].set_title(title)

    # Second subplot for temperatures
    axes[1].plot(t, t_tank, label="t_tank", **plot_styles[0])
    axes[1].plot(t, t_floor, label="t_floor", **plot_styles[1])
    axes[1].plot(t, t_room, label="t_room", **plot_styles[2])
    axes[1].plot(t, t_amb, label="t_amb", **plot_styles[3])
    axes[1].set_ylabel("Temperature (K)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)
    ax1 = axes[1].twinx()
    ax1.plot(t, e_bat, label="E_bat", **plot_styles[4])
    ax1.set_ylabel("Ws")
    ax1.legend(loc="upper right")

    # Third subplot for controls
    axes[2].plot(t, p_grid, label="p_grid", **plot_styles[0])
    axes[2].plot(t, p_compressor, label="p_comp", **plot_styles[1])
    axes[2].plot(t, p_bat, label="p_bat", **plot_styles[2])
    axes[2].set_ylabel("Power[W]")
    axes[2].legend(loc="upper left")
    axes[2].grid(True)
    axes[2].set_xlabel("Time [h]")
    ax2 = axes[2].twinx()
    ax2.plot(t, m_dot_cond, label="m_dot_cond", **plot_styles[4])
    ax2.plot(t, m_dot_heating, label="m_dot_heating", **plot_styles[5])
    ax2.set_ylabel("Mass flow rates")
    ax2.legend(loc="upper right")

    # Show the plots
    if show:
        # Save and close plot
        # plt.ion()  # Turn on the interactive mode
        plt.show(block=block)  # Draw the figure
        plt.pause(0.3)  # Time for the figure to load

    if save:
        plt.savefig(f"tmp/frame_{time.time()}.png")


def save_plot(fig, ax, filename):
    fig.savefig(filename, format="svg")
    plt.close(fig)


def save_plots(y, u, i, histories, parameters, title=None, show=True, block=True, save=True):
    # States
    t_cond = y[0] - 273  # K to ºC
    t_tank = y[1] - 273  # K to ºC
    t_floor = y[2] - 273  # K to ºC
    t_room = y[3] - 273  # K to ºC

    # Controls
    m_dot_cond = u[0]
    m_dot_heating = u[1]
    p_compressor = u[2] / 1000  # to kW
    t_amb = u[3] - 273  # to C

    # Design variables
    e_bat = histories["e_bat"][i]
    p_bat = histories["p_bat"][i] / 1000  # to kW

    # Parameters
    solar_size = parameters["SOLAR_SIZE"]
    e_bat_max = parameters["E_BAT_MAX"]
    e_bat_max_kwh = parameters["E_BAT_MAX"] / (1000 * 3600)
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]
    p_required = parameters["p_required"]
    h = parameters["H"]
    t_amb = parameters["t_amb"] - 273  # K to ºC
    n_steps = parameters["t_amb"].shape[0]
    t = np.linspace(0, n_steps * h, n_steps)
    t = t / 3600
    pvpc_prices = parameters["pvpc_prices"] * (1000 * 3600)  # $/(Ws) to $/(kWh)
    excess_prices = parameters["excess_prices"] * (1000 * 3600)  # $/(Ws) to $/(kWh)
    p_required = parameters["p_required"] / 1000  # to kW
    p_solar = parameters["w_solar_per_w_installed"] * solar_size / 1000  # to kW
    p_grid = -p_solar + p_compressor + p_bat + p_required

    # Plot: prices
    fig, ax = plt.subplots(figsize=(8.27, 2.4))  # A4: (8.27, 11.69)
    ax.plot(t, pvpc_prices, label="PVPC", **plot_styles[0])
    ax.plot(t, excess_prices, label="Diario", **plot_styles[1])
    ax.set_ylabel(r"€$/(kW \cdot h)$")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_prices.svg")

    # Plot: solar
    fig, ax = plt.subplots(figsize=(8.27, 2.4))  # Half of A4 height
    ax.plot(t, p_solar, label="Solar", **plot_styles[0])
    ax.plot(t, p_required, label="Consumo", **plot_styles[1])
    ax.set_ylabel("Potencia [kW]")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_generated_consumed.svg")

    # Plot: Temperatures
    fig, ax = plt.subplots(figsize=(8.27, 2.4))
    ax.plot(t, t_cond, label="Salida bomba calor", **plot_styles[0])
    ax.plot(t, t_tank, label="Tanque", **plot_styles[1])
    ax.plot(t, t_floor, label="Suelo", **plot_styles[2])
    ax.plot(t, t_room, label="Habitación", **plot_styles[3])
    ax.plot(t, t_amb, label="Ambiente", **plot_styles[4])
    ax.set_ylabel("Temperatura [ºC]")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_temperatures.svg")

    # Plot: battery energy
    fig, ax = plt.subplots(figsize=(8.27, 2.4))
    ax.plot(t, e_bat / e_bat_max, label=f"{e_bat_max_kwh:.2f}kWh", **plot_styles[0])
    ax.set_ylabel("SOC Batería")
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_battery_soc.svg")

    # Plot: Controls
    fig, ax = plt.subplots(figsize=(8.27, 2.4))
    ax.plot(t, p_grid, label="Red", **plot_styles[0])
    ax.plot(t, p_compressor, label="Compresor", **plot_styles[1])
    ax.plot(t, p_bat, label="Batería", **plot_styles[2])
    ax.set_ylabel("Potencia [kW]")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xlabel("Tiempo [h]")
    # ax_right = ax.twinx()
    # ax_right.plot(t, m_dot_cond, label="m_dot_cond", **plot_styles[4])
    # ax_right.plot(t, m_dot_heating, label="m_dot_heating", **plot_styles[5])
    # ax_right.set_ylabel("Mass Flow Rates")
    # ax_right.legend(loc="upper right", fontsize=8)
    save_plot(fig, ax, f"saves/plot_{title}_controls.svg")


def get_costs(histories, y, u, parameters):
    # States
    t_room = y[3]

    # Design variables
    p_compressor = histories["p_compressor"][-1]
    p_bat = histories["p_bat"][-1]

    # Parameters
    solar_size = parameters["SOLAR_SIZE"]
    p_grid_max = parameters["P_GRID_MAX"]
    e_bat_max = parameters["E_BAT_MAX"]
    tank_volume = parameters["TANK_VOLUME"]
    p_compressor_max = parameters["P_COMPRESSOR_MAX"]
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]
    p_required = parameters["p_required"]
    t_target = parameters["T_TARGET"]

    h = np.ones(p_compressor.shape[0]) * parameters["H"]
    p_solar = parameters["w_solar_per_w_installed"] * solar_size
    p_grid = -p_solar + p_compressor + p_bat + p_required

    return {
        "variable_energy_cost": np.sum(h * np.maximum(pvpc_prices * p_grid, excess_prices * p_grid)),
        "fixed_energy_cost": np.sum(h * get_fixed_energy_cost_by_second(p_grid_max)),
        "battery drep": np.sum(h * np.abs(p_bat) * get_battery_depreciation_by_joule(e_bat_max)),
        "solar drep": np.sum(h * get_solar_panels_depreciation_by_second(solar_size)),
        "HP drep": np.sum(h * np.abs(p_compressor) * get_hp_depreciation_by_joule(p_compressor_max)),
        "tank drep": np.sum(h * get_tank_depreciation_by_second(tank_volume)),
        "T penalization": np.sum(1e-4 * jnp.square(t_room - t_target)),
    }


def plot_history(hist, only_last=True):
    # Get inputs
    storeHistory = History(hist)
    histories = storeHistory.getValues()

    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    t0 = PARAMS["T0"]

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["excess_prices"] = dynamic_parameters["excess_prices"]
    parameters["pvpc_prices"] = dynamic_parameters["pvpc_prices"]
    parameters["excess_prices"] = dynamic_parameters["excess_prices"]
    n_steps = parameters["t_amb"].shape[0]
    y0 = np.array(list(Y0.values())[:4])  # get only state temperatures for dae
    parameters["y0"] = y0

    if only_last:
        indices = [-1]  # Only take the last index
    else:
        # Loop through every x opt results
        x = 10
        indices = list(range(0, len(histories["p_compressor"]), x))

    # loop through histories
    for iter, i in enumerate(indices):
        u = np.zeros((4, n_steps))
        u[0] = histories["m_dot_cond"][i]
        u[1] = histories["m_dot_heating"][i]
        u[2] = histories["p_compressor"][i]
        u[3] = parameters["t_amb"]

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
                parameters["TANK_VOLUME"] * parameters["RHO_WATER"],  # tank mass [kg]
                parameters["U_TANK"],
                6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3),  # tank area [m2]
            ]
        )

        y = dae_forward(y0, u, dae_p, h, n_steps)
        print("y[1]: ", y[:, 1])
        print("u[1]: ", u[:, 1])
        # print("solution:", y[:, -1])
        if only_last:
            title = hist.replace(".hst", "")
            title = title.replace("saves/", "")
            save_plots(y, u, i, histories, parameters, title=title, save=False, show=True)
            # plot_full(y, u, i, histories, parameters, save=False, show=True)
            # plot_thermals(y, u, n_steps, dae_p, parameters, save=False)
            # save_simulation_plots(y, u, n_steps, dae_p, parameters)

            # Print statistics
            costs = get_costs(histories, y, u, parameters)
            print(costs)
            return
        else:
            title = f"iter: {iter}/{len(indices)}"
            plot_full(y, u, i, histories, parameters, title=title, show=True)

    # create animation with pictures from tmp folder
    filename_gif = hist.replace(".hst", ".gif")
    plot_film(filename_gif)


def main():
    # Get inputs
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    t0 = PARAMS["T0"]

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["excess_prices"] = dynamic_parameters["excess_prices"]
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    n_steps = parameters["t_amb"].shape[0]

    # t_cond        = y[0]
    # t_tank        = y[1]
    # t_floor       = y[2]
    # t_room        = y[3]
    y0 = np.array(list(Y0.values())[:4])  # get only state temperatures for dae
    parameters["y0"] = y0

    # m_dot_cond    = u[0]
    # m_dot_heating = u[1]
    # p_compressor  = u[2]
    # t_amb         = u[3]
    u = np.zeros((4, n_steps))

    # m_dot_cond
    m_dot_cond = np.ones((n_steps)) * 0.3  # kg/s
    # m_dot_cond = np.ones((n_steps)) * 1  # kg/s
    # m_dot_cond[-int(n_steps / 4) :] = 1e-3
    m_dot_cond[-int(n_steps / 4) :] = 0.1
    m_dot_cond[-int(n_steps / 6) :] = 0.3

    # m_dot_heating
    # m_dot_heating = np.ones((n_steps)) * 1e-4  # kg/s
    m_dot_heating = np.ones((n_steps)) * 0.1  # kg/s
    m_dot_heating[: -int(n_steps / 2)] = 0.2
    m_dot_heating[-int(n_steps / 5) :] = 0.2

    # m_dot_heating = np.ones((n_steps)) * 1e-3  # kg/s
    # m_dot_heating[-int(n_steps / 3) :] = 0.1

    # p_compressor
    # p_compressor = np.ones((n_steps)) * parameters["P_COMPRESSOR_MAX"]
    p_compressor = np.ones((n_steps)) * 300
    p_compressor[-int(n_steps / 3) :] = 1e-6
    p_compressor[-int(n_steps / 4) :] = parameters["P_COMPRESSOR_MAX"] / 10
    # P_comp[-int(n_steps / 2) :] = parameters["P_COMPRESSOR_MAX"]

    u[0, :] = m_dot_cond
    u[1, :] = m_dot_heating
    u[2, :] = p_compressor
    u[3, :] = parameters["t_amb"]

    # floor_mass                      =         p[0]
    # cp_concrete                     =         p[1]
    # gravity_acceleration            =         p[2]
    # air_volumetric_expansion_coeff  =         p[3]
    # floor_width                     =         p[4]
    # nu_air                          =         p[5]
    # Pr_air                          =         p[6]
    # k_air                           =         p[7]
    # tube_inner_diameter             =         p[8]
    # floor_area                      =         p[9]
    # stefan_boltzmann_constant       =         p[10]
    # epsilon_concrete                =         p[11]
    # cp_water                        =         p[12]
    # mu_water_at_320K                =         p[13]
    # Pr_water                        =         p[14]
    # k_water                         =         p[15]
    # k_pex                           =         p[16]
    # tube_thickness                  =         p[17]
    # A_tubes                         =         p[18]
    # room_air_mass                   =         p[19]
    # cp_air                          =         p[20]
    # A_walls                         =         p[21]
    # A_roof                          =         p[22]
    # A_windows                       =         p[23]
    # U_walls                         =         p[24]
    # U_roof                          =         p[25]
    # U_windows                       =         p[26]
    # m_tank                          =         p[27]
    # U_tank                          =         p[28]
    # A_tank                          =         p[29]
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
            parameters["TANK_VOLUME"] * parameters["RHO_WATER"],  # tank mass [kg]
            parameters["U_TANK"],
            6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3),  # tank area [m2]
        ]
    )

    y = dae_forward(y0, u, dae_p, h, n_steps)
    print("solution:", y[:, -1])
    print(y.shape)
    print("y[2]: ", y[:, 2])
    print("u[2]: ", u[:, 2])
    plot_thermals(y, u, n_steps, dae_p, parameters, save=False)
    save_simulation_plots(y, u, n_steps, dae_p, parameters)

    # j_compressor_cost_extra_args = [parameters["excess_prices"]]
    # j_t_room_min_extra_args = []
    # # FD derivatives
    # dCdy_0_fd, dCdp_fd, dCdu_fd = fd_gradients(
    #     y0, u, dae_p, h, n_steps, j_compressor_cost, j_compressor_cost_extra_args
    # )
    # # dCdy_0_fd, dCdp_fd, dCdu_fd = fd_gradients(y0, u, dae_p, h, n_steps, j_t_room_min, j_t_room_min_extra_args)
    # print("(finite diff) dC/dy0", dCdy_0_fd)
    # print("(finite diff) dC/dp: ", dCdp_fd)
    # print("(finite diff) dC/du_3: ", dCdu_fd)
    #
    # # Adjoint derivatives
    # dCdy_0_adj, dCdp_adj, dCdu_adj = dae_adjoints(
    #     y, u, dae_p, h, n_steps, dae_system, j_compressor_cost, j_compressor_cost_extra_args
    # )
    # # dCdy_0_adj, dCdp_adj, dCdu_adj = dae_adjoints(
    # #     y, u, dae_p, h, n_steps, dae_system, j_t_room_min, j_t_room_min_extra_args
    # # )
    # print("(adjoint) dC/dy_0: ", dCdy_0_adj)
    # print("(adjoint) dC/dp: ", dCdp_adj)
    # print("(adjoint) dC/du_3: ", dCdu_adj[:, 3])
    #
    # # Discrepancies
    # print(f"Discrepancy dC/dy_0: {np.abs(dCdy_0_adj - dCdy_0_fd) / (np.abs(dCdy_0_fd) + 1e-9) * 100}%")
    # print(f"Discrepancy dC/dp: {np.abs(dCdp_adj - dCdp_fd) / (np.abs(dCdp_fd) + 1e-9) * 100}%")
    # print(f"Discrepancy dC/du: {np.abs(dCdu_adj[:, 3] - dCdu_fd) / (np.abs(dCdu_fd) + 1e-9) * 100}%")


if __name__ == "__main__":
    # main()
    plot_history(hist="saves/control_adjoints_regulated.hst", only_last=True)
    # plot_history(hist="saves/control_sand_regulated.hst", only_last=True)
