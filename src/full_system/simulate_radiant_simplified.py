import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from parameters import PARAMS
from utils import get_dynamic_parameters, plot_styles, jax_to_numpy, plot_film, load_dict_from_file, cop, get_dcopdT
from pyoptsparse import History
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def cost_function(y, u, parameters):
    cost = 0
    for i in range(y.shape[1]):
        cost += r(
            y[:, i],
            u[:, i],
            parameters["H"],
            parameters["daily_prices"][i],
            parameters["t_amb"][i],
            parameters["LOAD_HX_EFF"],
            parameters["CP_WATER"],
        )
    return cost


def r(y, u, h, daily_prices, t_amb, load_hx_eff, cp_water):
    p_compressor = u[0]
    r = h * daily_prices * (p_compressor)
    return r


def get_t_out_heating(y, parameters, i):
    # T_out_heating = T_tank - load_hx_eff * (T_tank - T_room)
    t_tank = y[0]
    t_room = parameters["t_room"]
    load_hx_eff = parameters["LOAD_HX_EFF"]
    return t_tank - load_hx_eff * (t_tank - t_room)


def dae_system(y, y_prev, p, u, h):
    r"""
    Solve the following system of non-linear equations

    ∘ floor_mass * cp_concrete * ((t_floor - t_floor_prev)/h)
        - q_conduction_floor
        + q_convection_floor
        + q_radiation_floor
        = 0

    ∘ room_air_mass * cp_air * ((t_room - t_room_prev)/h)
        - q_convection_floor
        - q_radiation_floor
        + U_walls * A_walls * (t_room - t_amb)
        + U_roof * A_roof * (t_room - t_amb)
        + U_windows * A_windows * (t_room - t_amb)
        = 0

    ∘ q_convection_floor - h_floor_air * floor_area * (t_floor - t_room) = 0
        --------------------------------------
        ∘ Nu_floor_air = (h_floor_air * floor_width) / k_air
        ∘ Nu_floor_air = 0.15 * Ra_floor_air**(1/3)
        ∘ Ra_floor_air = Gr_air * Pr_air
        ∘ Gr_floor_air = (gravity_acceleration * air_volumetric_expansion_coeff * np.abs(t_floor - t_room) * L**3) / nu_air**2

    ∘ q_radiation_floor - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4) = 0

    ∘ q_conduction_floor - m_dot_heating * cp_water * (t_tank - t_out_heating) = 0
    ∘ q_conduction_floor - U_tubes * A_tubes * DeltaT_tubes = 0
        --------------------------------------
        ∘ DeltaT_tubes = ((t_tank - t_floor) - (t_out_heating - t_floor)) / (np.log((t_tank - t_floor) / (t_out_heating - t_floor)))
        ∘ U_tubes = 1 / ((1 / h_tube_water) + (1 / (k_pex / tube_thickness)))
        ∘ Nu_tube_water = (h_tube_water * tube_inner_diameter) / k_water
        ∘ Nu_tube_water = 0.023 * Re_tube_water**0.8 * Pr_water**(1/3)
        ∘ Re_tube_water = (tube_inner_diameter * v * rho_water) / mu_water_at_320K = (4 * m_dot_heating) / (pi * mu_water_at_320K * tube_inner_diameter)

    substitute the values for q_conduction, q_radiation and q_convection in the equations
    q_conduction_floor = m_dot_heating * cp_water * (t_tank - t_out_heating)
    q_convection_floor = h_floor_air * floor_area * (t_floor - t_room)
    q_radiation_floor = stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)

    ∘ m_dot_heating * cp_water * (t_tank - t_out_heating) - U_tubes * A_tubes * DeltaT_tubes = 0

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


    with 3 equations for 3 unknowns:
    t_floor, t_out_heating, t_room

    Making:
    t_out_heating      =  y[0]
    t_floor            =  y[1]
    t_room             =  y[2]

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
    t_tank                          =         p[27]

    m_dot_heating = u[0]
    t_amb         = u[1]
    """
    f_result = f(y, y_prev, p, u, h)
    g_result = g(y, p, u, h)
    return f_result + g_result


def get_h_floor_air(
    t_floor, t_room, gravity_acceleration, air_volumetric_expansion_coeff, floor_width, nu_air, Pr_air, k_air, A_roof
):
    """
    Convective heat transfer coefficient for natural convection over a radiant floor
    """
    # Gr_air = (gravity_acceleration * air_volumetric_expansion_coeff * np.abs(T_floor - T_room) * L**3) / nu_air**2
    # Ra_air = Gr_air * Pr_air
    # Nu_air = 0.15 * Ra_air**(1/3)
    # h_air = Nu_air * k_air / floor_width

    floor_perimeter = 4 * floor_width
    floor_area = A_roof
    L = floor_area / floor_perimeter  # characteristic_length

    # Absolute value for the difference between t_floor and t_room
    # since if negative, would imply a negative h, which doesn't have physical meaning
    Gr_floor_air = (gravity_acceleration * air_volumetric_expansion_coeff * np.abs(t_floor - t_room) * L**3) / nu_air**2
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


# def get_h_tube_water(tube_inner_diameter, mu_water_at_320K, Pr_water, k_water, m_dot_heating):
#     """
#     Assuming that the flow is always in turbulent region
#     Which is not the case
#     So we are overestimating the convective coefficient here
#
#     Using Sieder-Tate correlation for the Nusselt number
#     """
#     # Re_water = (4 * m_dot_heating) / (pi * mu_water_at_320K * tube_inner_diameter)
#     # Nu_water = 0.023 * Re_water**0.8 * Pr_water**0.3
#     # h_tube_water = Nu_water * k_water / tube_inner_diameter
#     pi = 3.141592
#
#     Re_tube_water = (4 * m_dot_heating) / (pi * mu_water_at_320K * tube_inner_diameter)
#     mu_water_at_300K = 0.000866
#     Nu_tube_water = 0.027 * Re_tube_water**0.8 * Pr_water ** (1 / 3) * (mu_water_at_320K / mu_water_at_300K) ** 0.14
#     h_tube_water = Nu_tube_water * k_water / tube_inner_diameter
#     return h_tube_water


def f(y, y_prev, p, u, h):
    # ∘ floor_mass * cp_concrete * ((t_floor - t_floor_prev)/h)
    #     - m_dot_heating * cp_water * (t_tank - t_out_heating)
    #     + h_floor_air * floor_area * (t_floor - t_room)
    #     + stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
    #     = 0
    #
    # ∘ room_air_mass * cp_air * ((t_room - t_room_prev)/h)
    #     - h_floor_air * floor_area * (t_floor - t_room)
    #     - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
    #     + U_walls * A_walls * (t_room - t_amb)
    #     + U_roof * A_roof * (t_room - t_amb)
    #     + U_windows * A_windows * (t_room - t_amb)
    #     = 0
    #
    #     --------------------------------------
    #     ∘ Nu_floor_air = (h_floor_air * floor_width) / k_air
    #     ∘ Nu_floor_air = 0.15 * Ra_floor_air**(1/3)
    #     ∘ Ra_floor_air = Gr_air * Pr_air
    #     ∘ Gr_floor_air = (gravity_acceleration * air_volumetric_expansion_coeff * np.abs(t_floor - t_room) * L**3) / nu_air**2

    t_out_heating = y[0]
    t_floor = y[1]
    t_floor_prev = y_prev[1]
    t_room = y[2]
    t_room_prev = y_prev[2]

    floor_mass = p[0]
    cp_concrete = p[1]
    gravity_acceleration = p[2]
    air_volumetric_expansion_coeff = p[3]
    floor_width = p[4]
    nu_air = p[5]
    Pr_air = p[6]
    k_air = p[7]
    floor_area = p[9]
    stefan_boltzmann_constant = p[10]
    epsilon_concrete = p[11]
    cp_water = p[12]
    room_air_mass = p[19]
    cp_air = p[20]
    A_walls = p[21]
    A_roof = p[22]
    A_windows = p[23]
    U_walls = p[24]
    U_roof = p[25]
    U_windows = p[26]
    t_tank = p[27]

    m_dot_heating = u[0]
    t_amb = u[1]

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

    return [
        floor_mass * cp_concrete * ((t_floor - t_floor_prev) / h)
        - m_dot_heating * cp_water * (t_tank - t_out_heating)
        + h_floor_air * floor_area * (t_floor - t_room)
        + stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4),
        room_air_mass * cp_air * ((t_room - t_room_prev) / h)
        - h_floor_air * floor_area * (t_floor - t_room)
        - stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)
        + U_walls * A_walls * (t_room - t_amb)
        + U_roof * A_roof * (t_room - t_amb)
        + U_windows * A_windows * (t_room - t_amb),
    ]


def g(y, p, u, h):
    # ∘ m_dot_heating * cp_water * (t_tank - t_out_heating) - U_tubes * A_tubes * DeltaT_tubes = 0
    #     --------------------------------------
    #     ∘ DeltaT_tubes = ((t_tank - t_floor) - (t_out_heating - t_floor)) / (np.log((t_tank - t_floor) / (t_out_heating - t_floor)))
    #     ∘ U_tubes = 1 / ((1 / h_tube_water) + (1 / (k_pex / tube_thickness)))
    #     ∘ Nu_tube_water = (h_tube_water * tube_inner_diameter) / k_water
    #     ∘ Nu_tube_water = 0.023 * Re_tube_water**0.8 * Pr_water**0.3
    #     ∘ Re_tube_water = (tube_inner_diameter * v * rho_water) / mu_water_at_320K = (4 * m_dot_heating) / (pi * mu_water_at_320K * tube_inner_diameter)
    t_out_heating = y[0]
    t_floor = y[1]

    tube_inner_diameter = p[8]
    cp_water = p[12]
    mu_water_at_320K = p[13]
    Pr_water = p[14]
    k_water = p[15]
    k_pex = p[16]
    tube_thickness = p[17]
    A_tubes = p[18]
    t_tank = p[27]

    m_dot_heating = u[0]

    h_tube_water = get_h_tube_water(
        tube_inner_diameter,
        mu_water_at_320K,
        Pr_water,
        k_water,
        m_dot_heating,
    )

    DeltaT_tubes = ((t_tank - t_floor) - (t_out_heating - t_floor)) / (
        np.log(np.abs((t_tank - t_floor) / (t_out_heating - t_floor + 1e-6))) + 1e-6
    )  # LMTD with absolute differences
    # DeltaT_tubes = ((t_tank - t_floor) + np.abs(t_out_heating - t_floor)) / 2  # mean with absolutes

    U_tubes = 1 / ((1 / h_tube_water) + (1 / (k_pex / tube_thickness)))

    return [
        m_dot_heating * cp_water * (t_tank - t_out_heating) - U_tubes * A_tubes * DeltaT_tubes,
    ]


def solve(y_0, u, dae_p, h, n_steps):
    # Initialize parameters
    y = np.zeros((len(y_0), n_steps))

    # Initial conditions
    y[:, 0] = y_0

    # Time-stepping loop
    for n in range(1, n_steps):
        # Use fsolve to solve the nonlinear system
        y_n = fsolve(dae_system, y[:, n - 1], args=(y[:, n - 1], dae_p, u[:, n], h))
        y[:, n] = y_n

    return y


def adjoint_gradients(y, u, p, n_steps, parameters):
    r"""
    Adjoint gradients of the cost function with respect to parameters

    min_p C = Σr(y_n, p, u_n)
    s.t.
        f(y_n, y_(n-1), p, u_n) = 0
        g(y_n, p, u_n) = 0

    where f are the differential constraints
    and g are the algebraic constraints

    by augmenting the system
    p' = 0

    min_{p_n} C = Σr(y_n, p_n, u_n)
        f(y_n, y_(n-1), p_n, u_n) = 0
        g(y_n, p_n, u_n) = 0
        (p_n - p_(n-1)) / h = 0

    Lagrangian:
    L = Σr(y_n, p_n, u_n)
        - Σ λ_n f(y_n, y_(n-1), p_n, u_n)
        - Σ ν_n g(y_n, p_n, u_n)
        - Σ μ_n (p_n - p_(n-1))

    ∂L/∂y_n = 0 = ∂r(y_n, p_n, u_n)/∂y_n
                  - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂y_n
                  - λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1), u_(n+1))/∂y_n
                  - ν_n ∂g(y_n, p_n, u_n)/∂y_n
    ∂L/∂p_n = 0 = ∂r(y_n, p_n, u_n)/∂p_n
                  - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂p_n
                  - ν_n ∂g(y_n, p_n, u_n)/∂p_n
                  - μ_n + μ_(n+1)
    ∂L/∂u_n = ∂C/∂u_n = ∂r(y_n, p_n, u_n)/∂u_n
                        - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂u_n
                        - ν_n ∂g(y_n, p_n, u_n)/∂u_n

    Solve for λ_n and μ_n at each step:
    [∂f/∂y_n^T  ∂g/∂y_n^T  0] [λ_n]   [(∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1}, u_{n+1})/∂y_n)^T]
    [∂f/∂p_n^T  ∂g/∂p_n^T  I] [ν_n] = [(∂r/∂p_n + μ_{n+1})^T                               ]
                              [μ_n]
    Terminal conditions:
    [∂f/∂y_N^T  ∂g/∂y_N^T  0] [λ_N]   [(∂r/∂y_N)^T]
    [∂f/∂p_N^T  ∂g/∂p_N^T  I] [ν_N] = [(∂r/∂p_N)^T]
                              [μ_N]
    Solve for initial timestep:
    ∂L/∂y_0 = ∂C/∂y_0 = (∂r/∂y_0 - λ_1 ∂f(y_1, y_0, p_1, u_1)/∂y_0)^T
    ∂L/∂p_0 = ∂C/∂p_0 = (∂r/∂p_0 + μ_1)^T
    """
    h = parameters["H"]

    # Obtain shapes of jacobians
    # JACOBIANS
    # jax.jacobian will automatically
    # use jax.jacrev or jax.jacfwd
    # based on the number of inputs and outputs
    get_drdy_jax = jax.jit(jax.jacobian(r, argnums=0))
    get_drdu_jax = jax.jit(jax.jacobian(r, argnums=1))
    get_dfdy_jax = jax.jit(jax.jacobian(f, argnums=0))
    get_dfdy_prev_jax = jax.jit(jax.jacobian(f, argnums=1))
    get_dfdp_jax = jax.jit(jax.jacobian(f, argnums=2))
    get_dfdu_jax = jax.jit(jax.jacobian(f, argnums=3))
    get_dgdy_jax = jax.jit(jax.jacobian(g, argnums=0))
    get_dgdp_jax = jax.jit(jax.jacobian(g, argnums=1))
    get_dgdu_jax = jax.jit(jax.jacobian(g, argnums=2))

    # Convert JAX jacobian functions to NumPy functions
    get_drdy = jax_to_numpy(get_drdy_jax)
    get_drdu = jax_to_numpy(get_drdu_jax)
    get_dfdy = jax_to_numpy(get_dfdy_jax)
    get_dfdy_prev = jax_to_numpy(get_dfdy_prev_jax)
    get_dfdp = jax_to_numpy(get_dfdp_jax)
    get_dfdu = jax_to_numpy(get_dfdu_jax)
    get_dgdy = jax_to_numpy(get_dgdy_jax)
    get_dgdp = jax_to_numpy(get_dgdp_jax)
    get_dgdu = jax_to_numpy(get_dgdu_jax)

    # Obtain shapes of jacobians
    dfdy = get_dfdy(y[:, 1], y[:, 0], p, u[:, 0], h)
    dgdp = get_dgdp(y[:, 1], p, u[:, 0], h)
    n_odes = dfdy.shape[0]
    n_states = dfdy.shape[1]
    n_algs = dgdp.shape[0]
    n_params = dgdp.shape[1]

    # Initialize adjoint variables
    adj_lambda = np.zeros((n_odes, n_steps + 1))
    adj_mu = np.zeros((n_params, n_steps + 1))
    adj_nu = np.zeros((n_algs, n_steps + 1))
    dCdy_0 = np.zeros(n_states)
    dCdp_0 = np.zeros(n_params)
    dCdu = np.zeros(u.shape)

    # Backward propagation of adjoint variables
    for n in range(n_steps - 1, -1, -1):
        # tic = time.time()
        y_current = y[:, n]
        y_prev = y[:, n - 1]
        u_current = u[:, n]
        dfdy_n = get_dfdy(y_current, y_prev, p, u_current, h)
        dfdu_n = get_dfdu(y_current, y_prev, p, u_current, h)
        dfdp_n = get_dfdp(y_current, y_prev, p, u_current, h)
        dgdy_n = get_dgdy(y_current, p, u_current, h)
        dgdu_n = get_dgdu(y_current, p, u_current, h)
        dgdp_n = get_dgdp(y_current, p, u_current, h)
        drdy_n = get_drdy(
            y_current,
            u_current,
            h,
            parameters["daily_prices"][n],
            parameters["t_amb"][n],
            parameters["LOAD_HX_EFF"],
            parameters["CP_WATER"],
        )
        drdu_n = get_drdu(
            y_current,
            u_current,
            h,
            parameters["daily_prices"][n],
            parameters["t_amb"][n],
            parameters["LOAD_HX_EFF"],
            parameters["CP_WATER"],
        )
        drdp_n = np.zeros(n_params)

        if n == n_steps - 1:
            # Terminal condition
            # [∂f/∂y_n^T  ∂g/∂y_n^T  0] [λ_n]   [(∂r/∂y_n)^T]
            # [∂f/∂p_n^T  ∂g/∂p_n^T  I] [ν_n] = [(∂r/∂p_n)^T]
            #                           [μ_n]
            A = np.block([[dfdy_n.T, dgdy_n.T, np.zeros((n_states, n_params))], [dfdp_n.T, dgdp_n.T, np.eye(n_params)]])
            b = np.concatenate([drdy_n, drdp_n])
            adjs = np.linalg.solve(A, b)
            adj_lambda[:, n] = adjs[:n_odes]
            adj_nu[:, n] = adjs[n_odes : (n_odes + n_algs)]
            adj_mu[:, n] = adjs[(n_odes + n_algs) :]
        elif n == 0:
            # Initial timestep
            # ∂L/∂y_0 = ∂C/∂y_0 = (∂r/∂y_0 - λ_1 ∂f(y_1, y_0, p_1, u_1)/∂y_0)^T
            # ∂L/∂p_0 = ∂C/∂p_0 = (∂r/∂p_0 + μ_1)^T
            dfdy_prev = get_dfdy_prev(y[:, 1], y[:, 0], p, u[:, 1], h)
            dCdy_0 = drdy_n - np.dot(adj_lambda[:, 1].T, dfdy_prev)
            dCdp_0 = drdp_n + adj_mu[:, 1]
        else:
            # [∂f/∂y_n^T  ∂g/∂y_n^T  0] [λ_n]   [(∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T]
            # [∂f/∂p_n^T  ∂g/∂p_n^T  I] [ν_n] = [(∂r/∂p_n + μ_{n+1})^T                               ]
            #                           [μ_n]
            dfdy_prev = get_dfdy_prev(y[:, n + 1], y[:, n], p, u[:, n + 1], h)
            A = np.block([[dfdy_n.T, dgdy_n.T, np.zeros((n_states, n_params))], [dfdp_n.T, dgdp_n.T, np.eye(n_params)]])
            b = np.concatenate([drdy_n - np.dot(adj_lambda[:, n + 1].T, dfdy_prev), drdp_n + adj_mu[:, n + 1]])
            adjs = np.linalg.solve(A, b)
            adj_lambda[:, n] = adjs[:n_odes]
            adj_nu[:, n] = adjs[n_odes : (n_odes + n_algs)]
            adj_mu[:, n] = adjs[(n_odes + n_algs) :]

        # ∂L/∂u_n = ∂C/∂u_n = ∂r(y_n, p_n, u_n)/∂u_n
        #                     - λ_n ∂f(y_n, y_(n-1), p_n, u_n)/∂u_n
        #                     - ν_n ∂g(y_n, p_n, u_n)/∂u_n
        dCdu[:, n] = drdu_n - adj_lambda[:, n].T @ dfdu_n - adj_nu[:, n].T @ dgdu_n
        # toc = time.time()
        # print(toc - tic)

    return dCdy_0, dCdp_0, dCdu


def dae_forward(y0, u, dae_p, n_steps):
    h = PARAMS["H"]
    y = solve(y0, u, dae_p, h, n_steps)
    return y


def dae_adjoints(y, u, dae_p, n_steps, parameters):
    dCdy_0_adj, dCdp_adj, dCdu_adj = adjoint_gradients(y, u, dae_p, n_steps, parameters)
    return dCdy_0_adj, dCdp_adj, dCdu_adj


fig = None


def plot_only_thermals(y, u, n_steps, dae_p, parameters, title=None, show=True, block=True, save=True):
    global fig
    # Close the previous figure if it exists
    if fig:
        plt.close(fig)

    h = parameters["H"]
    t = np.linspace(0, n_steps * h, n_steps) / 3600  # seconds to hours
    t_tank = parameters["T_TANK"]
    t_out_heating = y[0]
    t_floor = y[1]
    t_room = y[2]

    gravity_acceleration = dae_p[2]
    air_volumetric_expansion_coeff = dae_p[3]
    floor_width = dae_p[4]
    nu_air = dae_p[5]
    Pr_air = dae_p[6]
    k_air = dae_p[7]
    floor_area = dae_p[9]
    stefan_boltzmann_constant = dae_p[10]
    epsilon_concrete = dae_p[11]
    cp_water = dae_p[12]
    A_roof = dae_p[22]
    t_tank = dae_p[27]

    m_dot_heating = u[0]
    t_amb = u[1]

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

    q_conduction_floor = m_dot_heating * cp_water * (t_tank - t_out_heating)
    q_convection_floor = h_floor_air * floor_area * (t_floor - t_room)
    q_radiation_floor = stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_room**4)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].plot(t, t_out_heating, label="t_out_heating", **plot_styles[0])
    axes[0].plot(t, t_floor, label="t_floor", **plot_styles[1])
    axes[0].plot(t, t_room, label="t_room", **plot_styles[2])
    axes[0].plot(t, np.full_like(t_floor, t_tank), label="t_tank", **plot_styles[3])
    axes[0].legend(loc="upper left")

    ax0 = axes[0].twinx()
    ax0.plot(t, t_amb, label="t_amb", **plot_styles[4])
    ax0.legend(loc="upper right")

    axes[1].plot(t, q_conduction_floor, label="q_conduction_floor", **plot_styles[0])
    axes[1].plot(t, q_convection_floor, label="q_convection_floor", **plot_styles[1])
    axes[1].plot(t, q_radiation_floor, label="q_radiation_floor", **plot_styles[2])
    axes[1].plot(t, q_convection_floor + q_radiation_floor, label="q_rad+conv,", **plot_styles[3])
    axes[1].legend()

    axes[2].plot(t, m_dot_heating, label="m_dot_heating", **plot_styles[0])
    axes[2].legend()

    # Show the plots
    if show:
        # Save and close plot
        # plt.ion()  # Turn on the interactive mode
        plt.show(block=block)  # Draw the figure
        plt.pause(0.3)  # Time for the figure to load

    if save:
        plt.savefig(f"tmp/frame_{time.time()}.png")


def main():
    # Get inputs
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    t0 = PARAMS["T0"]

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["daily_prices"] = dynamic_parameters["daily_prices"]
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    n_steps = parameters["t_amb"].shape[0]

    # t_out_heating      =  y[0]
    # t_floor            =  y[1]
    # t_room             =  y[2]
    # Be sure that: T_tank > T_out_heating > T_floor > T_room
    parameters["T_TANK"] = 320  # 47C
    y0 = np.array([311.13048069, 283.79712778, 282.07130183])
    parameters["y0"] = y0

    # m_dot_heating = u[0]
    # t_amb         = u[1]
    u = np.zeros((2, n_steps))

    m_dot_heating = np.ones((n_steps)) * 1e-3  # kg/s
    m_dot_heating[: -int(n_steps / 2)] = 0.1
    m_dot_heating[-int(n_steps / 5) :] = 0.1

    # m_dot_heating = np.ones((n_steps)) * 1e-3  # kg/s
    # m_dot_heating[-int(n_steps / 3) :] = 0.1

    u[0, :] = m_dot_heating
    u[1, :] = parameters["t_amb"]

    # p[0] = floor_mass
    # p[1] = cp_concrete
    # p[2] = gravity_acceleration
    # p[3] = air_volumetric_expansion_coeff
    # p[4] = floor_width
    # p[5] = nu_air
    # p[6] = Pr_air
    # p[7] = k_air
    # p[8] = tube_inner_diameter
    # p[9] = floor_area
    # p[10] = stefan_boltzmann_constant
    # p[11] = epsilon_concrete
    # p[12] = cp_water
    # p[13] = mu_water_at_320K
    # p[14] = Pr_water
    # p[15] = k_water
    # p[16] = k_pex
    # p[17] = tube_thickness
    # p[18] = A_tubes
    # p[19] = room_air_mass
    # p[20] = cp_air
    # p[21] = A_walls
    # p[22] = A_roof
    # p[23] = A_windows
    # p[24] = U_walls
    # p[25] = U_roof
    # p[26] = U_windows
    # p[27] = T_tank
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
            parameters["T_TANK"],
        ]
    )

    y = dae_forward(y0, u, dae_p, n_steps)
    print("solution:", y[:, -1])
    print(y.shape)
    print("y[2]: ", y[:, 2])
    print("u[2]: ", u[:, 2])
    plot_only_thermals(y, u, n_steps, dae_p, parameters, save=False)
    # plot_animation(y, u, n_steps, parameters)

    # # FD derivatives
    # dCdy_0_fd, dCdp_fd, dCdu_fd = fd_gradients(y0, u, dae_p, n_steps, parameters)
    # print("(finite diff) dC/dy0", dCdy_0_fd)
    # print("(finite diff) dC/dp: ", dCdp_fd)
    # print("(finite diff) dC/du_3: ", dCdu_fd)
    #
    # # # Adjoint derivatives
    # dCdy_0_adj, dCdp_adj, dCdu_adj = dae_adjoints(y, u, dae_p, n_steps, parameters)
    # print("(adjoint) dC/dy_0: ", dCdy_0_adj)
    # print("(adjoint) dC/dp: ", dCdp_adj)
    # print("(adjoint) dC/du_3: ", dCdu_adj[:, 3])
    #
    # # Discrepancies
    # print(f"Discrepancy dC/dy_0: {np.abs(dCdy_0_adj - dCdy_0_fd) / (np.abs(dCdy_0_fd) + 1e-9) * 100}%")
    # print(f"Discrepancy dC/dp: {np.abs(dCdp_adj - dCdp_fd) / (np.abs(dCdp_fd) + 1e-9) * 100}%")
    # print(f"Discrepancy dC/du: {np.abs(dCdu_adj[:, 3] - dCdu_fd) / (np.abs(dCdu_fd) + 1e-9) * 100}%")


if __name__ == "__main__":
    main()
