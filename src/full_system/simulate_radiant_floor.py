import time
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


def get_t_load(y, parameters, i):
    # T_load = T_tank - load_hx_eff * (T_tank - T_target)
    t_tank = y[0]
    t_target = parameters["T_TARGET"]
    load_hx_eff = parameters["LOAD_HX_EFF"]
    return t_tank - load_hx_eff * (t_tank - t_target)


def dae_system(y, y_prev, p, u, h):
    r"""
    Solve the following system of non-linear equations

    ∘ floor_mass * cp_concrete * (dT_floor/dt)
        = Q_load - Q_convection - Q_radiation

    ∘ Q_convection = h_air * floor_area * (T_floor - T_target)
        ∘ Nu_air = (h_air * floor_width) / k_air
        ∘ Nu_air = 0.15 * Ra_air**(1/3)
        ∘ Ra_air = Gr_air * Pr_air
        ∘ Gr_air = (gravity_acceleration * air_volumetric_expansion_coeff * (T_floor - T_target) * floor_width**3) / nu_air**2

    ∘ Q_radiation = stefan_boltzmann_constant * epsilon_concrete * floor_area * (T_floor**4 - T_target**4)

    ∘ Q_load = m_dot_load * cp_water * (T_tank - T_load)
    ∘ Q_load = U_tubes * A_tubes * LMTD_tubes
        ∘ LMTD_tubes = ((T_tank - T_floor) - (T_load - T_floor)) / (ln((T_tank - T_floor) / (T_load - T_floor)))
        ∘ U_tubes = 1 / ((1 / h_water) + (1 / (k_pex / tube_thickness)))
        ∘ Nu_water = (h_water * tube_inner_diameter) / k_water
        ∘ Nu_water = 0.023 * Re_water**0.8 * Pr_water**0.3
        ∘ Re_water = (tube_inner_diameter * v * rho_water) / mu_water = (4 * m_dot_load) / (pi * mu_water * tube_inner_diameter)

    I can merge the first 3 equations into a single one

    ∘ floor_mass * cp_concrete * (dT_floor/dt)
        - Q_load
        + h_air * floor_area * (T_floor - T_target)
        + stefan_boltzmann_constant * epsilon_concrete * floor_area * (T_floor**4 - T_target**4)
        = 0
        -----------------
        ∘ Nu_air = (h_air * floor_width) / k_air
        ∘ Nu_air = 0.15 * Ra_air**(1/3)
        ∘ Ra_air = Gr_air * Pr_air
        ∘ Gr_air = (gravity_acceleration * air_volumetric_expansion_coeff * (T_floor - T_target) * floor_width**3) / nu_air**2
        -----------------
    ∘ Q_load - m_dot_load * cp_water * (T_tank - T_load) = 0
    ∘ Q_load - U_tubes * A_tubes * LMTD_tubes = 0
        -----------------
        ∘ LMTD_tubes = ((T_tank - T_floor) - (T_load - T_floor)) / (ln((T_tank - T_floor) / (T_load - T_floor)))
        ∘ U_tubes = 1 / ((1 / h_water) + (1 / (k_pex / tube_thickness)))
        ∘ Nu_water = (h_water * tube_inner_diameter) / k_water
        ∘ Nu_water = 0.023 * Re_water**0.8 * Pr_water**0.3
        ∘ Re_water = (4 * m_dot_load) / (pi * mu_water * tube_inner_diameter)
        -----------------

    for the first equation, I can insert the value for h_air
    h_air = (0.15 * (((gravity_acceleration * air_volumetric_expansion_coeff * (T_floor - T_target) * floor_width**3) / nu_air**2) * Pr_air)**(1/3)) * k_air / floor_width

    and for the third equation, the U_tubes value
    U_tubes = 1 / ((1 / ((0.023 * ((4 * m_dot_load) / (pi * mu_water * tube_inner_diameter))**0.8 * Pr_water**0.3) * k_water / tube_inner_diameter)) + (1 / (k_pex / tube_thickness)))

    so the equations become
    ∘ floor_mass * cp_concrete * (dT_floor/dt)
        - Q_load
        + ((0.15 * (((gravity_acceleration * air_volumetric_expansion_coeff * (T_floor - T_target) * floor_width**3) / nu_air**2) * Pr_air)**(1/3)) * k_air / floor_width) * floor_area * (T_floor - T_target)
        + stefan_boltzmann_constant * epsilon_concrete * floor_area * (T_floor**4 - T_target**4)
        = 0
    ∘ Q_load - m_dot_load * cp_water * (T_tank - T_load) = 0
    ∘ Q_load - (1 / ((1 / ((0.023 * ((4 * m_dot_load) / (pi * mu_water * tube_inner_diameter))**0.8 * Pr_water**0.3) * k_water / tube_inner_diameter)) + (1 / (k_pex / tube_thickness)))) * A_tubes * LMTD_tubes = 0

    with 3 equations for 3 unknowns:
    T_floor, T_load, Q_load

    Discretized with Backward Euler
    ∘ floor_mass * cp_concrete * ((T_floor - T_floor_prev)/h)
        - Q_load
        + (
            (0.15 * (((gravity_acceleration * air_volumetric_expansion_coeff * (T_floor - T_target) * floor_width**3) / nu_air**2) * Pr_air)**(1/3)) * k_air / floor_width
        )
        * floor_area
        * (T_floor - T_target)
        + stefan_boltzmann_constant * epsilon_concrete * floor_area * (T_floor**4 - T_target**4)
        = 0
    ∘ Q_load - m_dot_load * cp_water * (T_tank - T_load) = 0
    ∘ Q_load - (
          1 / ((1 / ((0.023 * ((4 * m_dot_load) / (pi * mu_water * tube_inner_diameter))**0.8 * Pr_water**0.3) * k_water / tube_inner_diameter)) + (1 / (k_pex / tube_thickness)))
        )
        * A_tubes
        * ((T_tank - T_floor) - (T_load - T_floor)) / (ln((T_tank - T_floor) / (T_load - T_floor)))
        = 0

    Making:
    y[0] = T_floor
    y[1] = T_load
    y[2] = Q_load

    p[0] = floor_mass
    p[1] = cp_concrete
    p[2] = gravity_acceleration
    p[3] = air_volumetric_expansion_coeff
    p[4] = floor_width
    p[5] = nu_air
    p[6] = Pr_air
    p[7] = k_air
    p[8] = tube_inner_diameter
    p[9] = floor_area
    p[10] = stefan_boltzmann_constant
    p[11] = epsilon_concrete
    p[12] = cp_water
    p[13] = mu_water
    p[14] = Pr_water
    p[15] = k_water
    p[16] = k_pex
    p[17] = tube_thickness
    p[18] = A_tubes
    p[19] = T_target
    p[20] = T_tank

    u[0] = m_dot_load
    """
    f_result = f(y, y_prev, p, u, h)
    g_result = g(y, p, u, h)
    return f_result + g_result


def get_h_air(y, p, u):
    # Gr_air = (gravity_acceleration * air_volumetric_expansion_coeff * (T_floor - T_target) * floor_width**3) / nu_air**2
    # Ra_air = Gr_air * Pr_air
    # Nu_air = 0.15 * Ra_air**(1/3)
    # h_air = Nu_air * k_air / floor_width
    t_floor = y[0]
    gravity_acceleration = p[2]
    air_volumetric_expansion_coeff = p[3]
    floor_width = p[4]
    nu_air = p[5]
    Pr_air = p[6]
    k_air = p[7]
    t_target = p[19]

    Gr_air = (
        (gravity_acceleration * air_volumetric_expansion_coeff * (t_floor - t_target) * floor_width**3)
        / nu_air**2
    )
    Ra_air = Gr_air * Pr_air
    Nu_air = 0.15 * Ra_air**(1/3)
    h_air = Nu_air * k_air / floor_width
    return h_air


def get_h_water(y, p, u):
    # Re_water = (4 * m_dot_load) / (pi * mu_water * tube_inner_diameter)
    # Nu_water = 0.023 * Re_water**0.8 * Pr_water**0.3
    # h_water = Nu_water * k_water / tube_inner_diameter
    m_dot_load = u[0]
    pi = 3.141592
    mu_water = p[13]
    tube_inner_diameter = p[8]
    Pr_water = p[14]
    k_water = p[15]

    Re_water = (4 * m_dot_load) / (pi * mu_water * tube_inner_diameter)
    Nu_water = 0.023 * Re_water**0.8 * Pr_water**0.3
    h_water = Nu_water * k_water / tube_inner_diameter
    return h_water


def f(y, y_prev, p, u, h):
    # ∘ floor_mass * cp_concrete * ((T_floor - T_floor_prev)/h)
    #     - Q_load
    #     + h_air * floor_area * (T_floor - T_target)
    #     + stefan_boltzmann_constant * epsilon_concrete * floor_area * (T_floor**4 - T_target**4)
    #     = 0
    h_air = get_h_air(y, p, u)
    return [
        p[0] * p[1] * ((y[0] - y_prev[0])/h)
        - y[2]
        + h_air * p[9] * (y[0] - p[19])
        + p[10] * p[11] * p[9] * (y[0]**4 - p[19]**4)
    ]


def g(y, p, u, h):
    # ∘ Q_load - m_dot_load * cp_water * (T_tank - T_load) = 0
    # ∘ Q_load - U_tubes * A_tubes * LMTD_tubes = 0
    # ∘ LMTD_tubes = ((T_tank - T_floor) - (T_load - T_floor)) / log((T_tank - T_floor) / (T_load - T_floor))
    # ∘ U_tubes = 1 / ((1 / h_water) + (1 / (k_pex / tube_thickness)))
    h_water = get_h_water(y, p, u)
    LMTD_tubes = ((p[20] - y[0]) - (y[1] - y[0])) / (np.log((p[20] - y[0]) / (y[1] - y[0])))
    U_tubes = 1 / ((1 / h_water) + (1 / (p[16] / p[17])))
    return [
        y[2] - u[0] * p[12] * (p[20] - y[1]),
        y[2] - U_tubes * p[18] * LMTD_tubes,

    ]


# def f(y, y_prev, p, u, h):
#     # ∘ floor_mass * cp_concrete * ((T_floor - T_floor_prev)/h)
#     #     - Q_load
#     #     + (
#     #         (0.15 * (((gravity_acceleration * air_volumetric_expansion_coeff * (T_floor - T_target) * floor_width**3) / nu_air**2) * Pr_air)**(1/3)) * k_air / floor_width
#     #     )
#     #     * floor_area
#     #     * (T_floor - T_target)
#     #     + stefan_boltzmann_constant * epsilon_concrete * floor_area * (T_floor**4 - T_target**4)
#     #     = 0
#     return [
#         p[0] * p[1] * ((y[0] - y_prev[0]) / h)
#             - y[2]
#             + (
#                (0.15 * (((p[2] * p[3] * (y[0] - p[19]) * p[4]**3) / p[5]**2) * p[6])**(1/3)) * p[7] / p[4]
#             )
#              * p[9]
#              * (y[0] - p[19])
#              + p[10] * p[11] * p[9] * (y[0]**4 - p[19]**4)
#     ]
#
#
# def g(y, p, u, h):
#     # ∘ Q_load - m_dot_load * cp_water * (T_tank - T_load) = 0
#     # ∘ Q_load - (
#     #       1 / ((1 / ((0.023 * ((4 * m_dot_load) / (pi * mu_water * tube_inner_diameter))**0.8 * Pr_water**0.3) * k_water / tube_inner_diameter)) + (1 / (k_pex / tube_thickness)))
#     #     )
#     #     * A_tubes
#     #     * (((T_tank - T_floor) - (T_load - T_floor)) / (ln((T_tank - T_floor) / (T_load - T_floor))))
#     #     = 0
#     # print("p[20], t_tank: ", p[20])
#     # print("y[0], t_floor: ", y[0])
#     # print("y[1], t_load: ", y[1])
#     # print("y[2], q_load: ", y[2])
#     return [
#         y[2] - u[0] * p[12] * (p[20] - y[1]),
#         y[2] - (
#               1 / ((1 / ((0.023 * ((4 * u[0]) / (np.pi * p[13] * p[8]))**0.8 * p[14]**0.3) * p[15] / p[8])) + (1 / (p[16] / p[17])))
#             )
#             * p[18]
#             * ((p[20] - y[0]) - (y[1] - y[0])) / (np.log((p[20] - y[0]) / (y[1] - y[0] + 1e-6)))
#     ]


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
    t_target = parameters["T_TARGET"]
    t_tank = parameters["T_TANK"]
    floor_area = parameters["FLOOR_AREA"]
    stefan_boltzmann_constant = parameters["STEFAN_BOLTZMANN_CONSTANT"]
    epsilon_concrete = parameters["EPSILON_CONCRETE"]

    m_dot_load = u[0]
    t_floor = y[0]
    t_load = y[1]
    q_load = y[2]
    # Q_convection = h_air * floor_area * (T_floor - T_target)
    h_air = get_h_air(y, dae_p, u)
    q_convection = h_air * floor_area * (t_floor - t_target)
    # Q_radiation = stefan_boltzmann_constant * epsilon_concrete * floor_area * (T_floor**4 - T_target**4)
    q_radiation = stefan_boltzmann_constant * epsilon_concrete * floor_area * (t_floor**4 - t_target**4)

    # Create time array
    t = np.linspace(0, n_steps * h, n_steps) / 3600  # seconds to hours
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].plot(t, t_floor, label="t_floor", **plot_styles[0])
    axes[0].plot(t, np.full_like(t_floor, t_target), label="t_target", **plot_styles[1])
    axes[0].plot(t, t_load, label="t_load", **plot_styles[2])
    axes[0].plot(t, np.full_like(t_floor, t_tank), label="t_tank", **plot_styles[3])
    axes[0].legend(loc="upper left")

    axes[1].plot(t, q_load, label="q_load", **plot_styles[0])
    axes[1].plot(t, q_convection, label="q_convection", **plot_styles[1])
    axes[1].plot(t, q_radiation, label="q_radiation", **plot_styles[2])
    axes[1].plot(t, q_convection + q_radiation, label="q_rad+conv,", **plot_styles[3])
    axes[1].legend()
    axes[2].plot(t, m_dot_load, label="m_dot_load", **plot_styles[0])
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

    # y[0] = T_floor
    # y[1] = T_load
    # y[2] = Q_load
    # Be sure that: T_tank > T_load > T_floor > T_target
    y0 = np.array([
        299.99924603,
        319.80780317,
        2413.60784287
    ])

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["daily_prices"] = dynamic_parameters["daily_prices"]
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["y0"] = y0
    n_steps = parameters["t_amb"].shape[0]


    # u[0] = m_dot_load
    u = np.zeros((1, n_steps))

    m_dot_load = np.ones((n_steps)) * 1e-4  # kg/s
    m_dot_load[-int(n_steps / 3) :] = 5
    m_dot_load[:-int(n_steps / 3)] = 0.1

    # m_dot_load = np.ones((n_steps)) * 3  # kg/s

    u[0, :] = m_dot_load

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
    # p[13] = mu_water
    # p[14] = Pr_water
    # p[15] = k_water
    # p[16] = k_pex
    # p[17] = tube_thickness
    # p[18] = A_tubes
    # p[19] = T_target
    # p[20] = T_tank
    parameters["T_TANK"] = 320  # 47C
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
            parameters["MU_WATER"],
            parameters["PR_WATER"],
            parameters["K_WATER"],
            parameters["K_PEX"],
            parameters["TUBE_THICKNESS"],
            parameters["A_TUBES"],
            parameters["T_TARGET"],
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
