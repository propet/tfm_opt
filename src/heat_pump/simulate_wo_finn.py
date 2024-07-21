import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation
from parameters import PARAMS
from utils import get_dynamic_parameters, plot_styles, jax_to_numpy
import jax
import jax.numpy as jnp
from jax import jit, lax

jax.config.update("jax_enable_x64", True)


"""
Here we only simulate the water tank side, but
now assume that the COP is variable:
From 3 at 273K or lower, to 0 at the maximum condenser temperature of 70º.

Considerations:
    effectiveness of hx's is constant, no matter the delta in temperature
"""


def cop(T):
    if isinstance(T, jnp.ndarray):
        return cop_jax(T)
    else:
        return cop_np(T)


def cop_jax(T):
    conditions = [T < 273, (T >= 273) & (T < 343), T >= 343]
    choices = [3.0, 14.7 - (3.0 / 70.0) * T, 0.0]
    return jnp.select(conditions, choices)


def cop_np(T):
    """
    piecewise lineal -> non-linear function overall

    COP is 0 for T > 343
    COP is 3 for T < 273
    COP varies linearly from 3 at 273K to 0 at 343K
    """
    conditions = [T < 273, (273 <= T) & (T <= 343), T > 343]
    functions = [lambda T: 3, lambda T: 14.7 - (3 / 70) * T, lambda T: 0]
    return np.piecewise(T, conditions, functions)


def get_dcopdT(T):
    """
    COP is 0 for T > 343
    COP is 3 for T < 273
    COP varies linearly from 3 at 273K to 0 at 343K
    """
    conditions = [T < 273, (273 <= T) & (T <= 343), T > 343]
    functions = [lambda T: 0, lambda T: -3 / 70, lambda T: 0]
    return np.piecewise(T, conditions, functions)


# def cost_function(y, u, parameters, h):
#     return np.sum(r(y, u, parameters, h))


def cost_function(y, u, parameters, h):
    cost = 0
    for i in range(y.shape[1]):
        cost += r(
            y[:, i],
            u[:, i],
            h,
            parameters["cost_grid"][i],
            parameters["q_dot_required"][i],
            parameters["t_amb"][i],
            parameters["LOAD_HX_EFF"],
            parameters["CP_WATER"],
        )
    return cost


def r(y, u, h, cost_grid, q_dot_required, t_amb, load_hx_eff, cp_water):
    p_compressor = u[0]
    p_heat = get_p_heat(y, u, q_dot_required, t_amb, load_hx_eff, cp_water)
    r = h * cost_grid * (p_compressor + p_heat)
    return r


def get_p_heat(y, u, q_dot_required, t_amb, load_hx_eff, cp_water):
    t_tank = y[0]
    m_dot_load = u[2]
    return q_dot_required - load_hx_eff * m_dot_load * cp_water * (t_tank - t_amb)


def get_t_load(y, parameters, i):
    t_tank = y[0]
    t_amb = parameters["t_amb"][i]
    load_hx_eff = parameters["LOAD_HX_EFF"]
    return t_tank - load_hx_eff * (t_tank - t_amb)


def dae_system(y, y_prev, p, u, h):
    r"""
    Solve the following system of non-linear equations

    ∘ m_tank * cp_water * (dT_tank/dt)
        = m_dot_cond * cp_water * T_cond
        + m_dot_load * cp_water * T_load
        - m_dot_tank * cp_water * T_tank
        - Q_dot_loss
    ∘ m_dot_tank = m_dot_cond + m_dot_load
    ∘ COP = cop(T_cond)
    ∘ Q_dot_cond = COP * P_comp # W
    ∘ Q_dot_cond = m_dot_cond * cp_water * (T_cond - T_tank)
    ∘ Q_dot_loss = U * A * (T_tank - T_amb)
    ∘ Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)
    ∘ Q_dot_load = m_dot_load * cp_water * (T_tank - T_load)
    ∘ q_dot_required = Q_dot_load + P_heat


    with unknowns:
    T_tank, T_load, T_cond, P_heat


    Discretized with Backward Euler
    ∘ m_tank * cp_water * ((T_tank - T_tank_prev)/h)
        - m_dot_cond * cp_water * T_cond
        - m_dot_load * cp_water * T_load
        + (m_dot_cond + m_dot_load) * cp_water * T_tank
        + U * A * (T_tank - T_amb)
        = 0
    ∘ cop(T_cond) * P_comp - m_dot_cond * cp_water * (T_cond - T_tank) = 0
    ∘ (q_dot_required - P_heat) - load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb) = 0
    ∘ (q_dot_required - P_heat) - m_dot_load * cp_water * (T_tank - T_load) = 0

    From the equation
    ∘ (q_dot_required - P_heat) - load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb) = 0
    I can isolate P_heat:
    ∘ P_heat = q_dot_required - load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)
    And plug it into the equation ∘ (q_dot_required - P_heat) - m_dot_load * cp_water * (T_tank - T_load) = 0
    To isolate T_load:
    T_load = T_tank - load_hx_eff * (T_tank - T_amb)

    I can plug T_load into the first equation
    ∘ m_tank * cp_water * ((T_tank - T_tank_prev)/h)
        - m_dot_cond * cp_water * T_cond
        - m_dot_load * cp_water * T_load
        + (m_dot_cond + m_dot_load) * cp_water * T_tank
        + U * A * (T_tank - T_amb)
        = 0
    to get
    ∘ m_tank * cp_water * ((T_tank - T_tank_prev)/h)
        - m_dot_cond * cp_water * T_cond
        - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_amb))
        + (m_dot_cond + m_dot_load) * cp_water * T_tank
        + U * A * (T_tank - T_amb)
        = 0

    So now I have a decoupled system, which consist of a dae of two equations:
    ∘ m_tank * cp_water * ((T_tank - T_tank_prev)/h)
        - m_dot_cond * cp_water * T_cond
        - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_amb))
        + (m_dot_cond + m_dot_load) * cp_water * T_tank
        + U * A * (T_tank - T_amb)
        = 0
    ∘ cop(T_cond) * P_comp - m_dot_cond * cp_water * (T_cond - T_tank) = 0

    and then another two trivial algebraic equations to compute P_heat and T_load:
    ∘ P_heat = q_dot_required - load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)
    ∘ T_load = T_tank - load_hx_eff * (T_tank - T_amb)


    Making:
    y[0] = T_tank
    y[1] = T_cond

    p[0] = cp_water
    p[1] = m_tank
    p[2] = U
    p[3] = A
    p[4] = T_amb
    p[5] = load_hx_eff

    u[0] = P_comp
    u[1] = m_dot_cond
    u[2] = m_dot_load

    """
    f_result = f(y, y_prev, p, u, h)
    g_result = g(y, p, u, h)
    return f_result + g_result


def f(y, y_prev, p, u, h):
    # m_tank * cp_water * ((T_tank - T_tank_prev)/h)
    #     - m_dot_cond * cp_water * T_cond
    #     - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_amb))
    #     + (m_dot_cond + m_dot_load) * cp_water * T_tank
    #     + U * A * (T_tank - T_amb)
    #     = 0
    return [
        p[1] * p[0] * ((y[0] - y_prev[0]) / h)
        - u[1] * p[0] * y[1]
        - u[2] * p[0] * (y[0] - p[5] * (y[0] - p[4]))
        + (u[1] + u[2]) * p[0] * y[0]
        + p[2] * p[3] * (y[0] - p[4])
    ]


def g(y, p, u, h):
    # cop(T_cond) * P_comp - m_dot_cond * cp_water * (T_cond - T_tank) = 0
    return [cop(y[1]) * u[0] - u[1] * p[0] * (y[1] - y[0])]


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


def adjoint_gradients(y, u, p, h, parameters, n_steps):
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
            parameters["cost_grid"][n],
            parameters["q_dot_required"][n],
            parameters["t_amb"][n],
            parameters["LOAD_HX_EFF"],
            parameters["CP_WATER"],
        )
        drdu_n = get_drdu(
            y_current,
            u_current,
            h,
            parameters["cost_grid"][n],
            parameters["q_dot_required"][n],
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


def dae_adjoints(y, u, dae_p, parameters, n_steps):
    h = PARAMS["H"]
    dCdy_0_adj, dCdp_adj, dCdu_adj = adjoint_gradients(y, u, dae_p, h, parameters, n_steps)
    return dCdy_0_adj, dCdp_adj, dCdu_adj


def plot(y, u, n_steps, h, parameters):
    q_dot_required = parameters["q_dot_required"]

    # Create time array
    t = np.linspace(0, n_steps * h, n_steps)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    T_load = np.zeros(y.shape[1])
    P_heat = np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        T_load[i] = get_t_load(y[:, i], parameters, i)
        P_heat[i] = get_p_heat(
            y[:, i],
            u[:, i],
            parameters["q_dot_required"][i],
            parameters["t_amb"][i],
            parameters["LOAD_HX_EFF"],
            parameters["CP_WATER"],
        )

    Q_dot_load = q_dot_required - P_heat

    # First subplot for T_tank, T_load, and T_cond
    axes[0].plot(t, y[0], label="T_tank", **plot_styles[0])
    axes[0].plot(t, y[1], label="T_cond", **plot_styles[1])
    axes[0].plot(t, T_load, label="T_load", **plot_styles[2])
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Temperature Profiles")
    # axes[0].legend()
    axes[0].legend(loc="upper left")
    axes[0].grid(True)

    ax0 = axes[0].twinx()
    ax0.plot(t, Q_dot_load, label="Q_dot_load", **plot_styles[3])
    ax0.plot(t, P_heat, label="P_heat", **plot_styles[4])
    ax0.set_ylabel("W")
    ax0.legend(loc="upper right")

    axes[1].plot(t, u[0], label="P_comp", **plot_styles[0])
    axes[1].plot(t, q_dot_required, label="q_dot_required", **plot_styles[1])
    axes[1].set_ylabel("Power[W]")
    axes[1].set_title("Control Variables")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

    ax1 = axes[1].twinx()
    ax1.plot(t, u[1], label="m_dot_cond", **plot_styles[2])
    ax1.plot(t, u[2], label="m_dot_load", **plot_styles[3])
    ax1.set_ylabel("Mass flow rates")
    ax1.legend(loc="upper right")

    # Set common x-axis label
    axes[1].set_xlabel("Time (s)")

    # Show the plots
    plt.tight_layout()
    plt.show()


def fd_gradients(y0, u, dae_p, parameters, h, n_steps):
    """
    Finite difference of function cost_function
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
        dfdy0.append((cost_function(y_perturbed, u, parameters, h) - cost_function(y, u, parameters, h)) / delta)

    dfdp = []
    for i in range(len(dae_p)):
        p_perturbed = dae_p.copy()  # Create a new copy of p
        p_perturbed[i] += delta
        y_perturbed = solve(y0, u, p_perturbed, h, n_steps)
        dfdp.append((cost_function(y_perturbed, u, parameters, h) - cost_function(y, u, parameters, h)) / delta)

    dfdu_3 = []
    for i in range(len(u[:, 0])):
        u_perturbed = u.copy()  # Create a new copy of u
        u_perturbed[i, 3] += delta
        y_perturbed = solve(y0, u_perturbed, dae_p, h, n_steps)
        dfdu_3.append(
            (cost_function(y_perturbed, u_perturbed, parameters, h) - cost_function(y, u, parameters, h)) / delta
        )

    return dfdy0, dfdp, dfdu_3


def main():
    # Get inputs
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    n_steps = int(horizon / h)
    t0 = 0
    y0 = np.array([298.34089176, 309.70395426])  # T_tank, T_cond

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["cost_grid"] = dynamic_parameters["cost_grid"]
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["p_solar_gen"] = dynamic_parameters["p_solar_gen"]
    parameters["p_solar_gen"] = dynamic_parameters["p_solar_gen"]
    parameters["y0"] = y0

    # Prepare DAE inputs
    P_comp = np.ones((n_steps)) * parameters["P_COMPRESSOR_MAX"]
    P_comp[-int(n_steps / 3) :] = 1e-6
    P_comp[-int(n_steps / 4) :] = parameters["P_COMPRESSOR_MAX"]

    m_dot_cond = np.ones((n_steps)) * 0.3  # kg/s
    m_dot_cond[-int(n_steps / 3) :] = 1e-6
    m_dot_cond[-int(n_steps / 6) :] = 0.3

    m_dot_load = np.ones((n_steps)) * 1e-6  # kg/s
    m_dot_load[-int(n_steps / 3) :] = 0.2

    u = np.zeros((3, n_steps))
    u[0, :] = P_comp  # P_comp
    u[1, :] = m_dot_cond
    u[2, :] = m_dot_load

    dae_p = np.array(
        [
            parameters["CP_WATER"],
            parameters["TANK_VOLUME"] * parameters["RHO_WATER"],  # tank mass
            parameters["U"],
            6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3),  # tank surface area (m2)
            parameters["t_amb"][0],  # fixed t_amb for all times
            parameters["LOAD_HX_EFF"],
        ]
    )

    y = dae_forward(y0, u, dae_p, n_steps)
    print("solution:", y[:, -1])
    print("y[1]: ", y[:, 1])
    plot(y, u, n_steps, h, parameters)

    # FD derivatives
    dCdy_0_fd, dCdp_fd, dCdu_fd = fd_gradients(y0, u, dae_p, parameters, h, n_steps)
    print("(finite diff) dC/dy0", dCdy_0_fd)
    print("(finite diff) dC/dp: ", dCdp_fd)
    print("(finite diff) dC/du_3: ", dCdu_fd)

    # # Adjoint derivatives
    dCdy_0_adj, dCdp_adj, dCdu_adj = dae_adjoints(y, u, dae_p, parameters, n_steps)
    print("(adjoint) dC/dy_0: ", dCdy_0_adj)
    print("(adjoint) dC/dp: ", dCdp_adj)
    print("(adjoint) dC/du_3: ", dCdu_adj[:, 3])

    # Discrepancies
    print(f"Discrepancy dC/dy_0: {np.abs(dCdy_0_adj - dCdy_0_fd) / (np.abs(dCdy_0_fd) + 1e-9) * 100}%")
    print(f"Discrepancy dC/dp: {np.abs(dCdp_adj - dCdp_fd) / (np.abs(dCdp_fd) + 1e-9) * 100}%")
    print(f"Discrepancy dC/du: {np.abs(dCdu_adj[:, 3] - dCdu_fd) / (np.abs(dCdu_fd) + 1e-9) * 100}%")


if __name__ == "__main__":
    main()
