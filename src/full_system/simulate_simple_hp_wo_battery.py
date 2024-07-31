import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from parameters import PARAMS
from utils import get_dynamic_parameters, plot_styles, jax_to_numpy, plot_film, load_dict_from_file
from pyoptsparse import History
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def cop(T):
    if isinstance(T, jnp.ndarray):
        return cop_jax(T)
    else:
        return cop_np(T)


def cop_jax(T):
    """
    piecewise lineal -> non-linear function overall
    Approximate operating range for a heat pump:
        -20C to 40C, with a maximum temperature of 70C

    So we set the COP to 4 up to 40C, and linearly decreasing to 0 at 70C

    COP is 0 for T > 343
    COP is 4 for T < 313
    COP varies linearly from 4 at 313K to 0 at 343K
    """
    max_cop = 4
    min_cop = 0
    T_upper = 343
    T_lower = 313
    m = (max_cop - min_cop) / (T_upper - T_lower)
    T0 = max_cop + m * T_lower
    conditions = [T < T_lower, (T >= T_lower) & (T < T_upper), T >= T_upper]
    choices = [max_cop, T0 - m * T, min_cop]
    return jnp.select(conditions, choices)


def cop_np(T):
    """
    piecewise lineal -> non-linear function overall
    Approximate operating range for a heat pump:
        -20C to 40C, with a maximum temperature of 70C

    So we set the COP to 4 up to 40C, and linearly decreasing to 0 at 70C

    COP is 0 for T > 343
    COP is 4 for T < 313
    COP varies linearly from 4 at 313K to 0 at 343K
    """
    max_cop = 4
    min_cop = 0
    T_upper = 343
    T_lower = 313
    m = (max_cop - min_cop) / (T_upper - T_lower)
    T0 = max_cop + m * T_lower
    conditions = [T < T_lower, (T >= T_lower) & (T < T_upper), T >= T_upper]
    functions = [lambda T: max_cop, lambda T: T0 - m * T, lambda T: min_cop]
    return np.piecewise(T, conditions, functions)


def get_dcopdT(T):
    """
    piecewise lineal -> non-linear function overall
    Approximate operating range for a heat pump:
        -20C to 40C, with a maximum temperature of 70C

    So we set the COP to 4 up to 40C, and linearly decreasing to 0 at 70C

    COP is 0 for T > 343
    COP is 4 for T < 313
    COP varies linearly from 4 at 313K to 0 at 343K
    """
    max_cop = 4
    min_cop = 0
    T_upper = 343
    T_lower = 313
    m = (max_cop - min_cop) / (T_upper - T_lower)
    conditions = [T < T_lower, (T >= T_lower) & (T < T_upper), T >= T_upper]
    functions = [lambda T: 0, lambda T: -m, lambda T: 0]
    return np.piecewise(T, conditions, functions)


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


    with unknowns:
    T_tank, T_load, T_cond


    Discretized with Backward Euler
    ∘ m_tank * cp_water * ((T_tank - T_tank_prev)/h)
        - m_dot_cond * cp_water * T_cond
        - m_dot_load * cp_water * T_load
        + (m_dot_cond + m_dot_load) * cp_water * T_tank
        + U * A * (T_tank - T_amb)
        = 0
    ∘ cop(T_cond) * P_comp - m_dot_cond * cp_water * (T_cond - T_tank) = 0
    ∘ Q_dot_load - load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb) = 0
    ∘ Q_dot_load - m_dot_load * cp_water * (T_tank - T_load) = 0

    From the two last equations for Q_dot_load, I can get T_load a in terms of T_tank
    load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb) = m_dot_load * cp_water * (T_tank - T_load)
    ->
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

    and then to compute T_load:
    ∘ T_load = T_tank - load_hx_eff * (T_tank - T_amb)


    Making:
    y[0] = T_tank
    y[1] = T_cond

    p[0] = cp_water
    p[1] = m_tank
    p[2] = U
    p[3] = A
    p[4] = load_hx_eff

    u[0] = P_comp
    u[1] = m_dot_cond
    u[2] = m_dot_load
    u[3] = t_amb

    """
    f_result = f(y, y_prev, p, u, h)
    g_result = g(y, p, u, h)
    return f_result + g_result


def f(y, y_prev, p, u, h):
    # m_tank * cp_water * ((T_tank - T_tank_prev)/h)
    #   - m_dot_cond * cp_water * T_cond
    #     - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_amb))
    #   + (m_dot_cond + m_dot_load) * cp_water * T_tank
    #   + U * A * (T_tank - T_amb)
    #   = 0
    return [
        p[1] * p[0] * ((y[0] - y_prev[0]) / h)
        - u[1] * p[0] * y[1]
        - u[2] * p[0] * (y[0] - p[4] * (y[0] - u[3]))
        + (u[1] + u[2]) * p[0] * y[0]
        + p[2] * p[3] * (y[0] - u[3])
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
def plot(y, u, n_steps, parameters, title=None, show=True, block=True, save=True):
    print(f"plotting...{title}")
    global fig
    # Close the previous figure if it exists
    if fig:
        plt.close(fig)

    h = parameters["H"]
    q_dot_required = parameters["q_dot_required"]
    p_required = parameters["p_required"]
    p_solar = parameters["w_solar_per_w_installed"] * parameters["SOLAR_SIZE"]
    load_hx_eff = parameters["LOAD_HX_EFF"]
    cp_water = parameters["CP_WATER"]
    t_amb = parameters["t_amb"]

    p_compressor = u[0]
    m_dot_cond = u[1]
    m_dot_load = u[2]
    t_tank = y[0]
    t_cond = y[1]
    p_grid = - p_solar + p_compressor + p_required

    # Create time array
    t = np.linspace(0, n_steps * h, n_steps)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    t_load = np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        t_load[i] = get_t_load(y[:, i], parameters, i)

    q_dot_load = load_hx_eff * m_dot_load * cp_water * (t_tank - t_amb)

    # First subplot for inputs
    axes[0].plot(t, parameters["daily_prices"], label="C_grid", **plot_styles[0])
    axes[0].set_ylabel("money")
    axes[0].legend()
    axes[0].legend(loc="upper left")
    axes[0].grid(True)
    if title:
        axes[0].set_title(title)

    ax0 = axes[0].twinx()
    ax0.plot(t, q_dot_required, label="q_dot_required", **plot_styles[1])
    # ax0.plot(t, p_required, label="p_required", **plot_styles[2])
    # ax0.plot(t, p_solar, label="p_solar", **plot_styles[3])
    ax0.set_ylabel("W")
    ax0.legend(loc="upper right")

    # Second subplot for temperatures and e_bat
    axes[1].plot(t, t_tank, label="T_tank", **plot_styles[0])
    axes[1].plot(t, t_cond, label="T_cond", **plot_styles[1])
    axes[1].plot(t, t_load, label="T_load", **plot_styles[2])
    axes[1].plot(t, t_amb, label="T_amb", **plot_styles[3])
    axes[1].set_ylabel("Temperature (K)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

    ax1 = axes[1].twinx()
    ax1.plot(t, q_dot_load, label="Q_dot_load", **plot_styles[4])
    ax1.set_ylabel("Ws")
    ax1.legend(loc="upper right")


    axes[2].plot(t, p_compressor, label="P_comp", **plot_styles[0])
    axes[2].plot(t, p_grid, label="P_grid", **plot_styles[1])
    axes[2].plot(t, p_required, label="p_required", **plot_styles[2])
    axes[2].plot(t, -p_solar, label="p_solar", **plot_styles[3])
    axes[2].set_ylabel("Power[W]")
    axes[2].legend(loc="upper left")
    axes[2].grid(True)
    axes[2].set_xlabel("Time (s)")

    # ax2 = axes[2].twinx()
    # ax2.plot(t, m_dot_cond, label="m_dot_cond", **plot_styles[2])
    # ax2.plot(t, m_dot_load, label="m_dot_load", **plot_styles[3])
    # ax2.set_ylabel("Mass flow rates")
    # ax2.legend(loc="upper right")

    # Show the plots
    if show:
        # Save and close plot
        # plt.ion()  # Turn on the interactive mode
        plt.show(block=block)  # Draw the figure
        plt.pause(0.3)  # Time for the figure to load

    if save:
        plt.savefig(f"tmp/frame_{time.time()}.png")


def fd_gradients(y0, u, dae_p, n_steps, parameters):
    """
    Finite difference of function cost_function
    with respect
        initial conditions y0,
        parameters p,
        and control signals u at timestep 100

    f'(x) = (f(y+h) - f(y)) / h
    """
    delta = 1e-5
    h = parameters["H"]

    # Initial solution
    y = solve(y0, u, dae_p, h, n_steps)
    # print("fd solve: ", y)

    dfdy0 = []
    for i in range(len(y0)):
        y0_perturbed = y0.copy()  # Create a new copy of y0
        y0_perturbed[i] += delta
        y_perturbed = solve(y0_perturbed, u, dae_p, h, n_steps)
        dfdy0.append((cost_function(y_perturbed, u, parameters) - cost_function(y, u, parameters)) / delta)

    dfdp = []
    for i in range(len(dae_p)):
        p_perturbed = dae_p.copy()  # Create a new copy of p
        p_perturbed[i] += delta
        y_perturbed = solve(y0, u, p_perturbed, h, n_steps)
        dfdp.append((cost_function(y_perturbed, u, parameters) - cost_function(y, u, parameters)) / delta)

    dfdu_3 = []
    for i in range(len(u[:, 0])):
        u_perturbed = u.copy()  # Create a new copy of u
        u_perturbed[i, 3] += delta
        y_perturbed = solve(y0, u_perturbed, dae_p, h, n_steps)
        dfdu_3.append((cost_function(y_perturbed, u_perturbed, parameters) - cost_function(y, u, parameters)) / delta)

    return dfdy0, dfdp, dfdu_3


def plot_history(hist, only_last=True):
    # Get inputs
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    n_steps = int(horizon / h)
    t0 = 0
    y0 = np.array([298.34089176, 309.70395426])  # T_tank, T_cond

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["daily_prices"] = dynamic_parameters["daily_prices"]
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["y0"] = y0


    storeHistory = History(hist)
    histories = storeHistory.getValues()

    if only_last:
        indices = [-1]  # Only take the last index
    else:
        # Loop through every x opt results
        x = 1
        indices = list(range(0, len(histories["p_compressor"]), x))

    # loop through histories
    for iter, i in enumerate(indices):
        u = np.zeros((4, n_steps))
        u[0] = histories["p_compressor"][i]
        u[1] = histories["m_dot_cond"][i]
        u[2] = histories["m_dot_load"][i]
        u[3] = parameters["t_amb"]

        # p[0] = cp_water
        # p[1] = m_tank
        # p[2] = U
        # p[3] = A
        # p[4] = load_hx_eff
        dae_p = np.array(
            [
                parameters["CP_WATER"],
                parameters["TANK_VOLUME"] * parameters["RHO_WATER"],  # tank mass
                parameters["U"],
                6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3),  # tank surface area (m2)
                parameters["LOAD_HX_EFF"],
            ]
        )

        y = dae_forward(y0, u, dae_p, n_steps)
        # print("y[1]: ", y[:, 1])
        # print("u[1]: ", u[:, 1])
        # print("solution:", y[:, -1])
        if only_last:
            plot(y, u, n_steps, parameters, save=False, show=True)
            return
        else:
            title = f"iter: {iter}/{len(indices)}"
            plot(y, u, n_steps, parameters, title=title, show=False)

    plot_film("saves/simulate_simple_hp_wo_battery.gif")  # create animation with pictures from tmp folder



def main(hist=None):
    # Get inputs
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    n_steps = int(horizon / h)
    t0 = 0
    y0 = np.array([300, 309])  # T_tank, T_cond

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["daily_prices"] = dynamic_parameters["daily_prices"]
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["y0"] = y0

    # Prepare DAE inputs
    u = np.zeros((4, n_steps))

    if hist:
        storeHistory = History(hist)
        histories = storeHistory.getValues()
        u[0] = histories["p_compressor"][-1]
        u[1] = histories["m_dot_cond"][-1]
        u[2] = histories["m_dot_load"][-1]
        u[3] = parameters["t_amb"]

    else:
        P_comp = np.ones((n_steps)) * parameters["P_COMPRESSOR_MAX"]
        P_comp[-int(n_steps / 3) :] = 1e-6
        P_comp[-int(n_steps / 4) :] = parameters["P_COMPRESSOR_MAX"]
        # P_comp[-int(n_steps / 2) :] = parameters["P_COMPRESSOR_MAX"]

        m_dot_cond = np.ones((n_steps)) * 0.3  # kg/s
        # m_dot_cond = np.ones((n_steps)) * 1  # kg/s
        m_dot_cond[-int(n_steps / 4) :] = 1e-3
        # m_dot_cond[-int(n_steps / 6) :] = 0.3

        m_dot_load = np.ones((n_steps)) * 1e-6  # kg/s
        # m_dot_load = np.ones((n_steps)) * 0.3  # kg/s
        m_dot_load[-int(n_steps / 3) :] = 0.2

        u[0, :] = P_comp  # P_comp
        u[1, :] = m_dot_cond
        u[2, :] = m_dot_load
        u[3, :] = parameters["t_amb"]

    # p[0] = cp_water
    # p[1] = m_tank
    # p[2] = U
    # p[3] = A
    # p[4] = load_hx_eff
    dae_p = np.array(
        [
            parameters["CP_WATER"],
            parameters["TANK_VOLUME"] * parameters["RHO_WATER"],  # tank mass
            parameters["U"],
            6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3),  # tank surface area (m2)
            parameters["LOAD_HX_EFF"],
        ]
    )

    y = dae_forward(y0, u, dae_p, n_steps)
    print("solution:", y[:, -1])
    print(y.shape)
    print("y[1]: ", y[:, 1])
    print("u[1]: ", u[:, 1])
    plot(y, u, n_steps, parameters, save=False)
    # plot_animation(y, u, n_steps, parameters)

    # FD derivatives
    dCdy_0_fd, dCdp_fd, dCdu_fd = fd_gradients(y0, u, dae_p, n_steps, parameters)
    print("(finite diff) dC/dy0", dCdy_0_fd)
    print("(finite diff) dC/dp: ", dCdp_fd)
    print("(finite diff) dC/du_3: ", dCdu_fd)

    # # Adjoint derivatives
    dCdy_0_adj, dCdp_adj, dCdu_adj = dae_adjoints(y, u, dae_p, n_steps, parameters)
    print("(adjoint) dC/dy_0: ", dCdy_0_adj)
    print("(adjoint) dC/dp: ", dCdp_adj)
    print("(adjoint) dC/du_3: ", dCdu_adj[:, 3])

    # Discrepancies
    print(f"Discrepancy dC/dy_0: {np.abs(dCdy_0_adj - dCdy_0_fd) / (np.abs(dCdy_0_fd) + 1e-9) * 100}%")
    print(f"Discrepancy dC/dp: {np.abs(dCdp_adj - dCdp_fd) / (np.abs(dCdp_fd) + 1e-9) * 100}%")
    print(f"Discrepancy dC/du: {np.abs(dCdu_adj[:, 3] - dCdu_fd) / (np.abs(dCdu_fd) + 1e-9) * 100}%")


if __name__ == "__main__":
    # main()
    plot_history(hist="saves/sand_wo_battery_wo_finn.hst", only_last=True)
    # plot_history(hist="saves/sand_wo_battery_wo_finn.hst", only_last=False)
    # plot_history(hist="saves/mdf_wo_finn.hst", only_last=True)
