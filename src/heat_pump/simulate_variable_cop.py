import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from parameters import PARAMS

"""
Here we only simulate the water tank side, but
now assume that the COP is variable:
From 3 at 273K or lower, to 0 at the maximum condenser temperature of 70º.

Considerations:
    effectiveness of hx's is constant, no matter the delta in temperature
"""


def cop(T):
    """
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


def cost_function(y):
    return sum(y[0])


def get_drdy(y, p, u, h):
    # ∂r(y_n,p_n)/∂y_n
    # r = y[0]
    drdy = np.zeros(4)
    drdy[0] = 1
    drdy[1] = 0
    drdy[2] = 0
    drdy[3] = 0
    return drdy


def get_drdp(y, p, u, h):
    # ∂r(y_n,p_n)/∂p_n
    # r = y[0]
    drdp = np.zeros(6)
    return drdp


def dae_system(y, y_prev, p, u_prev, h):
    r"""
    Solve the following system of non-linear equations

    ∘ m_tank * cp_water * (dT_tank/dt)
        = m_dot_cond * cp_water * T_cond
        + m_dot_load * cp_water * T_load
        - m_dot_tank * cp_water * T_tank
        - Q_dot_loss
    ∘ COP = cop(T_cond)
    ∘ Q_dot_cond = COP * P_comp # W
    ∘ Q_dot_cond = m_dot_cond * cp_water * (T_cond - T_tank)
    ∘ Q_dot_loss = U * A * (T_tank - T_amb)
    ∘ Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)
    ∘ Q_dot_load = m_dot_load * cp_water * (T_tank - T_load)


    with unknowns:
    T_tank, T_load, T_cond, Q_dot_load


    Discretized with Backward Euler
    ∘ m_tank * cp_water * ((T_tank - T_tank_prev)/h)
        - m_dot_cond * cp_water * T_cond
        - m_dot_load * cp_water * T_load
        + m_dot_tank * cp_water * T_tank
        + U * A * (T_tank - T_amb)
        = 0
    ∘ cop(T_cond) * P_comp - m_dot_cond * cp_water * (T_cond - T_tank) = 0
    ∘ Q_dot_load - load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb) = 0
    ∘ Q_dot_load - m_dot_load * cp_water * (T_tank - T_load) = 0


    Making:
    y[0] = T_tank
    y[1] = T_cond
    y[2] = T_load
    y[3] = Q_dot_load

    p[0] = cp_water
    p[1] = m_tank
    p[2] = U
    p[3] = A
    p[4] = T_amb
    p[5] = load_hx_eff

    u_prev[0] = P_comp
    u_prev[1] = m_dot_cond
    u_prev[2] = m_dot_load
    m_dot_tank = u_prev[1] + u_prev[2]


    ∘ p[1] * p[0] * ((y[0] - y_prev[0])/h)
        - u_prev[1] * p[0] * y[1]
        - u_prev[2] * p[0] * y[2]
        + (u_prev[1] + u_prev[2]) * p[0] * y[0]
        + p[2] * p[3] * (y[0] - p[4])
        = 0
    ∘ cop(y[1]) * u_prev[0] - u_prev[1] * p[0] * (y[1] - y[0]) = 0
    ∘ y[3] - p[5] * u_prev[2] * p[0] * (y[0] - p[4]) = 0
    ∘ y[3] - u_prev[2] * p[0] * (y[0] - y[2]) = 0

    Which are divided in differential equations (f(y,y_prev,p,u_prev) = 0)
    and algebraic equations (g(y,p,u_prev) = 0)
    """
    return [
        # f
        p[1] * p[0] * ((y[0] - y_prev[0]) / h)
        - u_prev[1] * p[0] * y[1]
        - u_prev[2] * p[0] * y[2]
        + (u_prev[1] + u_prev[2]) * p[0] * y[0]
        + p[2] * p[3] * (y[0] - p[4]),
        # g
        cop(y[1]) * u_prev[0] - u_prev[1] * p[0] * (y[1] - y[0]),
        y[3] - p[5] * u_prev[2] * p[0] * (y[0] - p[4]),
        y[3] - u_prev[2] * p[0] * (y[0] - y[2]),
    ]


def get_dfdp(y, y_prev, p, u_prev, h):
    # Compute ∂f(y_n,p_n,u_n)/∂p
    dfdp = np.zeros((1, 6))
    dfdp[0, 0] = p[1] * ((y[0] - y_prev[0]) / h) - u_prev[1] * y[1] - u_prev[2] * y[2] + (u_prev[1] + u_prev[2]) * y[0]
    dfdp[0, 1] = p[0] * ((y[0] - y_prev[0]) / h)
    dfdp[0, 2] = p[3] * (y[0] - p[4])
    dfdp[0, 3] = p[2] * (y[0] - p[4])
    dfdp[0, 4] = -p[2] * p[3]
    dfdp[0, 5] = 0
    return dfdp


def get_dfdu(y, y_prev, p, u_prev, h):
    # Compute ∂f(y_n,p_n,u_n)/∂p
    dfdu = np.zeros((1, 3))
    dfdu[0, 0] = 0
    dfdu[0, 1] = -p[0] * y[1] + p[0] * y[0]
    dfdu[0, 2] = -p[0] * y[2] + p[0] * y[0]
    return dfdu


def get_dgdp(y, p, u_prev, h):
    # Compute ∂g(y_n,p_n,u_n)/∂p_n
    dgdp = np.zeros((3, 6))
    dgdp[0, 0] = -u_prev[1] * (y[1] - y[0])
    dgdp[0, 1] = 0
    dgdp[0, 2] = 0
    dgdp[0, 3] = 0
    dgdp[0, 4] = 0
    dgdp[0, 5] = 0

    dgdp[1, 0] = -p[5] * u_prev[2] * (y[0] - p[4])
    dgdp[1, 1] = 0
    dgdp[1, 2] = 0
    dgdp[1, 3] = 0
    dgdp[1, 4] = p[5] * u_prev[2] * p[0]
    dgdp[1, 5] = -u_prev[2] * p[0] * (y[0] - p[4])

    dgdp[2, 0] = -u_prev[2] * (y[0] - y[2])
    dgdp[2, 1] = 0
    dgdp[2, 2] = 0
    dgdp[2, 3] = 0
    dgdp[2, 4] = 0
    dgdp[2, 5] = 0
    return dgdp


def get_dgdu(y, p, u_prev, h):
    # Compute ∂g(y_n,p_n,u_n)/∂p_n
    # cop(y[1]) * u_prev[0] - u_prev[1] * p[0] * (y[1] - y[0]),
    dgdu = np.zeros((3, 3))
    dgdu[0, 0] = cop(y[1])
    dgdu[0, 1] = -p[0] * (y[1] - y[0])
    dgdu[0, 2] = 0

    # y[3] - p[5] * u_prev[2] * p[0] * (y[0] - p[4]),
    dgdu[1, 0] = 0
    dgdu[1, 1] = 0
    dgdu[1, 2] = -p[5] * p[0] * (y[0] - p[4])

    # y[3] - u_prev[2] * p[0] * (y[0] - y[2]),
    dgdu[2, 0] = 0
    dgdu[2, 1] = 0
    dgdu[2, 2] = -p[0] * (y[0] - y[2])
    return dgdu


def get_dgdy(y, p, u_prev, h):
    # Compute ∂g(y_n,p_n,u_n)/∂y_n
    # ∘ cop(y[1]) * u_prev[0] - u_prev[1] * p[0] * (y[1] - y[0]) = 0
    dgdy = np.zeros((3, 4))
    dgdy[0, 0] = u_prev[1] * p[0]
    dgdy[0, 1] = get_dcopdT(y[1]) * u_prev[0] - u_prev[1] * p[0]
    dgdy[0, 2] = 0
    dgdy[0, 3] = 0

    # ∘ y[3] - p[5] * u_prev[2] * p[0] * (y[0] - p[4]) = 0
    dgdy[1, 0] = -p[5] * u_prev[2] * p[0]
    dgdy[1, 1] = 0
    dgdy[1, 2] = 0
    dgdy[1, 3] = 1

    # ∘ y[3] - u_prev[2] * p[0] * (y[0] - y[2]) = 0
    dgdy[2, 0] = -u_prev[2] * p[0]
    dgdy[2, 1] = 0
    dgdy[2, 2] = u_prev[2] * p[0]
    dgdy[2, 3] = 1
    return dgdy


def get_dfdy(y, y_prev, p, u_prev, h):
    # Compute ∂f(y_n,p_n,u_n)/∂y_n
    dfdy = np.zeros((1, 4))
    dfdy[0, 0] = p[1] * p[0] / h + (u_prev[1] + u_prev[2]) * p[0] + p[2] * p[3]
    dfdy[0, 1] = -u_prev[1] * p[0]
    dfdy[0, 2] = -u_prev[2] * p[0]
    dfdy[0, 3] = 0
    return dfdy


def get_dfdy_prev(y, y_prev, p, u_prev, h):
    # Compute ∂f(y_n,p_n,u_n)/∂y_n
    dfdy_prev = np.zeros((1, 4))
    dfdy_prev[0, 0] = -p[1] * p[0] / h
    dfdy_prev[0, 1] = 0
    dfdy_prev[0, 2] = 0
    dfdy_prev[0, 3] = 0
    return dfdy_prev


def solve(y_0, p, u, h, n_steps):
    # Initialize parameters
    y = np.zeros((4, n_steps + 1))

    # Initial conditions
    y[:, 0] = y_0

    # Time-stepping loop
    for n in range(n_steps):
        # Use fsolve to solve the nonlinear system
        y_next = fsolve(dae_system, y[:, n], args=(y[:, n], p, u[:, n], h))
        y[:, n + 1] = y_next

    return y


def adjoint_gradients(y, p, u, h, n_steps):
    r"""
    Adjoint gradients of the cost function with respect to parameters

    min_p C = Σr(y_n, p)
    s.t.
        f(y_n, y_(n-1), p, u_(n-1)) = 0
        g(y_n, p, u_(n-1)) = 0

    where f are the differential constraints
    and g are the algebraic constraints

    by augmenting the system
    p' = 0

    min_{p_n} C = Σr(y_n, p_n)
        f(y_n, y_(n-1), p_n) = 0
        g(y_n, p_n) = 0
        (p_n - p_(n-1)) / h = 0

    Lagrangian:
    L = Σr(y_n, p_n)
        - Σ λ_n f(y_n, y_(n-1), p_n, u_(n-1))
        - Σ ν_n g(y_n, p_n, u_(n-1))
        - Σ μ_n (p_n - p_(n-1))

    ∂L/∂y_n = 0 = ∂r(y_n, p_n)/∂y_n - λ_n ∂f(y_n, y_(n-1), p_n)/∂y_n - λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1))/∂y_n - ν_n ∂g/∂y_n
    ∂L/∂p_n = 0 = ∂r(y_n, p_n)/∂p_n - λ_n ∂f(y_n, y_(n-1), p_n)/∂p_n - ν_n ∂g/py_n - μ_n + μ_(n+1)

    Solve for λ_n and μ_n at each step:
    [∂f/∂y_n^T  ∂g/∂y_n^T  0] [λ_n]   [(∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T]
    [∂f/∂p_n^T  ∂g/∂p_n^T  I] [ν_n] = [(∂r/∂p_n + μ_{n+1})^T                               ]
                              [μ_n]
    Terminal conditions:
    [∂f/∂y_N^T  ∂g/∂y_N^T  0] [λ_N]   [(∂r/∂y_N)^T]
    [∂f/∂p_N^T  ∂g/∂p_N^T  I] [ν_N] = [(∂r/∂p_N)^T]
                              [μ_N]
    Solve for initial timestep:
    ∂L/∂y_0 = ∂C/∂y_0 = (∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T
    ∂L/∂p_0 = ∂C/∂p_0 = (∂r/∂p_n + μ_{n+1})^T
    """
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
    for n in range(n_steps, -1, -1):
        y_current = y[:, n]
        y_prev = y[:, n - 1]
        u_prev = u[:, n - 1]
        dfdy_n = get_dfdy(y_current, y_prev, p, u_prev, h)
        dfdp_n = get_dfdp(y_current, y_prev, p, u_prev, h)
        dgdy_n = get_dgdy(y_current, p, u_prev, h)
        dgdp_n = get_dgdp(y_current, p, u_prev, h)
        drdy_n = get_drdy(y_current, p, u_prev, h)
        drdp_n = get_drdp(y_current, p, u_prev, h)
        dfdu_n = get_dfdu(y[:, n], y[:, n - 1], p, u[:, n - 1], h)
        dgdu_n = get_dgdu(y[:, n], p, u[:, n - 1], h)

        if n == n_steps:
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
            # Inital timestep
            # ∂L/∂y_0 = ∂C/∂y_0 = (∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T
            # ∂L/∂p_0 = ∂C/∂p_0 = (∂r/∂p_n + μ_{n+1})^T
            dfdy_prev = get_dfdy_prev(y[:, n + 1], y[:, n], p, u[:, n], h)
            dCdy_0 = drdy_n - np.dot(adj_lambda[:, 1].T, dfdy_prev)
            dCdp_0 = drdp_n + adj_mu[:, 1]
        else:
            # [∂f/∂y_n^T  ∂g/∂y_n^T  0] [λ_n]   [(∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T]
            # [∂f/∂p_n^T  ∂g/∂p_n^T  I] [ν_n] = [(∂r/∂p_n + μ_{n+1})^T                               ]
            #                           [μ_n]
            dfdy_prev = get_dfdy_prev(y[:, n + 1], y[:, n], p, u[:, n], h)
            A = np.block([[dfdy_n.T, dgdy_n.T, np.zeros((n_states, n_params))], [dfdp_n.T, dgdp_n.T, np.eye(n_params)]])
            b = np.concatenate([drdy_n - np.dot(adj_lambda[:, n + 1].T, dfdy_prev), drdp_n + adj_mu[:, n + 1]])
            adjs = np.linalg.solve(A, b)
            adj_lambda[:, n] = adjs[:n_odes]
            adj_nu[:, n] = adjs[n_odes : (n_odes + n_algs)]
            adj_mu[:, n] = adjs[(n_odes + n_algs) :]

        # ∂L/∂u_n = ∂C/∂u_n = -λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1), u_n)/∂u_n - ν_(n+1) ∂g(y_(n+1), p_(n+1), u_n)/∂u_n
        dCdu[:, n - 1] = - adj_lambda[:, n].T @ dfdu_n - adj_nu[:, n].T @ dgdu_n

    return dCdy_0, dCdp_0, dCdu


def plot(y, u, n_steps, h):
    # Create time array
    t = np.linspace(0, n_steps * h, n_steps + 1)

    # Create the figure and the subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # First subplot for T_tank, T_load, and T_cond
    axes[0].plot(t, y[0], label="T_tank")
    axes[0].plot(t, y[1], label="T_cond")
    axes[0].plot(t, y[2], label="T_load")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Temperature Profiles")
    # axes[0].legend()
    axes[0].legend(loc="upper left")
    axes[0].grid(True)

    # Second subplot for Q_dot_load
    # axes[1].plot(t, y[3], label="Q_dot_load", color="tab:orange")
    # axes[1].set_xlabel("Time (s)")
    # axes[1].set_ylabel("Q_dot_load (W)")
    # axes[1].set_title("Heat Load")
    # axes[1].legend()
    # axes[1].grid(True)

    ax0 = axes[0].twinx()
    ax0.plot(t, y[3], "--", label="Q_dot_load", color="red")
    ax0.set_ylabel("Q_dot_load (W)")
    ax0.legend(loc="upper right")

    # Set common x-axis label
    # axes[1].set_xlabel("Time (s)")

    # Third subplot for control variables
    axes[1].plot(t[:-1], u[0], label="P_comp")
    axes[1].set_ylabel("Compressor power")
    axes[1].set_title("Control Variables")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

    # Create a secondary y-axis for u[1] and u[2]
    ax1 = axes[1].twinx()
    ax1.plot(t[:-1], u[1], label="m_dot_cond", color="tab:red")
    ax1.plot(t[:-1], u[2], label="m_dot_load", color="tab:green")
    ax1.set_ylabel("Mass flow rates")
    ax1.legend(loc="upper right")

    # Set common x-axis label
    axes[1].set_xlabel("Time (s)")

    # Show the plots
    plt.tight_layout()
    plt.show()


def fd_gradients(y0, p, u, h, n_steps):
    """
    Finite difference of function cost_function
    with respect
        initial conditions y0,
        parameters p,
        and control signals u at timestep 100

    f'(x) = (f(y+h) - f(y)) / h
    """
    delta = 1e-6

    # Initial solution
    y = solve(y0, p, u, h, n_steps)

    dfdy0 = []
    for i in range(len(y0)):
        y0_perturbed = y0.copy()  # Create a new copy of y0
        y0_perturbed[i] += delta
        y_perturbed = solve(y0_perturbed, p, u, h, n_steps)
        dfdy0.append((cost_function(y_perturbed) - cost_function(y)) / delta)

    dfdp = []
    for i in range(len(p)):
        p_perturbed = p.copy()  # Create a new copy of p
        p_perturbed[i] += delta
        y_perturbed = solve(y0, p_perturbed, u, h, n_steps)
        dfdp.append((cost_function(y_perturbed) - cost_function(y)) / delta)

    dfdu_3 = []
    for i in range(len(u[:, 0])):
        u_perturbed = u.copy()  # Create a new copy of u
        u_perturbed[i, 3] += delta
        y_perturbed = solve(y0, p, u_perturbed, h, n_steps)
        dfdu_3.append((cost_function(y_perturbed) - cost_function(y)) / delta)

    return dfdy0, dfdp, dfdu_3


def main():
    horizon = PARAMS["HORIZON"]
    h = PARAMS["H"]
    n_steps = int(horizon / h)
    y0, p, u = get_inputs(n_steps, h)
    y = solve(y0, p, u, h, n_steps)
    print("solution:", y[:, -1])
    print("y[2]", y[:, 2])
    plot(y, u, n_steps, h)

    # FD derivatives
    dCdy_0_fd, dCdp_fd, dCdu_fd = fd_gradients(y0, p, u, h, n_steps)
    print("(finite diff) df/dy0", dCdy_0_fd)
    print("(finite diff) df/dp: ", dCdp_fd)
    print("(finite diff) df/du_3: ", dCdu_fd)

    # Adjoint derivatives
    dCdy_0_adj, dCdp_adj, dCdu_adj = adjoint_gradients(y, p, u, h, n_steps)
    print("(adjoint) df/dy_0: ", dCdy_0_adj)
    print("(adjoint) df/dp: ", dCdp_adj)
    print("(finite diff) df/du_3: ", dCdu_adj[:, 3])

    # Discrepancies
    print(f"Discrepancy df/dy_0: {np.abs(dCdy_0_adj - dCdy_0_fd) / (np.abs(dCdy_0_fd) + 1e-6) * 100}%")
    print(f"Discrepancy df/dp: {np.abs(dCdp_adj - dCdp_fd) / (np.abs(dCdp_fd) + 1e-6) * 100}%")
    print(f"Discrepancy df/du: {np.abs(dCdu_adj[:, 3] - dCdu_fd) / (np.abs(dCdu_fd) + 1e-6) * 100}%")


def get_inputs(n_steps, h):
    # Parameters
    U = 0.04  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
    # U = 1  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
    # V = 1  # Tank volume (m3)
    V = 0.01  # Tank volume (m3)
    A = 6 * np.pi * (V / (2 * np.pi)) ** (2 / 3)  # Tank surface area (m2)
    rho_water = 1000  # Water density (kg/m3)
    cp_water = 4186  # Specific heat capacity of water (J/(kg·K))
    m_tank = V * rho_water  # Mass of water in the tank (kg)
    print("m_tank: ", m_tank)
    T_amb = 298  # [K] (25ºC)
    load_hx_eff = 0.8

    p = [cp_water, m_tank, U, A, T_amb, load_hx_eff]

    y0 = [298, 309, 298, 0]

    # Fix control variables
    P_comp_max = 10000  # W
    total_time = h * n_steps
    f = 1 / total_time
    desired_frequency = 10 * f
    time_steps = np.arange(n_steps) * h
    # P_comp = P_comp_max * np.sin(2 * np.pi * desired_frequency * time_steps) + P_comp_max
    P_comp = np.ones((n_steps)) * P_comp_max
    # P_comp = P_comp_max * np.sin(2 * np.pi * f * 100 * np.arange(n_steps)) + P_comp_max
    P_comp[-int(n_steps / 3) :] = 1e-6
    P_comp[-int(n_steps / 4) :] = P_comp_max
    m_dot_cond = np.ones((n_steps)) * 0.3  # kg/s
    m_dot_cond[-int(n_steps / 3) :] = 1e-6
    m_dot_cond[-int(n_steps / 6) :] = 0.3
    m_dot_load = np.ones((n_steps)) * 1e-6  # kg/s
    m_dot_load[-int(n_steps / 3) :] = 0.2
    u = np.zeros((3, n_steps))
    u[0, :] = P_comp  # P_comp
    u[1, :] = m_dot_cond  # m_dot_cond
    u[2, :] = m_dot_load  # m_dot_tank

    return y0, p, u


if __name__ == "__main__":
    main()
