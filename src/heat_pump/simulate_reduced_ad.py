import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
import jax
import jax.numpy as jnp

"""
Here we only simulate the water tank side, but
now assume that the COP is variable:
From 3 at 273K or lower, to 0 at the maximum condenser temperature of 70º.

Considerations:
    effectiveness of hx's is constant, no matter the delta in temperature
"""

rho_water = 1000  # Water density (kg/m3)
cp_water = 4186  # Specific heat capacity of water (J/(kg·K))
T_amb = 298  # [K] (25ºC)
load_hx_eff = 0.8
P_comp_max = 10000  # W


def cop(T):
    """
    piecewise lineal -> non-linear function overall

    COP is 0 for T > 343
    COP is 3 for T < 273
    COP varies linearly from 3 at 273K to 0 at 343K
    """
    conditions = [
        T < 273,
        (T >= 273) & (T < 343),
        T >= 343
    ]
    choices = [
        lambda T: 3,
        lambda T: 14.7 - (3 / 70) * T,
        lambda T: 0
    ]
    return jnp.select(conditions, [f(T) for f in choices])


def cost_function(y):
    return sum(y[0])


def get_drdy(y, p, u, h):
    # ∂r(y_n,p_n)/∂y_n
    # r = y[0]
    drdy = np.zeros(2)
    drdy[0] = 1
    drdy[1] = 0
    return drdy


def get_drdp(y, p, u, h):
    # ∂r(y_n,p_n)/∂p_n
    # r = y[0]
    drdp = np.zeros(6)
    return drdp


def get_p_heat(Q_dot_required, T_tank, m_dot_load):
    return Q_dot_required - load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)

def get_t_load(T_tank, m_dot_load):
    return T_tank - load_hx_eff * (T_tank - T_amb)


def f(y, y_prev, p, u_prev, h):
    # m_tank * cp_water * ((T_tank - T_tank_prev)/h)
    #     - m_dot_cond * cp_water * T_cond
    #     - m_dot_load * cp_water * (T_tank - load_hx_eff * (T_tank - T_amb))
    #     + (m_dot_cond + m_dot_load) * cp_water * T_tank
    #     + U * A * (T_tank - T_amb)
    #     = 0
    return jnp.array([
        p[1] * p[0] * ((y[0] - y_prev[0]) / h)
        - u_prev[1] * p[0] * y[1]
        - u_prev[2] * p[0] * (y[0] - p[5] * (y[0] - p[4]))
        + (u_prev[1] + u_prev[2]) * p[0] * y[0]
        + p[2] * p[3] * (y[0] - p[4])
    ])


def g(y, p, u_prev, h):
    # cop(T_cond) * P_comp - m_dot_cond * cp_water * (T_cond - T_tank) = 0
    return jnp.array([
        cop(y[1]) * u_prev[0] - u_prev[1] * p[0] * (y[1] - y[0])
    ])


def dae_system(y, y_prev, p, u_prev, h):
    f_result = f(y, y_prev, p, u_prev, h)
    g_result = g(y, p, u_prev, h)
    return jnp.concatenate((f_result, g_result))


# JACOBIANS
# jax.jacobian will automatically
# use jax.jacrev or jax.jacfwd
# based on the number of inputs and outputs
get_dfdy = jax.jacobian(f, argnums=0)
get_dfdy_prev = jax.jacobian(f, argnums=1)
get_dfdp = jax.jacobian(f, argnums=2)
get_dfdu = jax.jacobian(f, argnums=3)
get_dgdy = jax.jacobian(g, argnums=0)
get_dgdp = jax.jacobian(g, argnums=1)
get_dgdu = jax.jacobian(g, argnums=2)


def solve(y_0, p, u, h, n_steps):
    # Initialize parameters
    y = np.zeros((len(y_0), n_steps + 1))

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
    y = jnp.asarray(y)
    u = jnp.asarray(u)
    p = jnp.asarray(p)
    h = jnp.float32(h)

    dfdy = np.array(get_dfdy(y[:, 1], y[:, 0], p, u[:, 0], h))
    dgdp = np.array(get_dgdp(y[:, 1], p, u[:, 0], h))
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
        dfdy_n = np.array(get_dfdy(y_current, y_prev, p, u_prev, h))
        dfdp_n = np.array(get_dfdp(y_current, y_prev, p, u_prev, h))
        dfdu_n = np.array(get_dfdu(y[:, n], y[:, n - 1], p, u[:, n - 1], h))
        drdy_n = np.array(get_drdy(y_current, p, u_prev, h))
        drdp_n = np.array(get_drdp(y_current, p, u_prev, h))
        dgdu_n = np.array(get_dgdu(y[:, n], p, u[:, n - 1], h))
        dgdp_n = np.array(get_dgdp(y_current, p, u_prev, h))
        dgdy_n = np.array(get_dgdy(y_current, p, u_prev, h))

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
        dCdu[:, n - 1] = -adj_lambda[:, n].T @ dfdu_n - adj_nu[:, n].T @ dgdu_n

    return dCdy_0, dCdp_0, dCdu


def plot(y, u, n_steps, h, Q_dot_required):
    # Create time array
    t = np.linspace(0, n_steps * h, n_steps + 1)

    # Create the figure and the subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    T_load = get_t_load(y[0, :-1], u[2])
    Q_dot_required = np.ones((n_steps)) * P_comp_max * 2
    P_heat = get_p_heat(Q_dot_required, y[0, :-1], u[2])
    Q_dot_load = Q_dot_required - P_heat

    # First subplot for T_tank, T_load, and T_cond
    axes[0].plot(t, y[0], label="T_tank")
    axes[0].plot(t, y[1], label="T_cond")
    axes[0].plot(t[:-1], T_load, label="T_load")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Temperature Profiles")
    # axes[0].legend()
    axes[0].legend(loc="upper left")
    axes[0].grid(True)

    ax0 = axes[0].twinx()
    ax0.plot(t[:-1], Q_dot_load, "--", label="Q_dot_load", color="red")
    ax0.plot(t[:-1], P_heat, "--", label="P_heat", color="purple")
    ax0.set_ylabel("W")
    ax0.legend(loc="upper right")

    axes[1].plot(t[:-1], u[0], label="P_comp")
    axes[1].plot(t[:-1], Q_dot_required, label="Q_dot_required")
    axes[1].set_ylabel("Power[W]")
    axes[1].set_title("Control Variables")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

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
    time = 10000  # s
    h = 10  # timestep
    # n_steps = int(1e5)
    n_steps = int(time / h)
    # n_steps = int(5)
    y0, p, u, Q_dot_required = get_inputs(n_steps, h)
    y = solve(y0, p, u, h, n_steps)
    print("solution:", y[:, -1])
    print("y[1]: ", y[:, 1])
    plot(y, u, n_steps, h, Q_dot_required)

    # FD derivatives
    dCdy_0_fd, dCdp_fd, dCdu_fd = fd_gradients(y0, p, u, h, n_steps)
    print("(finite diff) dC/dy0", dCdy_0_fd)
    print("(finite diff) dC/dp: ", dCdp_fd)
    print("(finite diff) dC/du_3: ", dCdu_fd)

    # Adjoint derivatives
    dCdy_0_adj, dCdp_adj, dCdu_adj = adjoint_gradients(y, p, u, h, n_steps)
    print("(adjoint) dC/dy_0: ", dCdy_0_adj)
    print("(adjoint) dC/dp: ", dCdp_adj)
    print("(adjoint) dC/du_3: ", dCdu_adj[:, 3])

    # Discrepancies
    print(f"Discrepancy dC/dy_0: {np.abs(dCdy_0_adj - dCdy_0_fd) / (np.abs(dCdy_0_fd) + 1e-6) * 100}%")
    print(f"Discrepancy dC/dp: {np.abs(dCdp_adj - dCdp_fd) / (np.abs(dCdp_fd) + 1e-6) * 100}%")
    print(f"Discrepancy dC/du: {np.abs(dCdu_adj[:, 3] - dCdu_fd) / (np.abs(dCdu_fd) + 1e-6) * 100}%")


def get_inputs(n_steps, h):
    # Parameters
    U = 0.04  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
    # U = 1  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
    # V = 1  # Tank volume (m3)
    V = 0.01  # Tank volume (m3)
    A = 6 * np.pi * (V / (2 * np.pi)) ** (2 / 3)  # Tank surface area (m2)
    m_tank = V * rho_water  # Mass of water in the tank (kg)
    print("m_tank: ", m_tank)

    total_time = h * n_steps
    f = 1 / total_time
    desired_frequency = 10 * f
    time_steps = np.arange(n_steps) * h

    p = [cp_water, m_tank, U, A, T_amb, load_hx_eff]
    y0 = [298.34089176, 309.70395426]

    # Fix control variables
    # P_comp = P_comp_max * np.sin(2 * np.pi * desired_frequency * time_steps) + P_comp_max
    P_comp = np.ones((n_steps)) * P_comp_max
    # P_comp = P_comp_max * np.sin(2 * np.pi * f * 100 * np.arange(n_steps)) + P_comp_max
    P_comp[-int(n_steps / 3) :] = 1e-6
    P_comp[-int(n_steps / 4) :] = P_comp_max

    Q_dot_required = np.ones((n_steps)) * P_comp_max * 2

    m_dot_cond = np.ones((n_steps)) * 0.3  # kg/s
    m_dot_cond[-int(n_steps / 3) :] = 1e-6
    m_dot_cond[-int(n_steps / 6) :] = 0.3

    m_dot_load = np.ones((n_steps)) * 1e-6  # kg/s
    m_dot_load[-int(n_steps / 3) :] = 0.2

    u = np.zeros((3, n_steps))
    u[0, :] = P_comp  # P_comp
    u[1, :] = m_dot_cond
    u[2, :] = m_dot_load

    return y0, p, u, Q_dot_required


if __name__ == "__main__":
    main()
