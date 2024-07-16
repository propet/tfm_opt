import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation


# Define the DAE system
def dae_system(y, y_prev, p, u_prev, h):
    r"""
    Solve
    u[0]*dy[0]/dt + p[0]*y[0] - p[1]*y[1] + p[1]*y[1]*y[0] + p[1]*y[1]^2 = 0
    dy[1]/dt - p[1]*y[0]^2 + y[1] = 0
    y[2] - 1 + y[0] + y[1] = 0

    Discretized with Backward Euler

    u[0]*y[0] - u[0]*y_prev[0] + h*p[0]*y[0] - h*p[1]*y[1] + h*p[1]*y[1]*y[0] + h*p[1]*y[1]^2 = 0
    y[1] - y_prev[1] - h*p[1]*y[0]^2 + h*y[1] = 0
    y[2] - 1 + y[0] + y[1] = 0

    Which are divided in differential equations (f(y, y_prev, p, u_prev) = 0)
    u[0]*y[0] - u[0]*y_prev[0] + h*u[0]*p[0]*y[0] - h*p[1]*y[1] + h*p[1]*y[1]*y[0] + h*p[1]*y[1]^2 = 0
    u_prev[1]*y[1] - y_prev[1] - h*p[1]*y[0]^2 + h*y[1] = 0

    and algebraic equations (g(y,p,u) = 0)
    y[2] - 1 + y[0] + y[1] = 0
    """
    return [
        u_prev[0]*y[0] - u_prev[0]*y_prev[0] + h * p[0] * y[0] - h * p[1] * y[1] + h * p[1] * y[1] * y[0] + h * p[1] * y[1] ** 2,
        y[1] - y_prev[1] - h * p[1] * y[0] ** 2 + h * y[1],
        u_prev[1] * y[2] - 1 + y[0] + y[1],
    ]


def get_dfdy(y, y_prev, p, u_prev, h):
    # Compute ∂f(y_n,p)/∂y_n
    dfdy = np.zeros((2, 3))
    dfdy[0, 0] = u_prev[0] + h * p[0] + h * p[1] * y[1]
    dfdy[0, 1] = -h * p[1] + h * p[1] * y[0] + 2 * h * p[1] * y[1]
    dfdy[0, 2] = 0

    dfdy[1, 0] = -2 * h * p[1] * y[0]
    dfdy[1, 1] = 1 + h
    dfdy[1, 2] = 0
    return dfdy


def get_dfdy_prev(y, y_prev, p, u_prev, h):
    # Compute ∂f(y_n,p)/∂y_n
    dfdy_prev = np.zeros((2, 3))
    dfdy_prev[0, 0] = -u_prev[0]
    dfdy_prev[0, 1] = 0
    dfdy_prev[0, 2] = 0

    dfdy_prev[1, 0] = 0
    dfdy_prev[1, 1] = -1
    dfdy_prev[1, 2] = 0
    return dfdy_prev


def get_dfdp(y, p, u_prev, h):
    # Compute ∂f(y_n,p)/∂p
    dfdp = np.zeros((2, 2))
    dfdp[0, 0] = h * y[0]
    dfdp[0, 1] = -h * y[1] + h * y[1] * y[0] + h * y[1] ** 2

    dfdp[1, 0] = 0
    dfdp[1, 1] = -h * y[0] ** 2
    return dfdp


def get_dfdu(y, y_prev, p, u_prev, h):
    # Compute ∂f(y_n,p_n,u_(n-1))/∂u_(n-1)
    # u_prev[0]*y[0] - u_prev[0]*y_prev[0] + h * p[0] * y[0] - h * p[1] * y[1] + h * p[1] * y[1] * y[0] + h * p[1] * y[1] ** 2,
    dfdu = np.zeros((2, 2))
    dfdu[0, 0] = y[0] - y_prev[0]
    dfdu[0, 1] = 0

    # y[1] - y_prev[1] - h * p[1] * y[0] ** 2 + h * y[1],
    dfdu[1, 0] = 0
    dfdu[1, 1] = 0
    return dfdu


def get_dgdy(y, p, u_prev, h):
    # Compute ∂g(y_n,p_n)/∂y_n
    dgdy = np.zeros((1, 3))
    dgdy[0, 0] = 1
    dgdy[0, 1] = 1
    dgdy[0, 2] = u_prev[1]
    return dgdy


def get_dgdp(y, p, h):
    # Compute ∂g(y_n,p_n)/∂p_n
    dgdp = np.zeros((1, 2))
    dgdp[0, 0] = 0
    dgdp[0, 1] = 0
    return dgdp


def get_dgdu(y, p, u_prev, h):
    # Compute ∂g(y_n,p_n)/∂p_n
    # u_prev[1] * y[2] - 1 + y[0] + y[1],
    dgdu = np.zeros((1, 2))
    dgdu[0, 0] = 0
    dgdu[0, 1] = y[2]
    return dgdu


def cost_function(y):
    return sum(y[0])


def get_drdy(y, p, h):
    # ∂r(y_n,p_n)/∂y_n
    # r = y[0]
    drdy = np.zeros(3)
    drdy[0] = 1
    drdy[1] = 0
    drdy[2] = 0
    return drdy


def get_drdp(y, p, h):
    # ∂r(y_n,p_n)/∂p_n
    # r = y[0]
    drdp = np.zeros(2)
    return drdp


def solve(y_0, p, u, h, steps):
    # Initialize parameters
    y = np.zeros((3, steps + 1))

    # Initial conditions
    y[:, 0] = y_0

    # Time-stepping loop
    for n in range(steps):
        # Use fsolve to solve the nonlinear system
        y_next = fsolve(dae_system, y[:, n], args=(y[:, n], p, u[:, n], h))
        y[:, n + 1] = y_next

    return y


def adjoint_gradients(y, p, u, h, steps):
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
        f(y_n, y_(n-1), p_n, u_(n-1)) = 0
        g(y_n, p_n, u_(n-1)) = 0
        (p_n - p_(n-1)) / h = 0

    Lagrangian:
    L = Σr(y_n, p_n)
        - Σ λ_n f(y_n, y_(n-1), p_n, u_(n-1))
        - Σ ν_n g(y_n, p_n, u_(n-1))
        - Σ μ_n (p_n - p_(n-1))

    ∂L/∂y_n = 0 = ∂r(y_n, p_n)/∂y_n - λ_n ∂f(y_n, y_(n-1), p_n, u_(n-1))/∂y_n - λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1), u_n)/∂y_n - ν_n ∂g/∂y_n
    ∂L/∂p_n = 0 = ∂r(y_n, p_n)/∂p_n - λ_n ∂f(y_n, y_(n-1), p_n, u_(n-1))/∂p_n - ν_n ∂g/py_n - μ_n + μ_(n+1)

    ∂L/∂u_n = ∂C/∂u_n = -λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1), u_n)/∂u_n - ν_(n+1) ∂g(y_(n+1), p_(n+1), u_n)/∂u_n

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
    dgdp = get_dgdp(y[:, 1], p, h)
    n_odes = dfdy.shape[0]
    n_states = dfdy.shape[1]
    n_algs = dgdp.shape[0]
    n_params = dgdp.shape[1]

    # Initialize adjoint variables
    adj_lambda = np.zeros((n_odes, steps + 1))
    adj_mu = np.zeros((n_params, steps + 1))
    adj_nu = np.zeros((n_algs, steps + 1))
    dCdy_0 = np.zeros(n_states)
    dCdp_0 = np.zeros(n_params)
    dCdu = np.zeros(u.shape)

    # Backward propagation of adjoint variables
    for n in range(steps, -1, -1):
        dfdy_n = get_dfdy(y[:, n], y[:, n - 1], p, u[:, n - 1], h)
        dfdp_n = get_dfdp(y[:, n], p, u[:, n - 1], h)
        dgdy_n = get_dgdy(y[:, n], p, u[:, n - 1], h)
        dgdp_n = get_dgdp(y[:, n], p, h)
        drdy_n = get_drdy(y[:, n], p, h)
        drdp_n = get_drdp(y[:, n], p, h)
        dfdu_n = get_dfdu(y[:, n], y[:, n - 1], p, u[:, n - 1], h)
        dgdu_n = get_dgdu(y[:, n], p, u[:, n - 1], h)

        if n == steps:
            # Terminal condition
            # [∂f/∂y_n^T  ∂g/∂y_n^T  0] [λ_n]   [(∂r/∂y_n)^T]
            # [∂f/∂p_n^T  ∂g/∂p_n^T  I] [ν_n] = [(∂r/∂p_n)^T]
            #                           [μ_n]
            A = np.block([
                [dfdy_n.T, dgdy_n.T, np.zeros((n_states, n_params))],
                [dfdp_n.T, dgdp_n.T, np.eye(n_params)]
            ])
            b = np.concatenate([drdy_n, drdp_n])
            adjs = np.linalg.solve(A, b)
            adj_lambda[:, n] = adjs[:n_odes]
            adj_nu[:, n] = adjs[n_odes : (n_odes + n_algs)]
            adj_mu[:, n] = adjs[(n_odes + n_algs) :]
        elif n == 0:
            # Inital timestep
            # ∂L/∂y_0 = ∂C/∂y_0 = (∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T
            # ∂L/∂p_0 = ∂C/∂p_0 = (∂r/∂p_n + μ_{n+1})^T
            dfdy_prev = get_dfdy_prev(y[:, 1], y[:, 0], p, u[:, 0], h)
            dCdy_0 = drdy_n - np.dot(adj_lambda[:, 1].T, dfdy_prev)
            dCdp_0 = drdp_n + adj_mu[:, 1]
        else:
            # [∂f/∂y_n^T  ∂g/∂y_n^T  0] [λ_n]   [(∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T]
            # [∂f/∂p_n^T  ∂g/∂p_n^T  I] [ν_n] = [(∂r/∂p_n + μ_{n+1})^T                               ]
            #                           [μ_n]
            dfdy_prev = get_dfdy_prev(y[:, n+1], y[:, n], p, u[:, n], h)
            A = np.block([
                [dfdy_n.T, dgdy_n.T, np.zeros((n_states, n_params))],
                [dfdp_n.T, dgdp_n.T, np.eye(n_params)]
            ])
            b = np.concatenate([
                drdy_n - np.dot(adj_lambda[:, n+1].T, dfdy_prev),
                drdp_n + adj_mu[:, n+1]
            ])
            adjs = np.linalg.solve(A, b)
            adj_lambda[:, n] = adjs[:n_odes]
            adj_nu[:, n] = adjs[n_odes : (n_odes + n_algs)]
            adj_mu[:, n] = adjs[(n_odes + n_algs) :]

        # ∂L/∂u_n = ∂C/∂u_n = -λ_(n+1) ∂f(y_(n+1), y_n, p_(n+1), u_n)/∂u_n - ν_(n+1) ∂g(y_(n+1), p_(n+1), u_n)/∂u_n
        dCdu[:, n - 1] = - adj_lambda[:, n].T @ dfdu_n - adj_nu[:, n].T @ dgdu_n

    return dCdy_0, dCdp_0, dCdu

def fd_gradients(y0, p, u, h, n_steps):
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


def plot(y, steps, h):
    # Create time array
    t = np.linspace(0, steps * h, steps + 1)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, y[0], label="y_0")
    plt.plot(t, y[1], label="y_1")
    plt.plot(t, y[2], label="y_2")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("DAE Simulation Results")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    steps = 1000
    # steps = 4
    h = 0.0001  # timestep
    p = [1.0, 2.0]
    y_0 = [2, 2, -3]

    u = np.zeros((2, steps))
    u[0] = np.ones((steps)) * 0.2
    u[1] = np.ones((steps)) * 0.6

    y = solve(y_0, p, u, h, steps)
    print("solution:", y[:, -1])
    plot(y, steps, h)

    # FD derivatives
    dCdy_0_fd, dCdp_fd, dCdu_fd = fd_gradients(y_0, p, u, h, steps)
    print("(finite diff) df/dy0", dCdy_0_fd)
    print("(finite diff) df/dp: ", dCdp_fd)
    print("(finite diff) df/du_3: ", dCdu_fd)

    # Adjoint derivatives
    dCdy_0_adj, dCdp_adj, dCdu_adj = adjoint_gradients(y, p, u, h, steps)
    print("(adjoint) df/dy_0: ", dCdy_0_adj)
    print("(adjoint) df/dp: ", dCdp_adj)
    print("(finite diff) df/du_3: ", dCdu_adj[:, 3])

    # Discrepancies
    print(f"Discrepancy df/dy_0: {np.abs(dCdy_0_adj - dCdy_0_fd) / (np.abs(dCdy_0_fd) + 1e-6) * 100}%")
    print(f"Discrepancy df/dp: {np.abs(dCdp_adj - dCdp_fd) / (np.abs(dCdp_fd) + 1e-6) * 100}%")
    print(f"Discrepancy df/du: {np.abs(dCdu_adj[:, 3] - dCdu_fd) / (np.abs(dCdu_fd) + 1e-6) * 100}%")


if __name__ == "__main__":
    main()
