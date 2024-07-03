import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation


# Define the DAE system
def dae_system(y, y_prev, p, h):
    r"""
    Solve
    dy[0]/dt + p[0]*y[0] - p[1]*y[1] + p[1]*y[1]*y[0] + p[1]*y[1]^2 = 0
    dy[1]/dt - p[1]*y[0]^2 + y[1] = 0
    y[2] - 1 + y[0] + y[1] = 0

    Discretized with Backward Euler

    y[0] - y_prev[0] + h*p[0]*y[0] - h*p[1]*y[1] + h*p[1]*y[1]*y[0] + h*p[1]*y[1]^2 = 0
    y[1] - y_prev[1] - h*p[1]*y[0]^2 + h*y[1] = 0
    y[2] - 1 + y[0] + y[1] = 0

    Which are divided in differential equations (f(y, y_prev, p) = 0)
    y[0] - y_prev[0] + h*p[0]*y[0] - h*p[1]*y[1] + h*p[1]*y[1]*y[0] + h*p[1]*y[1]^2 = 0
    y[1] - y_prev[1] - h*p[1]*y[0]^2 + h*y[1] = 0

    and algebraic equations (g(y,p) = 0)
    y[2] - 1 + y[0] + y[1] = 0
    """
    return [
        y[0] - y_prev[0] + h * p[0] * y[0] - h * p[1] * y[1] + h * p[1] * y[1] * y[0] + h * p[1] * y[1] ** 2,
        y[1] - y_prev[1] - h * p[1] * y[0] ** 2 + h * y[1],
        y[2] - 1 + y[0] + y[1],
    ]


def get_dfdy(y, y_prev, p, h):
    # Compute ∂f(y_n,p)/∂y_n
    dfdy = np.zeros((2, 3))
    dfdy[0, 0] = 1 + h * p[0] + h * p[1] * y[1]
    dfdy[0, 1] = -h * p[1] + h * p[1] * y[0] + 2 * h * p[1] * y[1]
    dfdy[0, 2] = 0

    dfdy[1, 0] = -2 * h * p[1] * y[0]
    dfdy[1, 1] = 1 + h
    dfdy[1, 2] = 0
    return dfdy


def get_dfdy_prev(y, y_prev, p, h):
    # Compute ∂f(y_n,p)/∂y_n
    dfdy_prev = np.zeros((2, 3))
    dfdy_prev[0, 0] = -1
    dfdy_prev[0, 1] = 0
    dfdy_prev[0, 2] = 0

    dfdy_prev[1, 0] = 0
    dfdy_prev[1, 1] = -1
    dfdy_prev[1, 2] = 0
    return dfdy_prev


def get_dfdp(y, p, h):
    # Compute ∂f(y_n,p)/∂p
    dfdp = np.zeros((2, 2))
    dfdp[0, 0] = h * y[0]
    dfdp[0, 1] = -h * y[1] + h * y[1] * y[0] + h * y[1] ** 2

    dfdp[1, 0] = 0
    dfdp[1, 1] = -h * y[0] ** 2
    return dfdp


def get_dgdy(y, p, h):
    # Compute ∂g(y_n,p_n)/∂y_n
    dgdy = np.zeros((1, 3))
    dgdy[0, 0] = 1
    dgdy[0, 1] = 1
    dgdy[0, 2] = 1
    return dgdy


def get_dgdp(y, p, h):
    # Compute ∂g(y_n,p_n)/∂p_n
    dgdp = np.zeros((1, 2))
    dgdp[0, 0] = 0
    dgdp[0, 1] = 0
    return dgdp


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


def solve(y_0, p, h, steps):
    # Initialize parameters
    y = np.zeros((3, steps + 1))

    # Initial conditions
    y[0][0] = y_0[0]
    y[1][0] = y_0[1]
    y[2][0] = y_0[2]

    # Time-stepping loop
    for n in range(steps):
        # Use fsolve to solve the nonlinear system
        y_next = fsolve(dae_system, y[:, n], args=(y[:, n], p, h))
        y[0][n + 1] = y_next[0]
        y[1][n + 1] = y_next[1]
        y[2][n + 1] = y_next[2]

    return y


def adjoint_gradients(y, p, h, steps):
    r"""
    Adjoint gradients of the cost function with respect to parameters

    min_p C = Σr(y_n, p)
    s.t.
        f(y_n, y_(n-1), p) = 0
        g(y_n, p) = 0

    where f are the differential constraints
    and g are the algebraic constraints

    by agumenting the system
    p' = 0

    min_{p_n} C = Σr(y_n, p_n)
        f(y_n, y_(n-1), p_n) = 0
        g(y_n, p_n) = 0
        (p_n - p_(n-1)) / h = 0

    Lagrangian:
    L = Σr(y_n, p_n)
        - Σ λ_n f(y_n, y_(n-1), p_n)
        - Σ ν_n g(y_n, p_n)
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
    dfdy = get_dfdy(y[:, 1], y[:, 0], p, h)
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

    # Backward propagation of adjoint variables
    for n in range(steps, -1, -1):
        dfdy_n = get_dfdy(y[:, n], y[:, n - 1], p, h)
        dfdp_n = get_dfdp(y[:, n], p, h)
        dgdy_n = get_dgdy(y[:, n], p, h)
        dgdp_n = get_dgdp(y[:, n], p, h)
        drdy_n = get_drdy(y[:, n], p, h)
        drdp_n = get_drdp(y[:, n], p, h)

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
            adj_nu[:, n] = adjs[n_odes]
            adj_mu[:, n] = adjs[n_odes+1:]
        elif n == 0:
            # Inital timestep
            # ∂L/∂y_0 = ∂C/∂y_0 = (∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T
            # ∂L/∂p_0 = ∂C/∂p_0 = (∂r/∂p_n + μ_{n+1})^T
            dfdy_prev = get_dfdy_prev(y[:, 1], y[:, 0], p, h)
            dCdy_0 = drdy_n - np.dot(adj_lambda[:, 1].T, dfdy_prev)
            dCdp_0 = drdp_n + adj_mu[:, 1]
        else:
            # [∂f/∂y_n^T  ∂g/∂y_n^T  0] [λ_n]   [(∂r/∂y_n - λ_{n+1} ∂f(y_{n+1}, y_n, p_{n+1})/∂y_n)^T]
            # [∂f/∂p_n^T  ∂g/∂p_n^T  I] [ν_n] = [(∂r/∂p_n + μ_{n+1})^T                               ]
            #                           [μ_n]
            dfdy_prev = get_dfdy_prev(y[:, n+1], y[:, n], p, h)
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
            adj_nu[:, n] = adjs[n_odes]
            adj_mu[:, n] = adjs[n_odes+1:]

    return dCdy_0, dCdp_0


def fd_gradients(y_0, p, h, steps):
    """
    Finite difference of function cost_function with respect parameters

    f'(x) = (f(y+h) - f(y)) / h
    """
    delta = 1e-3

    # Initial solution
    y = solve(y_0, p, h, steps)

    y_0_perturbed = y_0.copy()  # Create a new copy of y_0
    y_0_perturbed[0] += delta
    y_perturbed = solve(y_0_perturbed, p, h, steps)
    dfdy0_0 = (cost_function(y_perturbed) - cost_function(y)) / delta

    y_0_perturbed = y_0.copy()  # Create a new copy of y_0
    y_0_perturbed[1] += delta
    y_perturbed = solve(y_0_perturbed, p, h, steps)
    dfdy1_0 = (cost_function(y_perturbed) - cost_function(y)) / delta

    y_0_perturbed = y_0.copy()  # Create a new copy of y_0
    y_0_perturbed[2] += delta
    y_perturbed = solve(y_0_perturbed, p, h, steps)
    dfdy2_0 = (cost_function(y_perturbed) - cost_function(y)) / delta

    p_perturbed = p.copy()  # Create a new copy of p
    p_perturbed[0] += delta
    y_perturbed = solve(y_0, p_perturbed, h, steps)
    dfdp0 = (cost_function(y_perturbed) - cost_function(y)) / delta

    p_perturbed = p.copy()  # Create a new copy of p
    p_perturbed[1] += delta
    y_perturbed = solve(y_0, p_perturbed, h, steps)
    dfdp1 = (cost_function(y_perturbed) - cost_function(y)) / delta

    return [dfdy0_0, dfdy1_0, dfdy2_0], [dfdp0, dfdp1]


# def fd_central_gradients(y_0, p, h, steps):
#     delta = 1e-6
#     gradients = []
#     for i in range(len(p)):
#         p_plus = p.copy()
#         p_plus[i] += delta
#         y_plus = solve(y_0, p_plus, h, steps)
#         cost_plus = cost_function(y_plus)
#
#         p_minus = p.copy()
#         p_minus[i] -= delta
#         y_minus = solve(y_0, p_minus, h, steps)
#         cost_minus = cost_function(y_minus)
#
#         grad = (cost_plus - cost_minus) / (2 * delta)
#         gradients.append(grad)
#     return gradients


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
    # steps = 100000
    steps = 3
    h = 0.0001  # timestep
    p = [1.0, 2.0]
    y_0 = [2, 2, -3]
    y = solve(y_0, p, h, steps)
    print("solution:", y[:, -1])
    # plot(y, steps, h)

    # Derivatives
    dCdy_0_fd, dCdp_fd = fd_gradients(y_0, p, h, steps)
    print("(finite diff) df/dy_0", dCdy_0_fd)
    print("(finite diff) df/dp: ", dCdp_fd)

    # Adjoint derivatives
    dCdy_0_adj, dCdp_adj = adjoint_gradients(y, p, h, steps)
    print("(adjoint) df/dy_0: ", dCdy_0_adj)
    print("(adjoint) df/dp: ", dCdp_adj)

    # Discrepancies
    print(f"Discrepancy df/dy_0: {np.abs(dCdy_0_adj - dCdy_0_fd) / (np.abs(dCdy_0_fd) + 1e-6) * 100}%")
    print(f"Discrepancy df/dp: {np.abs(dCdp_adj - dCdp_fd) / (np.abs(dCdp_fd) + 1e-6) * 100}%")


if __name__ == "__main__":
    main()
