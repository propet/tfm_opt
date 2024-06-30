import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation


def explicit_solve_ode_system(y, h, p):
    r"""
    Solve with forward euler discretization
    dy0/dt = p0*y0 - p1*y0*y1
    dy1/dt = -p2*y1 + p3*y0*y1

    ((y_next[0] - y[0]) / h) = p[0] * y[0] - p[1] * y[0] * y[1]
    ((y_next[1] - y[1]) / h) = -p[2] * y[1] + p[3] * y[0] * y[1]

    Same as

    ((y_next[0] - y[0]) / h) - p[0] * y[0] + p[1] * y[0] * y[1] = 0
    ((y_next[1] - y[1]) / h) + p[2] * y[1] - p[3] * y[0] * y[1] = 0
    """
    return [
        y[0] + h * (p[0] * y[0] - p[1] * y[0] * y[1]),
        y[1] + h * (-p[2] * y[1] + p[3] * y[0] * y[1])
    ]


def get_dfdy(y, p):
    # λ_n = ∂C/∂y_n + λ_{n+1} + λ_{n+1} * h * ∂f(y_n;p)/∂y_n
    # y_n = y_{n-1} + h \cdot f(y_{n-1}, t_{n-1})
    # Compute ∂f(y_n;p)/∂y_n
    dfdy = np.zeros((2, 2))
    dfdy[0, 0] = p[0] - p[1] * y[1]
    dfdy[0, 1] = -p[1] * y[0]
    dfdy[1, 0] = p[3] * y[1]
    dfdy[1, 1] = -p[2] + p[3] * y[0]
    return dfdy


def get_dfdp(y, p):
    # ∂L/∂p = ∂C/∂p = Σ λ_n * h * (∂f(y_{n-1};p)/∂p)
    # Compute ∂f(y_n;p)/∂p
    dfdp = np.zeros((2, 4))
    dfdp[0, 0] = y[0]
    dfdp[0, 1] = -y[0] * y[1]
    dfdp[0, 2] = 0
    dfdp[0, 3] = 0
    dfdp[1, 0] = 0
    dfdp[1, 1] = 0
    dfdp[1, 2] = -y[1]
    dfdp[1, 3] = y[0] * y[1]
    return dfdp


def get_drdy(y, p):
    # ∂r(y_n,p_n)/∂y_n
    # r = y[0]
    drdy = np.zeros((1, 2))
    drdy[0, 0] = 1
    drdy[0, 1] = 0
    return drdy


def get_drdp(y, p):
    # ∂r(y_n,p_n)/∂p_n
    # r = y[0]
    drdp = np.zeros((1, 4))
    return drdp


def cost_function(y):
    return sum(y[0])


def solve(y_0, p, h, steps):
    # Initialize parameters
    y = np.zeros((2, steps + 1))

    # Initial conditions
    y[0][0] = y_0[0]
    y[1][0] = y_0[1]

    # Time-stepping loop
    for n in range(steps):
        y_next = explicit_solve_ode_system(y[:, n], h, p)
        y[0][n + 1] = y_next[0]
        y[1][n + 1] = y_next[1]

    return y


def adjoint_gradients(y, p, h, steps):
    r"""
    Adjoint gradients of the cost function with respect to parameters

    min_p C = Σr(y_n, p_n)
    s.t.
        y[0] - y_prev[0] - h * (p[0] * y_prev[0] - p[1] * y_prev[0] * y_prev[1]) = 0
        y[1] - y_prev[1] - h * (-p[2] * y_prev[1] + p[3] * y_prev[0] * y_prev[1]) = 0

    by agumenting the system
    p' = 0

    min_p C = Σr(y_n, p_n)
        y[0] - y_prev[0] - h * (p[0] * y_prev[0] - p[1] * y_prev[0] * y_prev[1]) = 0
        y[1] - y_prev[1] - h * (-p[2] * y_prev[1] + p[3] * y_prev[0] * y_prev[1]) = 0
        .
        .
        (p_n - p_prev) / h = 0

    Lagrangian:
    L = Σr(y_n, p_n)
        - Σ λ_n (y_n - y_(n-1) - h * f(y_(n-1), p_(n-1)))
        - Σ μ_n (p_n - p_(n-1))

    ∂L/∂y_n = 0 = ∂r(y_n,p_n)/∂y_n - λ_n + λ_(n+1) + λ_(n+1) * h * ∂f(y_n,p_n)/∂y_n
    ∂L/∂p_n = 0 = ∂r(y_n,p_n)/∂p_n + λ_(n+1) * h * ∂f(y_n,p_n)/∂p_n - μ_n + μ_(n+1)

    Solve system at each step:
    λ_n = λ_(n+1) + λ_(n+1) * h * ∂f(y_n,p_n)/∂y_n + ∂r(y_n,p_n)/∂y_n
    μ_n = μ_(n+1) + λ_(n+1) * h * ∂f(y_n,p_n)/∂p_n + ∂r(y_n,p_n)/∂p_n

    Terminal condition:
    ∂L/∂y_N = 0 = ∂r/∂y_N - λ_N
    ∂L/∂p_N = 0 = ∂r/∂p_N - μ_N

    Solve for initial timestep:
    ∂L/∂y_0 = ∂C/∂y_0 = ∂r(y_0,p_0)/∂y_0 + λ_1 + λ_1 * h * ∂f(y_0,p_0)/∂y_0
    ∂L/∂p_0 = ∂C/∂p_0 = ∂r(y_0,p_0)/∂p_0 + λ_1 * h * ∂f(y_0,p_0)/∂p_0 + μ_1
    """
    # Initialize adjoint variables
    adj_lambda = np.zeros((2, steps + 1))
    adj_mu = np.zeros((4, steps + 1))
    dCdy_0 = np.zeros(2)
    dCdp_0 = np.zeros(4)

    # Backward propagation of adjoint variables
    for n in range(steps, -1, -1):
        dfdy_n = get_dfdy(y[:, n], p)
        dfdp_n = get_dfdp(y[:, n], p)
        drdy_n = get_drdy(y[:, n], p)
        drdp_n = get_drdp(y[:, n], p)

        if n == steps:
            # Terminal condition
            adj_lambda[:, n] = drdy_n
            adj_mu[:, n] = drdp_n
        elif n == 0:
            # ∂L/∂y_0 = ∂C/∂y_0, ∂L/∂p_0 = ∂C/∂p_0
            dCdy_0 = drdy_n + adj_lambda[:, n + 1] + h * np.dot(adj_lambda[:, n + 1].T, dfdy_n)
            dCdp_0 = drdp_n + adj_mu[:, n + 1] + h * np.dot(adj_lambda[:, n + 1].T, dfdp_n)
        else:
            adj_lambda[:, n] = adj_lambda[:, n + 1] + h * np.dot(adj_lambda[:, n + 1].T, dfdy_n) + drdy_n
            adj_mu[:, n] = adj_mu[:, n + 1] + h * np.dot(adj_lambda[:, n + 1].T, dfdp_n) + drdp_n

    return dCdy_0, dCdp_0


def memory_efficient_adjoint_gradients(y, p, h, steps):
    """
    Don't require to store the intermediate adjoints
    λ_n = λ_(n+1) + λ_(n+1) * h * ∂f(y_n,p_n)/∂y_n + θ_(n+1) ∂r(y_n,p_n)/∂y_n
    μ_n = μ_(n+1) + λ_(n+1) * h * ∂f(y_n,p_n)/∂p_n + θ_(n+1) * ∂r(y_n,p_n)/∂p_n
    θ_n = θ_(n+1) = 1
    """
    # Initialize adjoint variables
    # Set the final condition
    adj_lambda = np.array([1, 0])  # ∂C(y_N,p_N)/∂y_N
    adj_mu = np.zeros(4)  # ∂C(y_N,p_N)/∂p_N

    # Backward propagation of adjoint variables
    for n in range(steps - 1, -1, -1):
        dfdy_n = get_dfdy(y[:, n], p)
        dfdp_n = get_dfdp(y[:, n], p)
        drdy_n = get_drdy(y[:, n], p)
        drdp_n = get_drdp(y[:, n], p)
        adj_mu = adj_mu + h * np.dot(adj_lambda, dfdp_n) + drdp_n
        adj_lambda = adj_lambda + h * np.dot(adj_lambda, dfdy_n) + drdy_n

    return adj_lambda, adj_mu


def fd_gradients(y_0, p, h, steps):
    """
    Finite difference of function cost_function with respect parameters

    f'(x) = (f(y+h) - f(y)) / h
    """
    delta = 1e-5

    # Initial solution
    y = solve(y_0, p, h, steps)

    # Perturbations
    y_0_perturbed = y_0.copy()  # Create a new copy of y_0
    y_0_perturbed[0] += delta
    y_perturbed = solve(y_0_perturbed, p, h, steps)
    dfdy0_0 = (cost_function(y_perturbed) - cost_function(y)) / delta

    y_0_perturbed = y_0.copy()  # Create a new copy of y_0
    y_0_perturbed[1] += delta
    y_perturbed = solve(y_0_perturbed, p, h, steps)
    dfdy1_0 = (cost_function(y_perturbed) - cost_function(y)) / delta

    p_perturbed = p.copy()  # Create a new copy of p
    p_perturbed[0] += delta
    y_perturbed = solve(y_0, p_perturbed, h, steps)
    dfdp0 = (cost_function(y_perturbed) - cost_function(y)) / delta

    p_perturbed = p.copy()  # Create a new copy of p
    p_perturbed[1] += delta
    y_perturbed = solve(y_0, p_perturbed, h, steps)
    dfdp1 = (cost_function(y_perturbed) - cost_function(y)) / delta

    p_perturbed = p.copy()  # Create a new copy of p
    p_perturbed[2] += delta
    y_perturbed = solve(y_0, p_perturbed, h, steps)
    dfdp2 = (cost_function(y_perturbed) - cost_function(y)) / delta

    p_perturbed = p.copy()  # Create a new copy of p
    p_perturbed[3] += delta
    y_perturbed = solve(y_0, p_perturbed, h, steps)
    dfdp3 = (cost_function(y_perturbed) - cost_function(y)) / delta

    return [dfdy0_0, dfdy1_0], [dfdp0, dfdp1, dfdp2, dfdp3]


def plot(y, steps, h):
    # Create time array
    t = np.linspace(0, steps * h, steps + 1)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, y[0], label="y_0")
    plt.plot(t, y[1], label="y_1")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("ODE Simulation Results")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # steps = 100000
    steps = 3
    h = 0.0001  # timestep
    p = [1.0, 2.0, 3.0, 2.0]
    y_0 = [2, 2]
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
