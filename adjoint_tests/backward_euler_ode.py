import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation


def ode_system(y, y_previous, h, p):
    r"""
    Solve with backward euler discretization (y_{n+1} = y_n + h \cdot f(y_{n+1}, t_{n+1}))

    dy0/dt = p0*y0 - p1*y0*y1
    dy1/dt = -p2*y1 + p3*y0*y1

    (y[0] - y_previous[0]) / h) = p[0] * y[0] - p[1] * y[0] * y[1]
    (y[1] - y_previous[1]) / h) = -p[2] * y[1] + p[3] * y[0] * y[1]

    Same as

    0 = -((y[0] - y_previous[0]) / h) + p[0] * y[0] - p[1] * y[0] * y[1]
    0 = -((y[1] - y_previous[1]) / h) - p[2] * y[1] + p[3] * y[0] * y[1]

    Same as

    0 = -y[0] + y_previous[0] + h * p[0] * y[0] - h * p[1] * y[0] * y[1]
    0 = -y[1] + y_previous[1] - h * p[2] * y[1] + h * p[3] * y[0] * y[1]

    Same as
    0 = -y[0] + y_previous[0] + h * p[0] * y[0] - h * p[1] * y[0] * y[1]
    0 = -y[1] + y_previous[1] - h * p[2] * y[1] + h * p[3] * y[0] * y[1]
    """
    return [
        -y[0] + y_previous[0] + h * p[0] * y[0] - h * p[1] * y[0] * y[1],
        -y[1] + y_previous[1] - h * p[2] * y[1] + h * p[3] * y[0] * y[1]
    ]


def get_dfdy(y, p):
    # y_n = y_{n-1} + h \cdot f(y_n, t_n)
    # Compute ∂f(y_n;p)/∂y_n
    # y[0] = y_previous[0] + h * (p[0] * y[0] - p[1] * y[0] * y[1])
    # y[1] = y_previous[1] + h * (-p[2] * y[1] + p[3] * y[0] * y[1])
    dfdy = np.zeros((2, 2))
    dfdy[0, 0] = p[0] - p[1] * y[1]
    dfdy[0, 1] = -p[1] * y[0]
    dfdy[1, 0] = p[3] * y[1]
    dfdy[1, 1] = -p[2] + p[3] * y[0]
    return dfdy


def get_dfdp(y, p):
    # y_n = y_{n-1} + h \cdot f(y_n, t_n)
    # Compute ∂f(y_n;p)/∂p
    # y[0] = y_previous[0] + h * (p[0] * y[0] - p[1] * y[0] * y[1])
    # y[1] = y_previous[1] + h * (-p[2] * y[1] + p[3] * y[0] * y[1])
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
        # print("forward step:", n)
        y_next = fsolve(ode_system, y[:, n], args=(y[:, n], h, p))
        y[0][n + 1] = y_next[0]
        y[1][n + 1] = y_next[1]

    return y


def adjoint_equation(h, lambd_np1, dfdy_n, dCdy_n):
    # ∂C/∂y_n - λ_n + λ_{n+1} + λ_n * h * ∂f(y_n;p)/∂y_n = 0
    # (I - h ∂f(y_n;p)/∂y_n) λ_n = ∂C/∂y_N + λ_{n+1}

    I = np.eye(dfdy_n.shape[0])
    A = I - h * dfdy_n
    b = dCdy_n + lambd_np1

    # Solve the linear system A * λ_n = b
    lambd_n = np.linalg.solve(A, b)
    return lambd_n


def adjoint_gradients(y, p, h, steps):
    r"""
    Adjoint gradients of the cost function with respect to parameters

    min_p C
    s.t.
        0 = -y[0] + y_previous[0] + h * (p[0] * y[0] - p[1] * y[0] * y[1])
        0 = -y[1] + y_previous[1] + h * (-p[2] * y[1] + p[3] * y[0] * y[1])


    with C = Σ y[0]_n

    Lagragian:
    L = C + Σ λ_n (-y_n + y_{n-1} + h * f(y_n;p))
    ∂L/∂y_n = ∂C/∂y_n - λ_n + λ_{n+1} + λ_n * h * ∂f(y_n;p)/∂y_n = 0
    ∂L/∂y_N = 0 = ∂C/∂y_N - λ_N + λ_N * h * ∂f(y_N;p)/∂y_N => λ_N
    .
    .
    ∂C/∂y_n - λ_n + λ_{n+1} + λ_n * h * ∂f(y_n;p)/∂y_n = 0

    ∂L/∂p = ∂C/∂p = Σ λ_n * h * (∂f(y_n;p)/∂p)
    """
    # Initialize adjoint variables
    lambd = np.zeros((2, steps + 1))
    # ∂L/∂y_N = 0 = ∂C/∂y_N - λ_N + λ_N * h * ∂f(y_n;p)/∂y_N
    dCdy_N = [1, 0]
    dfdy_N = get_dfdy(y[:, -1], p)
    I = np.eye(dfdy_N.shape[0])
    # (I - h ∂f(y_N;p)/∂y_N) λ_N = ∂C/∂y_N
    lambd_N = np.linalg.solve((I - h * dfdy_N.T), dCdy_N)
    lambd[:, -1] = lambd_N  # Set the final condition

    # Backward propagation of adjoint variables
    for n in range(steps-1, -1, -1):
        dfdy_n = get_dfdy(y[:, n], p)
        dCdy_n = np.array([1, 0])
        lambd[:, n] = adjoint_equation(h, lambd[:, n+1], dfdy_n.T, dCdy_n)
        # print(f"lambd_{n}", lambd[:, n])

    dCdy_0 = lambd[:, 0]


    # Compute gradient with respect to parameters
    dCdp = np.zeros(4)
    # ∂L/∂p = ∂C/∂p = Σ λ_n * h * (∂f(y_n;p)/∂p)
    for n in range(steps, 0, -1):
        dfdp = get_dfdp(y[:, n], p)
        dCdp += h * np.dot(lambd[:, n].T, dfdp)

    return dCdy_0, dCdp


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
    plt.plot(t, y[0], label='y_0')
    plt.plot(t, y[1], label='y_1')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('ODE Simulation Results')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    steps = 100000
    # steps = 3
    h = 0.0001  # timestep
    p = [1.0, 2.0, 3.0, 2.0]
    y_0 = [2, 2]
    y = solve(y_0, p, h, steps)
    print("solution:", y[:, -1])
    plot(y, steps, h)

    # Derivatives
    dCdy_0, dCdp = fd_gradients(y_0, p, h, steps)
    print("(finite diff) df/dy_0", dCdy_0)
    print("(finite diff) df/dp: ", dCdp)

    # # Adjoint derivatives
    dCdy_0, dCdp = adjoint_gradients(y, p, h, steps)
    print("(adjoint) df/dy_0: ", dCdy_0)
    print("(adjoint) df/dp: ", dCdp)


if __name__ == "__main__":
    main()
