import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation



# Define the DAE system
def dae_system(y_next, y, h, p):
    r"""
    Solve with Backward Euler discretization
    \dot{y}_0 + p_0 y_0 - p_1 * y_1 * (1 - y_0 - y_1) = 0 \\
    \dot{y}_1 - p_1 y_0^2 + y_1 = 0 \\
    y_2 - 1 + y_0 + y_1 = 0
    """
    return [
        (y_next[0] - y[0]) / h + p[0] * y_next[0] - p[1] * y_next[1] * (1 - y_next[0] - y_next[1]),
        (y_next[1] - y[1]) / h - p[1] * y_next[0]**2 + y_next[1],
        y_next[2] - 1 + y_next[0] + y_next[1]
    ]


def cost_function(y):
    return sum(y[0])


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
        y_next = fsolve(dae_system, y[:, n], args=(y[:, n], h, p))
        y[0][n + 1] = y_next[0]
        y[1][n + 1] = y_next[1]
        y[2][n + 1] = y_next[2]

    return y


def get_ode_J_current(y, y_next, h, p):
    # Compute ∂Ω(y_n, y_{n+1}, p)/∂y_n
    J = np.zeros((3, 3))
    J[0, 0] = -1/h
    J[1, 1] = -1/h
    return J


def get_ode_J_next(y, y_next, h, p):
    # Compute ∂Ω(y_n, y_{n+1}, p)/∂y_{n+1}
    J = np.zeros((3, 3))
    J[0, 0] = 1/h + p[0] + p[1] * y_next[1]
    J[0, 1] = -p[1] + p[1] * y_next[0] + 2 * p[1] * y_next[1]
    J[1, 0] = -2 * p[1] * y_next[0]
    J[1, 1] = 1/h + 1
    J[2, 0] = 1
    J[2, 1] = 1
    J[2, 2] = 1
    return J


def get_ode_Jp(y_next, p):
    # Compute ∂Ω(y_n, y_{n+1}, p)/∂p
    Jp = np.zeros((3, 2))
    Jp[0, 0] = y_next[0]
    Jp[0, 1] = -y_next[1] * (1 - y_next[0] - y_next[1])
    Jp[1, 1] = -y_next[0]**2
    return Jp


def adjoint_equation(lambd_n, lambd_np1, J_current, J_next, dCdy):
    return dCdy + np.dot(J_next.T, lambd_n) + np.dot(J_current.T, lambd_np1)


def adjoint_gradients(y, p, h, steps):
    r"""
    Adjoint gradients of the cost function with respect to parameters

    min_p C
    s.t. Ω(y_{n+1}, y_n, p) = 0

    with C = Σ y[0]_n

    Lagragian:
    L = C + Σ λ_n+1^T [Ω(y_{n+1}, y_n, p)]

    ∂L/∂y_n = ∂C/∂y_n + λ_n^T ∂Ω(y_n, y_{n-1}, p)/∂y_n + λ_{n+1}^T ∂Ω(y_{n+1}, y_n, p)/∂y_n = 0
    λ_N = ∂C/∂y_N = [1, 0, 0]
    .
    .
    ∂C/∂y_n + λ_n^T ∂Ω(y_{n-1}, y_n, p)/∂y_n + λ_{n+1}^T ∂Ω(y_n, y_{n+1}, p)/∂y_n = 0

    ∂L/∂p = ∂C/∂p = Σ λ_n+1 (∂Ω/∂p)
    """
    # Initialize adjoint variables
    lambd = np.zeros((3, steps + 1))
    lambd[:, -1] = [1, 0, 0]  # Set the final condition

    # To ask:
    # why is λ_N just like:
    # I mean, at the last step N, you would have:
    # ∂L/∂y_n = ∂C/∂y_n + λ_n^T ∂Ω(y_{n-1}, y_n, p)/∂y_n + λ_{n+1}^T ∂Ω(y_n, y_{n+1}, p)/∂y_n = 0
    # , right?

    # WRONG

    # Backward propagation of adjoint variables
    for n in range(steps, 0, -1):
        J_current = get_ode_J_current(y[:, n-1], y[:, n], h, p)
        J_next = get_ode_J_next(y[:, n-1], y[:, n], h, p)
        dCdy = np.array([1, 0, 0])

        # Solve the adjoint equation using fsolve
        lambd[:, n-1] = fsolve(adjoint_equation, lambd[:, n],
                       args=(lambd[:, n], J_current, J_next, dCdy))

        # As linear systems of equations
        # lambd[:, n-1] = -np.linalg.solve(J_current.T, dCdy + J_next.T @ lambd[:, n])

        # epsilon = 1e-10  # Small regularization term
        # A = J_current.T + epsilon * np.eye(3)
        # lambd[:, n-1] = -np.linalg.solve(A, dCdy + J_next.T @ lambd[:, n])

        # lambd[:, n-1] = -np.linalg.lstsq(J_current.T, dCdy + J_next.T @ lambd[:, n], rcond=None)[0]



    print(lambd)
    # print(lambd[:, n])

    # Compute gradient with respect to parameters
    dCdp = np.zeros(2)
    for n in range(steps):
        Jp = get_ode_Jp(y[:, n+1], p)
        dCdp += np.dot(lambd[:, n+1], Jp)

    return dCdp


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

    return [dfdy0_0, dfdy1_0, dfdy2_0, dfdp0, dfdp1]

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
    plt.plot(t, y[0], label='y_0')
    plt.plot(t, y[1], label='y_1')
    plt.plot(t, y[2], label='y_2')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('DAE Simulation Results')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    steps = 1000
    h = 0.001  # timestep
    p = [1.0, 2.0]
    y_0 = [2, 2, -3]
    y = solve(y_0, p, h, steps)
    print("solution:", y[:, -1])
    # plot(y, steps, h)

    # Derivatives
    # g = fd_central_gradients(y_0, p, h, steps)
    g = fd_gradients(y_0, p, h, steps)
    print("(finite diff) df/dp: ", g[3:])
    print("(finite diff) df/dy_0", g[0:2])

    # Adjoint derivatives
    gp = adjoint_gradients(y, p, h, steps)
    print("(adjoint) df/dp: ", gp)


if __name__ == "__main__":
    main()
