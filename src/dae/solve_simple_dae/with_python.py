import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation


# Constants
NUM_STEPS = 100


# Define the DAE system
def dae_system(y_next, y, h):
    r"""
    # Solve with Backward Euler discretization
    \dot{y}_0(t) = -y_0(t) + y_1(t) \\
    0 = y_0(t)^2 + y_1(t)^2 - 1
    """
    return [
        (y_next[0] - y[0]) / h + y_next[0] - y_next[1],
        y_next[0]**2 + y_next[1]**2 - 1  # Algebraic constraint
    ]


def solve(y_0):
    # Initialize parameters
    h = 0.1
    y = np.zeros((2, NUM_STEPS + 1))

    # Initial conditions
    y[0][0] = y_0[0]
    y[1][0] = y_0[1]

    # Time-stepping loop
    for n in range(NUM_STEPS):
        # Use fsolve to solve the nonlinear system
        y_next = fsolve(dae_system, y[:, n], args=(y[:, n], h))
        y[0][n + 1] = y_next[0]
        y[1][n + 1] = y_next[1]

    return y


def animation(y):
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    line, = ax.plot([], [], 'o-', lw=2)
    circ = plt.Circle((0, 0), 1, color='r', fill=False)
    ax.add_patch(circ)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(y[0][:frame], y[1][:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=NUM_STEPS+1, init_func=init, blit=True, interval=100)

    plt.show()


def cost_function(y):
    return y[0][-1]


def fd_gradients():
    """
    Finite difference of function f (last value of x given by DAE) with respect initial condition y[0]

    f'(x) = (f(y+h) - f(y)) / h
    """
    y_0 = [1.0, 0.0]
    y = solve(y_0)

    # Perturb y[0]
    delta = 0.00001
    y_0_perturbed = [y_0[0] + delta, y_0[1]]
    y_perturbed = solve(y_0_perturbed)

    return (cost_function(y_perturbed) - cost_function(y)) / delta


def main():
    # y_0 = [1.0, 0.0]
    # y_0 = [0.5, 0.8660254037]
    # y_0 = [-1.0, 0.0]
    # y_0 = [0.0, -1.0]
    # y_0 = [-0.5, 0.8660254037]
    y_0 = [0.8660254037, -0.5]
    y = solve(y_0)
    print("solution:", y[:, -1])
    animation(y)

    # Derivative of the dae function f with respect to initial condition for state variable x
    # g = fd_gradients()
    # print("df/fx: ", g)


if __name__ == "__main__":
    main()
