import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

"""
Here, I assume that the power provided by the condenser is constaint
no matter the water temperature.
So it can heat water at 10K as well as water at 10000K.
It's not reasonable, because in a real heat pump you can't reach more than
~70º, so the COP would tend to 0 as you approximate those temperatures.
"""

def dae_system(y, y_prev, p, u_prev, h):
    r"""
    Solve the following system of non-linear equations

    ∘ Q_dot_cond = COP * P_comp # W

    ∘ Q_dot_cond = m_dot_cond * cp_water * (T_cond - T_tank)

    ∘ m_tank * cp_water * (dT_tank/dt)
        = m_dot_cond * cp_water * T_cond
        + m_dot_load * cp_water * T_load
        - m_dot_tank * cp_water * T_tank
        - Q_dot_loss

    ∘ Q_dot_loss = U * A * (T_tank - T_amb)

    ∘ Q_dot_load = load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb)

    ∘ Q_dot_load = m_dot_load * cp_water * (T_tank - T_load)


    with unknowns:
    T_tank, T_load, T_cond, Q_dot_load


    Discretized with Backward Euler
    COP * P_comp - m_dot_cond * cp_water * (T_cond - T_tank)

    m_tank * cp_water * ((T_tank - T_tank_prev)/h)
        - m_dot_cond * cp_water * T_cond
        - m_dot_load * cp_water * T_load
        + m_dot_tank * cp_water * T_tank
        + U * A * (T_tank - T_amb)
        = 0

    Q_dot_load - load_hx_eff * m_dot_load * cp_water * (T_tank - T_amb) = 0

    Q_dot_load - m_dot_load * cp_water * (T_tank - T_load) = 0


    Making:
    y[0] = T_tank
    y[1] = T_cond
    y[2] = T_load
    y[3] = Q_dot_load

    p[0] = COP
    p[1] = cp_water
    p[2] = m_tank
    p[3] = U
    p[4] = A
    p[5] = T_amb
    p[6] = load_hx_eff

    u_prev[0] = P_comp
    u_prev[1] = m_dot_cond
    u_prev[2] = m_dot_load
    m_dot_tank = u_prev[1] + u_prev[2]


    p[0] * u_prev[0] - u_prev[1] * p[1] * (y[1] - y[0]) = 0
    p[2] * p[1] * ((y[0] - y_prev[0])/h) - u_prev[1] * p[1] * y[1]
        - u_prev[2] * p[1] * y[2]
        + (u_prev[1] + u_prev[2]) * p[1] * y[0]
        + p[3] * p[4] * (y[0] - p[5])
        = 0
    y[3] - p[6] * u_prev[2] * p[1] * (y[0] - p[5]) = 0
    y[3] - u_prev[2] * p[1] * (y[0] - y[2]) = 0

    Which are divided in differential equations (f(y, y_prev, p) = 0)
    and algebraic equations (g(y,p) = 0)
    """
    return [
        p[0] * u_prev[0] - u_prev[1] * p[1] * (y[1] - y[0]),
        p[2] * p[1] * ((y[0] - y_prev[0]) / h)
        - u_prev[1] * p[1] * y[1]
        - u_prev[2] * p[1] * y[2]
        + (u_prev[1] + u_prev[2]) * p[1] * y[0]
        + p[3] * p[4] * (y[0] - p[5]),
        y[3] - p[6] * u_prev[2] * p[1] * (y[0] - p[5]),
        y[3] - u_prev[2] * p[1] * (y[0] - y[2]),
    ]


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

        # COP * P_comp - m_dot_cond * cp_water * (T_cond - T_tank)
        # print("COP: ", p[0])
        # print("P_comp", u[0, n])
        # print("m_dot_cond", u[1, n])
        # print("cp_water", p[1])
        # print("T_cond", y[0, n])
        # print("T_tank", y[1, n])
        # print(p[0] * u[0, n])
        # print(u[1, n] * p[1] * (y[1, n] - y[0, n]))
        # print(p[0] * u[0, n] - u[1, n] * p[1] * (y[1, n] - y[0, n]))

        # m_tank * cp_water * ((T_tank - T_tank_prev)/h)
        #     - m_dot_cond * cp_water * T_cond
        #     - m_dot_load * cp_water * T_load
        #     + m_dot_tank * cp_water * T_tank
        #     + U * A * (T_tank - T_amb)
        #     = 0
        # print("-------------")
        # print("internal power: ", p[2] * p[1] * ((y[0, n+1] - y[0, n]) / h))
        # Q_cond = u[1, n] * p[1] * y[1, n+1]
        # print("Q_cond: ", Q_cond)
        # print("Q_load: ", u[2, n] * p[1] * y[2, n+1])
        # Q_tank = (u[1, n] + u[2, n]) * p[1] * y[0, n+1]
        # print("Q_tank: ", Q_tank)
        # print("Q_cond - Q_tank: ", Q_cond - Q_tank)
        # print("Q_loss: ", p[3] * p[4] * (y[0, n+1] - p[5]))
        # if n > 10:
        #     print("m_dot_cond: ", u[1, n])
        #     print("m_dot_tank: ", u[1, n] + u[2, n])
        #     exit(0)

    return y


# def plot(y, n_steps, h):
#     # Create time array
#     t = np.linspace(0, n_steps * h, n_steps + 1)
#
#     # Plot the results
#     plt.figure(figsize=(10, 6))
#     plt.plot(t, y[0], label="T_tank")
#     plt.plot(t, y[1], label="T_load")
#     plt.plot(t, y[2], label="T_cond")
#     plt.plot(t, y[3], label="Q_dot_load")
#     plt.xlabel("Time")
#     plt.ylabel("Values")
#     plt.title("DAE Simulation Results")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def plot(y, u, n_steps, h):
    # Create time array
    t = np.linspace(0, n_steps * h, n_steps + 1)

    # Create the figure and the subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # First subplot for T_tank, T_load, and T_cond
    axes[0].plot(t, y[0], label="T_tank")
    axes[0].plot(t, y[1], label="T_cond")
    axes[0].plot(t, y[2], label="T_load")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Temperature Profiles")
    axes[0].legend()
    axes[0].grid(True)

    # Second subplot for Q_dot_load
    axes[1].plot(t, y[3], label="Q_dot_load", color='tab:orange')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Q_dot_load (W)")
    axes[1].set_title("Heat Load")
    axes[1].legend()
    axes[1].grid(True)

    # # Control variables
    # axes[2].plot(t[:-1], u[0], label="P_comp")
    # # axes[2].plot(t, u[1], label="m_dot_cond")
    # # axes[2].plot(t, u[2], label="m_dot_load")
    # axes[2].set_ylabel("Compressor power")
    # axes[2].legend()
    # axes[2].grid(True)

    # Third subplot for control variables
    axes[2].plot(t[:-1], u[0], label="P_comp")
    axes[2].set_ylabel("Compressor power")
    axes[2].set_title("Control Variables")
    axes[2].legend(loc='upper left')
    axes[2].grid(True)

    # Create a secondary y-axis for u[1] and u[2]
    ax2 = axes[2].twinx()
    ax2.plot(t[:-1], u[1], label="m_dot_cond", color='tab:red')
    ax2.plot(t[:-1], u[2], label="m_dot_load", color='tab:green')
    ax2.set_ylabel("Mass flow rates")
    ax2.legend(loc='upper right')

    # Set common x-axis label
    axes[2].set_xlabel("Time (s)")


    # Show the plots
    plt.tight_layout()
    plt.show()


def main():
    time = 1000000 # s
    h = 1000  # timestep
    # n_steps = int(1e5)
    n_steps = int(time/h)
    # n_steps = int(5)
    y0, p, u = get_inputs(n_steps, h)
    y = solve(y0, p, u, h, n_steps)
    print("solution:", y[:, -1])
    plot(y, u, n_steps, h)


def get_inputs(n_steps, h):
    # Parameters
    U = 0.04  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
    # U = 1  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
    T_0 = 343  # [K] Initial temperature of water in the tank (70°C)
    V = 1  # Tank volume (m3)
    A = 4 * 3.141592 * (V / (2 * 3.1415)) ** (2 / 3)  # Tank surface area (m2)
    rho_water = 1000  # Water density (kg/m3)
    cp_water = 4186  # Specific heat capacity of water (J/(kg·K))
    m_tank = V * rho_water  # Mass of water in the tank (kg)
    print("m_tank: ", m_tank)
    T_amb = 298  # [K] (25ºC)
    COP = 3
    load_hx_eff = 0.8

    # p[0] = COP
    # p[1] = cp_water
    # p[2] = m_tank
    # p[3] = U
    # p[4] = A
    # p[5] = T_amb
    # p[6] = load_hx_eff
    p = [COP, cp_water, m_tank, U, A, T_amb, load_hx_eff]

    # y[0] = T_tank
    # y[1] = T_cond
    # y[2] = T_load
    # y[3] = Q_dot_load
    y0 = [298, 400, 300, 10]

    # u_prev[0] = P_comp
    # u_prev[1] = m_dot_cond
    # u_prev[2] = m_dot_load
    # m_dot_tank = m_dot_cond + m_dot_load
    # Fix control variables
    P_comp_max = 10000  # W
    total_time = h * n_steps
    f = 1 / total_time
    desired_frequency = 10 * f
    time_steps = np.arange(n_steps) * h
    # P_comp = P_comp_max * np.sin(2 * np.pi * desired_frequency * time_steps) + P_comp_max
    P_comp = np.ones((n_steps)) * P_comp_max
    # P_comp = P_comp_max * np.sin(2 * np.pi * f * 100 * np.arange(n_steps)) + P_comp_max
    P_comp[-int(n_steps/3):] = 0
    m_dot_cond = np.ones((n_steps)) * 0.3  # kg/s
    # m_dot_cond[:20000] = 0.3
    m_dot_load = np.ones((n_steps)) * 0.1 # kg/s
    u = np.zeros((3, n_steps))
    u[0, :] = P_comp  # P_comp
    u[1, :] = m_dot_cond  # m_dot_cond
    # u[2, :] = m_dot_load  # m_dot_tank
    u[2, :] = 0  # m_dot_tank

    return y0, p, u


if __name__ == "__main__":
    main()
