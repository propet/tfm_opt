import numpy as np
import matplotlib.pyplot as plt

# Default values for the parameters
rho_water = 1000 # Water density (kg/m3)
c_p = 4186  # Specific heat capacity of water (J/(kg·K))
m_dot_cond = 1  # Mass flow rate of water from the condenser (kg/s)
m_dot_load = 2  # Mass flow rate of water from the load (kg/s)
m_dot_tank = m_dot_cond + m_dot_load
T_out_c = 90  # Temperature of water from the condenser (°C)
T_out_L = 25  # Temperature of water from the load (°C)
T_amb = 20  # Ambient temperature (°C)
U = 0.04  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
T_0 = 70  # Initial temperature of water in the tank (°C)
V = 1 # Tank volume (m3)
A = 4 * 3.141592 * (V / (2 * 3.1415))**(2/3) # Tank surface area (m2)
m_tank = V * rho_water  # Mass of water in the tank (kg)


# Calculate the constants
C = (m_dot_cond * c_p * T_out_c + m_dot_load * c_p * T_out_L + U * A * T_amb) / (m_dot_tank * c_p + U * A)
B = T_0 - C

# Time array
t = np.linspace(0, 3000, 1000)

# Temperature array
T_tank = (T_0 - C) * np.exp(-t * (m_dot_tank * c_p + U * A) / (m_tank * c_p)) + C

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(t, T_tank, 'b-', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel(r'$T_{tank}$ (°C)', fontsize=14)
plt.grid(True)
plt.show()
