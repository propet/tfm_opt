# PARAMS = {
#     "N_HOURS": 8000,  # 8760 in a year
#     "MAX_BAT_CAPACITY": 120,  # kWh
#     "SOC_MIN": 0.1,  # Minimum State of Charge
#     "SOC_MAX": 0.9,  # Maximum State of Charge
#     "P_BAT_MAX": 10,  # kW
#     "P_GRID_MAX": 10,  # kW
#     "MAX_SOLAR_RADIATION": 10,  # kW
#     "MAX_ELECTRIC_DEMAND": 20,  # kW
#     "DK_RHO": 50,  # Kreisselmeier–Steinhauser factor
#     "BAT_ETA_CHARGE": 0.8,  # charge efficiency
#     "BAT_ETA_DISCHARGE": 0.9,  # discharge efficiency
# }

PARAMS = {
    "H": 1000,  # stepsize [s]
    # "T0": 100 * 24 * 3600,
    "T0": 0,
    "HORIZON": 1000000,  # s
    # "HORIZON": 365 * 24 * 3600,  # s
    # "N_HOURS": 1000,  # 8760 in a year

    "MAX_BAT_CAPACITY": 13000,  # W·h
    "SOLAR_SIZE": 3000,  # installed power in solar panels W
    "E_BAT_MAX": 13000 * 3600,  # 13000 Wh -> (13000*3600) W·s
    "SOC_MIN": 0.1,  # Minimum State of Charge
    "SOC_MAX": 0.9,  # Maximum State of Charge
    "P_BAT_MAX": 5000,  # W
    "P_GRID_MAX": 5000,  # W
    "MAX_SOLAR_RADIATION": 10000,  # W
    "MAX_HEAT_POWER": 10000,  # W
    "MAX_ELECTRIC_DEMAND": 4000,  # W
    "DK_RHO": 50,  # Kreisselmeier–Steinhauser factor
    "BAT_ETA_CHARGE": 0.8,  # charge efficiency
    "BAT_ETA_DISCHARGE": 0.9,  # discharge efficiency
    "BAT_ETA": 0.9486832980505138,  # charge/discharge efficiency (includes losses in DC/AC conversion). from round-trip efficiency of tesla powerwall


    "CP_WATER": 4186,  # Specific heat capacity of water (J/(kg·K))
    "RHO_WATER": 1000,  # Water density (kg/m3)
    "MU_WATER": 0.001,  # Water dynamic viscosity (Pa·s)
    "K_WATER": 0.6, # Water thermal conductivity [W/(m·K)]
    "PR_WATER": 6.9,  # Prandtl number (dimensionless) for water at 300K $$Pr = \frac{c_p \mu}{k_water}$$
    "TANK_VOLUME": 0.1,  # m3
    "MAX_TANK_VOLUME": 1000, # m3
    "MIN_TANK_VOLUME": 1, # m3
    "U": 0.245,  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
    "T_AMB": 300,  # K (27ºC)
    "P_COMPRESSOR_MAX": 3000,  # W

    "LOAD_HX_EFF": 0.8,

    # "M_DOT_COND_MAX": 1, # kg/s
    # "M_DOT_LOAD_MAX": 1, # kg/s
    "M_DOT_COND_MAX": 0.5, # kg/s
    "M_DOT_LOAD_MAX": 0.5, # kg/s

    "MAX_Q_DOT_REQUIRED": 8000,  # W
    "FLOOR_WIDTH": 10,  # m2 (quare plant)
    "FLOOR_AREA": 100,  # m2
    "FLOOR_VOLUME": 100 * 0.05,  # [m3], area 100 * 5cm deep
    "FLOOR_MASS": 2300 * 100 * 0.05,  # floor mass [kg], rho_concrete * floor_volume
    "CP_CONCRETE": 880,  # Specific heat capacity of concrete (J/(kg·K))
    "K_CONCRETE": 1.4,  # concrete thermal conductivity [W/(m·K)]
    "RHO_CONCRETE": 2300,  # density of concrete [kg/m3]
    "EPSILON_CONCRETE": 0.93,  # emissivity of concrete at 300K (dimensionless)
    "T_TARGET": 298,  # K
    # "T_TARGET": 293,  # K
    "ROCK_WOOL_U": 0.245,  # W/(m2·K)
    "ROCK_WOOL_AREA": 175,  # m2
    "WINDOWS_U": 1.3,  # W/(m2·K)
    "WINDOWS_AREA": 25,  # m2

    "K_PEX": 0.41, # PEX thermal conductivity [W/(m·K)]
    "TUBE_INNER_DIAMETER": 0.022225,  # tube inner diameter [m], for 1" pex tube
    "TUBE_OUTER_DIAMTER": 0.028575,  # tube outer dimater [m], for 1" pex tube
    "TUBE_THICKNESS": 0.00635,  # tube thickness [m], for 1" pex tube
    "TUBES_LENGTH": 21,  # [m] with spacing of 0.5m between tubes
    "A_TUBES": 21 * 3.1415 * 0.028575,  # tubes surface area [m2], length * pi * outer_diameter

    "K_AIR": 0.0263,  # air at 300K thermal conductivity [W/(m·K)]
    "PR_AIR": 0.707,  # prandtl number for air at 300K
    "CP_AIR": 1007,  # Specific heat capacity of air at 300K (J/(kg·K))
    "RHO_AIR": 1.1614,  # Air density at 300K (kg/m3)
    "MU_AIR": 18.46e-6,  # air dynamic viscosity of air at 300K [kg/(m·s)]
    "NU_AIR": 15.89e-6, # air kinematic viscosity at 300K [m2/s]
    "AIR_VOLUMETRIC_EXPANSION_COEFF": 1 / 300,  # β = 1/T for ideal gases like air [K-1]

    "STEFAN_BOLTZMANN_CONSTANT": 5.67e-8,  # [W/(m2·K4)]
    "GRAVITY_ACCELERATION": 9.8,  # [m/s2]

}
