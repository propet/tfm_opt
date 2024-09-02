PARAMS = {
    "H": 1000,  # stepsize [s]
    # "H": 100,  # stepsize [s]
    # "H": 10,  # stepsize [s]
    # "T0": 200 * 24 * 3600,
    "T0": 0,
    # "HORIZON": 10000,  # s
    "HORIZON": 365 * 24 * 3600,  # s
    # "HORIZON": 30 * 24 * 3600,  # s
    # "HORIZON": 7 * 24 * 3600,  # s
    # "HORIZON": 1 * 24 * 3600,  # s
    # "N_HOURS": 100000,  # 8760 in a year

    "MAX_BAT_CAPACITY": 13000,  # W·h
    "SOLAR_SIZE": 3000,  # installed power in solar panels W
    "SOLAR_SIZE_MAX": 20000,  # W for a roof of 100m2, since each kW takes 5.2m2
    "E_BAT_MAX": 13000 * 3600,  # 13000 Wh -> (13000*3600) W·s
    "E_BAT_MAX_LIMIT_100KWH": 360000000,  # W·s -> 100kWh
    "E_BAT_MAX_LIMIT_10KWH": 36000000,  # W·s -> 10kWh
    "E_BAT_MAX_LIMIT_1KWH": 3600000,  # W·s -> 1kWh
    "E_BAT_MAX_LIMIT_001KWH": 36000,  # W·s -> 1kWh
    "C_RATE_BAT": 0.3,  # max C-rate for battery
    "V_BAT": 48,  # [V] nominal battery voltage (The most common voltage used for solar batteries are 6V, 12V, 24V and 48 Volts)
    "SOC_MIN": 0.3,  # Minimum State of Charge
    "SOC_MAX": 0.9,  # Maximum State of Charge
    "P_BAT_MAX": 5000,  # W
    "P_BAT_MAX_LIMIT": 0.3 * 100000,  # W for e_bat_max_limit 100kWh: c-rate * e_bat_max[Ws] * (1/3600)[h/s]
    "P_BAT_MAX_LIMIT_10KWH": 0.3 * 10000,  # W for e_bat_max_limit 10kWh
    "P_BAT_MAX_LIMIT_1KWH": 0.3 * 1000,  # [W] for e_bat_max_limit 1kWh
    "P_BAT_MAX_LIMIT_001KWH": 0.3 * 10,  # [W] for e_bat_max_limit 1kWh
    "P_GRID_MAX": 5000,  # W
    "P_GRID_MAX_LIMIT": 20000,  # W
    "MAX_SOLAR_RADIATION": 10000,  # W
    "MAX_HEAT_POWER": 10000,  # W
    "MAX_ELECTRIC_DEMAND": 4000,  # W
    "DK_RHO": 50,  # Kreisselmeier–Steinhauser factor
    "BAT_ETA_CHARGE": 0.8,  # charge efficiency
    "BAT_ETA_DISCHARGE": 0.9,  # discharge efficiency
    "BAT_ETA": 0.9486832980505138,  # charge/discharge efficiency (includes losses in DC/AC conversion). from round-trip efficiency of tesla powerwall

    # Lower Heating Value (LHV): 43.1 MJ/kg
    # density: 832 kg/m3
    # thus: LHV * density = 35859.2 MJ/m3 = 35859.2*10^6 J/m3
    # price is: 1500 $/m3
    # so price
    # so price per joule is: 1500[$/m3] * (1/(35859.2*10^6))[m3/J] = 4.18302695e-8 [$/m3]
    "DIESEL_PRICE": 4.18302695e-8,  # $/(W·s),
    "GENERATOR_EFFICIENCY": 0.3,  # 30%


    "CP_WATER": 4186,  # Specific heat capacity of water (J/(kg·K))
    "RHO_WATER": 1000,  # Water density (kg/m3)
    "MU_WATER_AT_300K": 0.000866,  # Water dynamic viscosity (Pa·s) at 300K
    "MU_WATER_AT_320K": 0.000577,  # Water dynamic viscosity (Pa·s) at 320K
    "K_WATER": 0.6, # Water thermal conductivity [W/(m·K)]
    "PR_WATER": 6.9,  # Prandtl number (dimensionless) for water at 300K $$Pr = \frac{c_p \mu}{k_water}$$
    "TANK_VOLUME": 0.1,  # m3
    "MAX_TANK_VOLUME": 1000, # m3
    "MIN_TANK_VOLUME": 1, # m3
    "U": 0.245,  # (W/(m2·K)) overall coefficient for 20cm of rock wool insulation
    "U_TANK": 0.245,  # (W/(m2·K)) overall coefficient for 20cm of rock wool insulation
    "T_AMB": 300,  # K (27ºC)
    "P_COMPRESSOR_MAX": 3000,  # W
    # "P_COMPRESSOR_MAX_LIMIT": 10000,  # W
    "P_COMPRESSOR_MAX_LIMIT": 5000,  # W

    "LOAD_HX_EFF": 0.8,

    # "M_DOT_COND_MAX": 1, # kg/s
    # "M_DOT_LOAD_MAX": 1, # kg/s
    "M_DOT_COND_MAX": 0.5, # kg/s
    "M_DOT_LOAD_MAX": 0.5, # kg/s
    "M_DOT_HEATING_MAX": 0.5, # kg/s

    "MAX_Q_DOT_REQUIRED": 8000,  # W
    "FLOOR_WIDTH": 10,  # m2 (quare plant)
    "FLOOR_AREA": 100,  # m2
    "FLOOR_VOLUME": 100 * 0.05,  # [m3], area 100 * 5cm deep
    "FLOOR_MASS": 2300 * 100 * 0.05,  # floor mass [kg], rho_concrete * floor_volume
    "CP_CONCRETE": 880,  # Specific heat capacity of concrete (J/(kg·K))
    "K_CONCRETE": 1.4,  # concrete thermal conductivity [W/(m·K)]
    "RHO_CONCRETE": 2300,  # density of concrete [kg/m3]
    "EPSILON_CONCRETE": 0.93,  # emissivity of concrete at 300K (dimensionless)
    "T_TARGET": 293,  # K
    "ROCK_WOOL_U": 0.245,  # W/(m2·K) for 20cm of rock wool insulation
    "ROCK_WOOL_AREA": 175,  # m2
    "A_WALLS": 100,  # m2, square of 10m side, with height 2.5: 10 * 2.5 * 4 = 100
    "A_ROOF": 100,  # m2, same as floor area
    "WINDOWS_U": 1.4,  # W/(m2·K)
    "WINDOWS_AREA": 25,  # m2
    "ROOM_VOLUME": 250,  # [m3], 100m2 de planta y 2.5m de alto
    "ROOM_AIR_MASS": 1.1614 * 250,  # [kg], rho_air * volume_room

    "H_WALL_IN": 8.9, # W/(m2·K) still air, vertical non reflective wall
    "H_WALL_OUT": 45, # W/(m2·K) moving air (mean of 6.7m/s and 3.4m/s values)
    "H_ROOF_IN": 9.26, # W/(m2·K) still air, horizontar surface with upwards heat
    "H_ROOF_OUT": 45, # W/(m2·K) moving air (mean of 6.7m/s and 3.4m/s values)
    "U_WALLS": 0.2371,  # [W/(m2·K)] ∘ 1 / U_walls = (1 / h_wall_in) + (1 / rock_wool_u) + (1 / h_wall_out)
    "U_ROOF": 0.2374,  # [W/(m2·K)] ∘ 1 / U_roof = (1 / h_roof_in) + (1 / rock_wool_u) + (1 / h_roof_out)


    "K_PEX": 0.41, # PEX thermal conductivity [W/(m·K)]
    "TUBE_INNER_DIAMETER": 0.022225,  # tube inner diameter [m], for 1" pex tube
    "TUBE_OUTER_DIAMETER": 0.028575,  # tube outer dimater [m], for 1" pex tube
    "TUBE_THICKNESS": 0.00635,  # tube thickness [m], for 1" pex tube
    "TUBES_LENGTH": 220,  # [m] with spacing of 0.5m between tubes
    "A_TUBES": 220 * 3.1415 * 0.028575,  # tubes surface area [m2], length * pi * outer_diameter

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

Y0_5eqs = {
    "t_cond": 298.73661156,
    "t_tank": 297.58993216,
    "t_out_heating": 295.85297099,
    "t_floor": 295.4605917,
    "t_room": 293.62059239,
    "p_bat": 1e-2,
}

Y0 = {
    "t_cond": 296.56909163,
    "t_tank": 296.05234612,
    "t_floor": 295.27582739,
    "t_room": 293.47756316,
}
