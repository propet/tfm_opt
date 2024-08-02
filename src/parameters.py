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
    # "HORIZON": 100000,  # s
    "HORIZON": 30 * 24 * 3600,  # s
    "N_HOURS": 1000,  # 8760 in a year
    "MAX_BAT_CAPACITY": 13000,  # Wh
    "SOLAR_SIZE": 3000,  # installed power in solar panels W
    "E_BAT_MAX": 13000 * 3600,  # 13000 Wh -> (13000*3600) Ws
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
    "TANK_VOLUME": 1,  # m3
    "MAX_TANK_VOLUME": 1000, # m3
    "MIN_TANK_VOLUME": 1, # m3
    # TODO: make insulation length a design variable
    "U": 0.2,  # Overall heat transfer coefficient of rockwool insulation (W/(m2·K))
    "T_AMB": 300,  # K (27ºC)
    "P_COMPRESSOR_MAX": 3000,  # W

    # "M_DOT_COND_MAX": 1, # kg/s
    # "M_DOT_LOAD_MAX": 1, # kg/s
    "M_DOT_COND_MAX": 0.3, # kg/s
    "M_DOT_LOAD_MAX": 0.3, # kg/s

    "MAX_Q_DOT_REQUIRED": 8000,  # W
    "T_TARGET": 298,  # K
    "ROCK_WOOL_U": 0.2,  # W/(m2K)
    "ROCK_WOOL_AREA": 175,  # m2
    "WINDOWS_U": 1.3,  # W/(m2K)
    "WINDOWS_AREA": 25,  # m2
    "LOAD_HX_EFF": 0.8,


}
