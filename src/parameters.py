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
    "N_HOURS": 8760,  # 8760 in a year
    "MAX_BAT_CAPACITY": 120000,  # Wh
    "SOC_MIN": 0.1,  # Minimum State of Charge
    "SOC_MAX": 0.9,  # Maximum State of Charge
    "P_BAT_MAX": 10000,  # W
    "P_GRID_MAX": 10000,  # W
    "MAX_SOLAR_RADIATION": 10000,  # W
    "MAX_ELECTRIC_DEMAND": 20000,  # W
    "DK_RHO": 50,  # Kreisselmeier–Steinhauser factor
    "BAT_ETA_CHARGE": 0.8,  # charge efficiency
    "BAT_ETA_DISCHARGE": 0.9,  # discharge efficiency
}
