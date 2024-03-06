import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, "..")


def get_solar_field_powers(max_solar_radiation, n_hours):
    HOURS_IN_DAY = 24
    min_solar_radiation = 0
    amplitude = (max_solar_radiation - min_solar_radiation) / 2
    vertical_shift = amplitude + min_solar_radiation
    # Generate an array of hours for one year
    hours = np.arange(n_hours)
    # Calculate the sinusoidal function value for each hour
    # máximo de radiación a mediodía y mínimo a medianoche (sinusoidal desplazada 6 horas a la derecha)
    # A * sin(w * t) = A * sin(2*pi*f*t)
    p_gen = amplitude * np.sin(2 * np.pi * (1 / HOURS_IN_DAY) * (hours - 6)) + vertical_shift
    return p_gen


def get_electric_demand_powers(max_electric_demand, n_hours):
    HOURS_IN_DAY = 24
    min_electric_demand = 0
    amplitude = (max_electric_demand - min_electric_demand) / 2
    vertical_shift = amplitude + min_electric_demand
    # Generate an array of hours for one year
    hours = np.arange(n_hours)
    # Calculate the sinusoidal function value for each hour
    # máximo de demanda a las 6 de la mañana y mínimo a las 6 de la tarde
    # A * sin(w * t) = A * sin(2*pi*f*t)
    p_electric_demand = amplitude * np.sin(2 * np.pi * (1 / HOURS_IN_DAY) * hours) + vertical_shift
    return p_electric_demand


def get_grid_prices_mwh(n_hours, year=2022):
    directory = f"{ROOT_DIR}/data/mercado_diario_precio_horario_{year}"
    grid_prices_mwh = []

    # Loop over every csv file
    for file in sorted(os.listdir(directory)):
        if file.startswith("marginalpdbc_") and file.endswith(".1"):
            filepath = os.path.join(directory, file)
            # Read csv with separator ";"
            df = pd.read_csv(filepath, sep=";", header=None, skiprows=1, skipfooter=1, engine="python")
            price_values = df.iloc[:, 5]  # Select last column
            # Append to array
            grid_prices_mwh.extend(price_values.tolist())

    grid_prices_mwh = grid_prices_mwh[:n_hours]
    return grid_prices_mwh
