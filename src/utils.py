import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_types import PlotData
import scienceplots


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, "..")


# Scienceplots style
plt.style.use(["science", "ieee"])
plt.rcParams.update({"figure.dpi": "300"})


def generic_plot(plot_data: PlotData, title=None, sharex=False, sharey=False):
    fig, axes = plt.subplots(
        nrows=plot_data["rows"],
        ncols=plot_data["columns"],
        figsize=(10, 10),
        sharex=sharex,
        sharey=sharey,
    )

    for axe_data in plot_data["axes_data"]:
        axe = None
        if plot_data["rows"] == 1 and plot_data["columns"] == 1:
            axe = axes
        elif plot_data["rows"] > 1 and plot_data["columns"] > 1:
            axe = axes["i"]["j"]
        else:
            if axe_data["i"] > axe_data["j"]:
                axe = axes[axe_data["i"]]
            else:
                axe = axes[axe_data["j"]]

        for array_data in axe_data["arrays_data"]:
            axe.plot(
                array_data["x"],
                array_data["y"],
                label=array_data["label"],
            )

        if "title" in axe_data:
            axe.title.set_text(axe_data["title"])
        if "xlabel" in axe_data:
            axe.set_xlabel(axe_data["xlabel"])
        if "ylabel" in axe_data:
            axe.set_ylabel(axe_data["ylabel"])
        axe.grid(True)
        axe.legend()

    if title:
        fig.suptitle(title, fontsize=16)
    plt.show()


def get_solar_field_powers(max_solar_radiation, n_hours):
    HOURS_IN_DAY = 24
    min_solar_radiation = 0
    amplitude = (max_solar_radiation - min_solar_radiation) / 2
    vertical_shift = amplitude + min_solar_radiation
    # Generate an array of hours for one year
    hours = np.arange(n_hours)
    # Calculate the sinusoidal function value for each hour
    # Max radiation at noon and minimum at midnight (sinusoidal shifted 6 hours to the right)
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
    # Max radiation at 06:00 and minimum at 18:00
    # A * sin(w * t) = A * sin(2*pi*f*t)
    p_electric_demand = amplitude * np.sin(2 * np.pi * (1 / HOURS_IN_DAY) * hours) + vertical_shift
    return p_electric_demand


def get_grid_prices_kwh(n_hours, year=2022):
    directory = f"{ROOT_DIR}/data/mercado_diario_precio_horario_{year}"
    grid_prices_mwh = []

    # Loop over every csv file
    for file in sorted(os.listdir(directory)):
        if file.startswith("marginalpdbc_") and file.endswith(".1"):
            filepath = os.path.join(
                directory,
                file,
            )
            # Read csv with separator ";"
            df = pd.read_csv(
                filepath,
                sep=";",
                header=None,
                skiprows=1,
                skipfooter=1,
                engine="python",
            )
            price_values = df.iloc[:, 5]  # Select last column
            # Append to array
            grid_prices_mwh.extend(price_values.tolist())

    grid_prices_mwh = grid_prices_mwh[:n_hours]
    grid_prices_kwh = np.array(grid_prices_mwh) * 1e-3
    return grid_prices_kwh
