import os
import pickle
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_types import PlotData
import scienceplots
from parameters import PARAMS
from cycler import cycler
import jax
import jax.numpy as jnp


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, "..")


# Scienceplots style
plt.style.use(["science", "ieee"])
plt.rcParams.update({"figure.dpi": "300"})
prop_cycle = cycler('color', ['k', 'r', 'b', 'g', 'purple']) + cycler('linestyle', ['-', '--', ':', '-.', (5, (10, 3))])
plot_styles = list(prop_cycle)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generic_plot(plot_data: PlotData, filename=None, title=None, sharex=False, sharey=False):
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

    # Save the figure
    directory = f"{ROOT_DIR}/figures"
    if not filename:
        filename = "_".join(f"{key}_{value}" for key, value in PARAMS.items()) + ".png"
    filepath = os.path.join(
        directory,
        filename,
    )
    fig.savefig(filepath)
    print(f"Figure saved as {filepath}")


def plot_film():
    directory = f"tmp"
    image_files = []
    for file in sorted(os.listdir(directory)):
        image_files.append(file)

    with imageio.get_writer("animation.gif", mode="I", duration=0.5, loop=0) as writer:  # Adjust duration as needed
        for filename in image_files:
            image = imageio.imread(f"{directory}/{filename}")
            writer.append_data(image)

    # Delete the files after creating the GIF
    for filename in image_files:
        os.remove(os.path.join(directory, filename))


def save_dict_to_file(dict):
    filename = "saves/dict_file.pkl"
    # Open a file in write-binary mode and use pickle to serialize the dictionary
    with open(filename, 'wb') as f:
        pickle.dump(dict, f)

    print(f"Saved to {filename}")


def load_dict_from_file(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f)
    return dict


def get_dynamic_parameters(t0, h, horizon):
    dynamic_parameters = {}
    dynamic_parameters["t_amb"] = get_t_amb(t0, h, horizon)
    dynamic_parameters["cost_grid"] = get_cost_grid(t0, h, horizon)
    dynamic_parameters["p_solar_gen"] = get_p_solar_gen(t0, h, horizon)
    dynamic_parameters["q_dot_required"] = get_q_dot_required(t0, h, horizon)
    dynamic_parameters["p_required"] = get_p_required(t0, h, horizon)
    return dynamic_parameters


# def get_q_dot_required(t0, h, horizon):
#     q_dot_required = np.ones((horizon)) * PARAMS["P_COMPRESSOR_MAX"] * 1.3
#     return q_dot_required[t0:t0+horizon:h]


def get_q_dot_required(t0, h, horizon):
    max_q_dot_required = PARAMS["MAX_Q_DOT_REQUIRED"]
    seconds_in_day = 24 * 3600
    min_q_dot_required = 0
    amplitude = (max_q_dot_required - min_q_dot_required) / 2
    vertical_shift = amplitude + min_q_dot_required
    seconds = np.arange(horizon)
    # A * sin(w * t) = A * sin(2*pi*f*t)
    q_dot_required = amplitude * np.sin(2 * np.pi * (1 / seconds_in_day) * (seconds - 6 * 3600)) + vertical_shift
    return q_dot_required[t0:t0+horizon:h]


def get_p_required(t0, h, horizon):
    max_p_required = PARAMS["MAX_Q_DOT_REQUIRED"]
    seconds_in_day = 24 * 3600
    min_p_required = 0
    amplitude = (max_p_required - min_p_required) / 2
    vertical_shift = amplitude + min_p_required
    seconds = np.arange(horizon)
    # A * sin(w * t) = A * sin(2*pi*f*t)
    p_required = amplitude * np.sin(2 * np.pi * (1 / seconds_in_day) * (seconds - 6 * 3600)) + vertical_shift
    return p_required[t0:t0+horizon:h]


def get_p_solar_gen(t0, h, horizon):
    max_solar_radiation = PARAMS["MAX_SOLAR_RADIATION"]
    seconds_in_day = 24 * 3600
    min_solar_radiation = 0
    amplitude = (max_solar_radiation - min_solar_radiation) / 2
    vertical_shift = amplitude + min_solar_radiation
    seconds = np.arange(horizon)
    # A * sin(w * t) = A * sin(2*pi*f*t)
    p_gen = amplitude * np.sin(2 * np.pi * (1 / seconds_in_day) * (seconds - 6 * 3600)) + vertical_shift
    return p_gen[t0:t0+horizon:h]


def get_t_amb(t0, h, horizon):
    t_amb = np.ones((horizon)) * 300  # 300 K == 27ºC
    return t_amb[t0:t0+horizon:h]


def get_cost_grid(t0, h, horizon, year=2022):
    n_hours = 8760
    cost_grid_by_hour = get_grid_prices_kwh(n_hours, year=year)
    cost_grid_by_second = np.repeat(cost_grid_by_hour, 3600)
    return cost_grid_by_second[t0:t0+horizon:h]


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


def ks_max(x: np.ndarray, rho: float = 100):
    """
    Calculates the KS function for given rho and x values.

    Parameters:
    rho (float): The rho parameter.
    x (np.ndarray): An array of x values.

    Returns:
    float: The result of the KS function.
    """
    max_x = np.max(x)
    sum_exp = np.sum(np.exp(np.dot(rho, (x - max_x))))
    return max_x + (1 / rho) * np.log(sum_exp)


# @jax.jit
# def ks_max_jax(x: jnp.ndarray, rho: float = 100.0):
#     """
#     Calculates the KS function for given rho and x values using JAX.
#
#     Parameters:
#     x (jnp.ndarray): An array of x values.
#     rho (float): The rho parameter.
#
#     Returns:
#     float: The result of the KS function.
#     """
#     max_x = jnp.max(x)
#     sum_exp = jnp.sum(jnp.exp(rho * (x - max_x)))
#     return max_x + (1 / rho) * jnp.log(sum_exp)
#
#
# def ks_max(x):
#     if isinstance(x, jnp.ndarray):
#         return ks_max_jax(x)
#     else:
#         return ks_max_np(x)


def ks_min(x: np.ndarray, rho: float = 100) -> float:
    """
    Calculates the KS function for given rho and x values.

    Parameters:
    rho (float): The rho parameter.
    x (np.ndarray): An array of x values.

    Returns:
    float: The result of the KS function.
    """
    min_x = np.min(x)
    sum_exp = np.sum(np.exp(-rho * (x - min_x)))
    return min_x - (1 / rho) * np.log(sum_exp)


def jax_to_numpy(jax_func):
    def numpy_func(*args):
        jax_args = [jnp.array(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
        result = jax_func(*jax_args)
        return np.array(result)
    return numpy_func


if __name__ == "__main__":
    t0 = 0
    horizon = 3600 * 24
    h = 3600
    t_amb = get_t_amb(t0, h, horizon)
    print(t_amb)
    print(t_amb.shape)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(t0, t0+horizon, h), t_amb, 'b-', linewidth=2)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel(r'solar_field', fontsize=14)
    plt.grid(True)
    plt.show()

