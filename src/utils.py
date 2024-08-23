import os
import json
import pickle
import csv
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from custom_types import PlotData
import scienceplots
from parameters import PARAMS
from cycler import cycler
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, "..")


# Scienceplots style
plt.style.use(["science", "ieee"])
plt.rcParams.update({"figure.dpi": "300"})
prop_cycle = cycler("color", ["k", "r", "b", "g", "purple", "salmon"]) + cycler(
    "linestyle", ["-", "--", ":", "-.", (5, (10, 3)), ":"]
)
plot_styles = list(prop_cycle)

# Set custom linewidth for all plot styles
custom_linewidth = 0.5
for style in plot_styles:
    style["linewidth"] = custom_linewidth


def get_tank_data():
    filepath = f"{ROOT_DIR}/data/equipment_costs/water_tanks.csv"
    df = pd.read_csv(
        filepath,
        sep=",",
        header=None,
        skiprows=1,
        skipfooter=0,
        engine="python",
    )
    volume = df.iloc[:, 4].to_numpy()
    cost = df.iloc[:, 5].to_numpy()
    return volume, cost


def get_batteries_data():
    filepath = f"{ROOT_DIR}/data/equipment_costs/lifepo4_batteries.csv"
    df = pd.read_csv(
        filepath,
        sep=",",
        header=None,
        skiprows=1,
        skipfooter=0,
        engine="python",
    )
    watts_second = df.iloc[:, 5].to_numpy()
    cost = df.iloc[:, 10].to_numpy()
    return watts_second, cost


def get_hp_data():
    filepath = f"{ROOT_DIR}/data/equipment_costs/heat_pumps.csv"
    df = pd.read_csv(
        filepath,
        sep=",",
        header=None,
        skiprows=1,
        skipfooter=0,
        engine="python",
    )
    heat = df.iloc[:, 3].to_numpy()
    power = df.iloc[:, 5].to_numpy()
    cost = df.iloc[:, 6].to_numpy()
    return power, cost, heat


def get_generator_data():
    filepath = f"{ROOT_DIR}/data/equipment_costs/gensets.csv"
    df = pd.read_csv(
        filepath,
        sep=",",
        header=None,
        skiprows=1,
        skipfooter=0,
        engine="python",
    )
    power = df.iloc[:, 3].to_numpy()
    cost = df.iloc[:, 4].to_numpy()
    return power, cost


def get_hp_depreciation_by_joule(hp_power):
    """
    hp_power [W]: maximum power for compressor, not heat capacity
    """
    years_lifespan = 10
    seconds_in_year = 365 * 24 * 3600
    seconds_in_lifespan = seconds_in_year * years_lifespan
    cost0 = 871.9339209961945  # from linear regression
    slope = 1.783401453095622  # [$/W] from linear regression
    cost = cost0 + slope * hp_power
    depreciation_by_joule = cost / (hp_power * seconds_in_lifespan)
    return depreciation_by_joule


def get_tank_depreciation_by_second(tank_volume):
    """
    tank_volume [m3]
    """
    seconds_in_year = 365 * 24 * 3600
    years_lifespan = 20
    cost0 = 663.2745177542383  # from linear regression
    slope = 5235.333812731176  # [$/m3] from linear regression
    cost = cost0 + slope * tank_volume
    depreciation_by_second = cost / (seconds_in_year * years_lifespan)
    return depreciation_by_second


def get_battery_depreciation_by_joule(e_bat_max):
    """
    e_bat_max [Ws]: max capacity of the battery
    """
    cost0 = 11.873235223976735  # from linear regression
    slope = 0.00016994110156603917  # [$/Ws] -> 600[$/kWh] from linear regression
    cost = cost0 + slope * e_bat_max

    # https://cdn.autosolar.es/pdf/fichas-tecnicas/Bat-LFP-12,8-25,6V-Smart-ES.pdf
    n_cycles = 3000
    dod = 0.7

    # Depreciate by how much energy has been transferred
    total_lifespan_energy_transferred = n_cycles * 2 * dod * e_bat_max  # times 2 because in a cycle we charge and discharge
    depreciation_by_joule = cost / total_lifespan_energy_transferred
    return depreciation_by_joule


def get_battery_depreciation_by_second(e_bat_max):
    """
    e_bat_max [Ws]: max capacity of the battery
    """
    seconds_in_year = 365 * 24 * 3600
    years_lifespan = 20
    seconds_in_lifespan = seconds_in_year * years_lifespan
    cost0 = 11.873235223976735  # from linear regression
    slope = 0.00016994110156603917  # [$/Ws] -> 600[$/kWh] from linear regression
    cost = cost0 + slope * e_bat_max
    depreciation_by_second = cost / seconds_in_lifespan
    return depreciation_by_second


def get_solar_panels_depreciation_by_second(solar_installed_power):
    """
    solar_installed_power [W]
    """
    seconds_in_year = 365 * 24 * 3600
    years_lifespan = 20
    price_per_w = 1.161
    cost = price_per_w * solar_installed_power
    depreciation_by_second = cost / (seconds_in_year * years_lifespan)
    return depreciation_by_second


def get_generator_depreciation_by_joule(generator_power):
    """
    AC generator_power [W]
    """
    years_lifespan = 10
    seconds_in_year = 365 * 24 * 3600
    seconds_in_lifespan = seconds_in_year * years_lifespan
    cost0 = 2637.2943203079794  # from linear regression
    slope = 0.18673399507937297  # [$/m3] from linear regression
    cost = cost0 + slope * generator_power
    depreciation_by_joule = cost / (generator_power * seconds_in_lifespan)
    return depreciation_by_joule


def get_fixed_energy_cost_by_second(p_grid_max):
    # 39.1 $/(kW*year)
    fixed_cost_by_second = (p_grid_max * 39.1) / (1000 * 8760 * 3600)
    return fixed_cost_by_second


def linear_regression(x, y):
    x = x.reshape(-1, 1)

    # Create and train the model
    model = LinearRegression()
    model.fit(x, y)

    # y = y0 + m * x
    m = model.coef_[0]
    y0 = model.intercept_

    return y0, m


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cop(T):
    if isinstance(T, jnp.ndarray):
        return cop_jax(T)
    else:
        return cop_np(T)


def cop_jax(t):
    """
    t: temperature [Kelvin]

    piecewise lineal -> non-linear function overall

    In the following heat pump technicals
    https://www.daikin.es/content/dam/DACS/document-library/Pdfs-subidos-2022/Monobloc%20EBLA%209-11%20-%2014-16.pdf

    it mentions reference SCOP of 4.8 at 35C (308K) (output water temperature),
    and SCOP 3.4 at 55C (328K)

    So we set the COP to 4.8 up to 35C, and linearly decreasing to 3.4 at 55C
    minimum COP value is 0
    """
    # cop = cop_0 + m * t
    # cop_1 = cop_0 + m * t_1 -> cop_0 = cop_1 - m * t_1
    # cop_3 = 0 = cop_0 + m * t_3 -> t_3 = - cop_0 / m
    cop_1 = 4.8
    cop_2 = 3.4
    t_1 = 308
    t_2 = 328
    cop_3 = 0
    m = (cop_1 - cop_2) / (t_1 - t_2)
    cop_0 = cop_1 - m * t_1
    t_3 = - cop_0 / m

    conditions = [t < t_1, (t >= t_1) & (t < t_3), t >= t_3]
    choices = [cop_1, cop_0 + m * t, cop_3]
    return jnp.select(conditions, choices)


def cop_np(t):
    # cop = cop_0 + m * t
    # cop_1 = cop_0 + m * t_1 -> cop_0 = cop_1 - m * t_1
    # cop_3 = 0 = cop_0 + m * t_3 -> t_3 = - cop_0 / m
    cop_1 = 4.8
    cop_2 = 3.4
    t_1 = 308
    t_2 = 328
    cop_3 = 0
    m = (cop_1 - cop_2) / (t_1 - t_2)
    cop_0 = cop_1 - m * t_1
    t_3 = - cop_0 / m

    conditions = [t < t_1, (t >= t_1) & (t < t_3), t >= t_3]
    functions = [lambda t: cop_1, lambda t: cop_0 + m * t, lambda t: cop_3]
    return np.piecewise(t, conditions, functions)


def get_dcopdT(t):
    # cop = cop_0 + m * t
    # cop_1 = cop_0 + m * t_1 -> cop_0 = cop_1 - m * t_1
    # cop_3 = 0 = cop_0 + m * t_3 -> t_3 = - cop_0 / m
    cop_1 = 4.8
    cop_2 = 3.4
    t_1 = 308
    t_2 = 328
    cop_3 = 0
    m = (cop_1 - cop_2) / (t_1 - t_2)
    cop_0 = cop_1 - m * t_1
    t_3 = - cop_0 / m

    conditions = [t < t_1, (t >= t_1) & (t < t_3), t >= t_3]
    functions = [lambda t: 0, lambda t: m, lambda t: 0]
    return np.piecewise(t, conditions, functions)


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


def plot_film(filename):
    print("Merging pictures into gif file")
    directory = f"tmp"
    image_files = []
    for file in sorted(os.listdir(directory)):
        image_files.append(file)
    image_files.remove(".gitignore")  # don't process .gitignore

    with imageio.get_writer(filename, mode="I", duration=500, loop=0) as writer:  # Adjust duration as needed
        for filename in image_files:
            image = imageio.imread(f"{directory}/{filename}")
            writer.append_data(image)

    # Delete the files after creating the GIF
    for filename in image_files:
        os.remove(os.path.join(directory, filename))


def save_dict_to_file(dict):
    filename = "saves/dict_file.pkl"
    # Open a file in write-binary mode and use pickle to serialize the dictionary
    with open(filename, "wb") as f:
        pickle.dump(dict, f)

    print(f"Saved to {filename}")


def load_dict_from_file(filename):
    with open(filename, "rb") as f:
        dict = pickle.load(f)
    return dict


def get_dynamic_parameters(t0, h, horizon, year=2022):
    dynamic_parameters = {}
    t_amb_every_second = get_t_amb_every_second(year=2022)
    dynamic_parameters["t_amb"] = get_t_amb(t_amb_every_second, t0, h, horizon)
    dynamic_parameters["w_solar_per_w_installed"] = get_w_solar_per_w_installed(t0, h, horizon, year=year)
    dynamic_parameters["q_dot_required"] = get_q_dot_required(t_amb_every_second, t0, h, horizon)
    dynamic_parameters["p_required"] = get_p_required(t0, h, horizon)
    dynamic_parameters["daily_prices"] = get_daily_prices(t0, h, horizon, year=year)
    dynamic_parameters["pvpc_prices"] = get_pvpc_prices(t0, h, horizon, year=year)
    dynamic_parameters["excess_prices"] = get_excess_prices(t0, h, horizon, year=year)
    return dynamic_parameters


# def get_q_dot_required(t0, h, horizon):
#     """
#     Constant q_dot_required
#     """
#     q_dot_required = np.ones((horizon)) * PARAMS["P_COMPRESSOR_MAX"] * 1.3
#     return q_dot_required[t0:t0+horizon:h]


def get_q_dot_required(t_amb, t0, h, horizon):
    """
    Based on ambient temperature and conduction loses.

    In my case I've assumed that in my house I have a target temperature for all the year of 293K (20C).
    And the heat consumption comes from heat loses due to conduction.
    So the heat consumption is proportional to the difference between my target temperature
    and the ambient temperature.

    Fourier law of heat conduction

    Q[W] = U[W/(m2*K)] A[m2] (T_target - T_amb)[K] = K[W/K] (T_target - T_amb)

    As an example, we can take a house of 100m2 of floor area, same as roof area.
    With walls 2.5m tall. Which would correspond to 100m2 of sides area.

    Taking 3/4 of the side area as wall, and the other 1/3 as windows.
    Consider the walls and roof have 20cm of rock wool insulator.
    """
    t_target = PARAMS["T_TARGET"]
    rock_wool_u = PARAMS["ROCK_WOOL_U"]
    rock_wool_area = PARAMS["ROCK_WOOL_AREA"]
    windows_u = PARAMS["WINDOWS_U"]
    windows_area = PARAMS["WINDOWS_AREA"]

    print("t_amb shape: ", t_amb.shape)

    q_dot_required = (rock_wool_u * rock_wool_area + windows_u * windows_area) * (t_target - t_amb)
    q_dot_required[q_dot_required < 0] = 0
    print("q_dot shape: ", q_dot_required.shape)
    print("q_dot_mean: ", np.mean(q_dot_required))
    print("q_dot_sum: ", np.sum(q_dot_required))
    print("deltaT_mean: ", np.mean(t_target - t_amb))
    return q_dot_required[t0 : t0 + horizon : h]


# def get_t_amb(t0, h, horizon):
#     t_amb = np.ones((horizon)) * 300  # 300 K == 27ºC
#     return t_amb[t0:t0+horizon:h]


def get_t_amb(t_amb_every_second, t0, h, horizon):
    return t_amb_every_second[t0 : t0 + horizon : h]


def get_t_amb_every_second(year=2022):
    filepath = f"{ROOT_DIR}/data/meteosat_madrid_every_15min/248481_40.41_-3.70_{year}.csv"

    df = pd.read_csv(
        filepath,
        sep=",",
        header=None,
        skiprows=3,
        skipfooter=0,
        engine="python",
    )

    t_amb_every_15min = df.iloc[:, 19].to_numpy()  # T column
    t_amb_every_15min += 273  # from Celsius to Kelvin

    t_amb_every_second = np.repeat(t_amb_every_15min, 900)  # 15min == 900s
    return t_amb_every_second


def get_daily_prices(t0, h, horizon, year=2022):
    n_hours = 8760
    daily_prices_by_hour = get_grid_prices_kwh(n_hours, year=year)
    daily_prices_by_second = np.repeat(daily_prices_by_hour, 3600)  # 1hour == 3600s
    daily_prices_by_second = daily_prices_by_second / (1000 * 3600)  # $/(kWh) to $/(Ws)
    return daily_prices_by_second[t0 : t0 + horizon : h]


def get_w_solar_per_w_installed(t0, h, horizon, year=2022):
    # Data obtained from SAM
    filepath = f"{ROOT_DIR}/data/sam_solar_power_madrid_every_15min/{year}_1kW.csv"

    df = pd.read_csv(
        filepath,
        sep=",",
        header=None,
        skiprows=1,
        skipfooter=0,
        engine="python",
    )

    w_solar_per_w_installed_every_15min = df.iloc[:, 1].to_numpy()  # second column
    w_solar_per_w_installed_every_second = np.repeat(w_solar_per_w_installed_every_15min, 900)  # 15min == 900s
    return w_solar_per_w_installed_every_second[t0 : t0 + horizon : h]


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


def get_p_required(t0, h, horizon):
    """
    My own electricity demand for a year
    Obtained through repsol clients webpage
    kWh for each hour of the year
    same as the power in kW for the whole hour
    """
    json_filename = f"{ROOT_DIR}/data/repsol_kwh_demand.json"
    json_file = open(json_filename)
    info = json.load(json_file)

    # Initialize an empty list to store the consumption values
    p_required_by_hour = []

    # Iterate through each day in the JSON data
    print("n_days: ", len(info))
    for day in info:
        # Iterate through each hour
        for hour_entry in day["data"]:
            # Append the consumption value to the list
            p_required_by_hour.append(hour_entry["consumption"])
            # print(len(day["data"]))
        if len(day["data"]) != 24:
            print([day["date"]])
        # print(day["date"])

    print("hours: ", len(p_required_by_hour))

    p_required_by_second = np.repeat(p_required_by_hour, 3600)  # 1hour == 3600s
    p_required_by_second = p_required_by_second * 1000  # kW to W
    return p_required_by_second[t0 : t0 + horizon : h]


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
    directory = f"{ROOT_DIR}/data/omie_daily_prices_hourly/mercado_diario_precio_horario_{year}"
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


def get_pvpc_prices(t0, h, horizon, year=2022):
    filepath = f"{ROOT_DIR}/data/pvpc_prices_hourly/pvpc_{year}.csv"
    df = pd.read_csv(
        filepath,
        sep=";",
        header=None,
        skiprows=1,
        skipfooter=0,
        engine="python",
    )
    pvpc_MWh_hourly = df.iloc[:, 4].to_numpy()
    pvpc_MWh_by_second = np.repeat(pvpc_MWh_hourly, 3600)  # 1hour == 3600s
    pvpc_Ws_by_second = pvpc_MWh_by_second / (1e6 * 3600)  # $/(MWh) to $/(Ws)
    return pvpc_Ws_by_second[t0 : t0 + horizon : h]


def get_excess_prices(t0, h, horizon, year=2022):
    filepath = f"{ROOT_DIR}/data/excess_power_prices_hourly/precio_compensacion_excedentes_{year}.csv"
    df = pd.read_csv(
        filepath,
        sep=";",
        header=None,
        skiprows=1,
        skipfooter=0,
        engine="python",
    )
    excess_price_MWh_hourly = df.iloc[:, 4].to_numpy()
    excess_price_MWh_by_second = np.repeat(excess_price_MWh_hourly, 3600)  # 1hour == 3600s
    excess_price_Ws_by_second = excess_price_MWh_by_second / (1e6 * 3600)  # $/(MWh) to $/(Ws)
    return excess_price_Ws_by_second[t0 : t0 + horizon : h]


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


def sparse_to_required_format(mat):
    return {"csr": [mat.indptr, mat.indices, mat.data], "shape": list(mat.shape)}

def plot_prices():
    t0 = 0
    # t0 = 24 * 3600 * 90
    h = 100
    horizon = 3600 * 24 * 365
    # horizon = 3600 * 24 * 30

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon, year=2022)
    parameters = PARAMS
    daily_prices = dynamic_parameters["daily_prices"]
    pvpc_prices = dynamic_parameters["pvpc_prices"]
    excess_prices = dynamic_parameters["excess_prices"]
    daily_prices = daily_prices * 1e3 * 3600  # Ws to kWh
    pvpc_prices = pvpc_prices * 1e3 * 3600  # Ws to kWh
    excess_prices = excess_prices * 1e3 * 3600  # Ws to kWh

    time = np.arange(t0, t0 + horizon, h)

    # plt.plot(time / 3600 / 24, daily_prices, label="diario", **plot_styles[0])
    plt.plot(time / 3600 / 24, pvpc_prices, label="pvpc", **plot_styles[0], linewidth=0.5)
    plt.plot(time / 3600 / 24, excess_prices, label="compensación", **plot_styles[1], linewidth=0.5)

    plt.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    plt.xlabel("días")
    plt.ylabel(r"€$/kWh$")
    plt.grid(True)
    plt.show()

def plot_dynamic_parameters():
    t0 = 0
    # t0 = 24 * 3600 * 90
    h = 100
    horizon = 3600 * 24 * 365
    # horizon = 3600 * 24 * 30

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["daily_prices"] = dynamic_parameters["daily_prices"]
    parameters["pvpc_prices"] = dynamic_parameters["pvpc_prices"]
    parameters["excess_prices"] = dynamic_parameters["excess_prices"]

    time = np.arange(t0, t0 + horizon, h)
    print("time.shape: ", time.shape)
    print("p_required shape: ", parameters["p_required"].shape)

    # fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    # # axes[0].plot(time / 3600, parameters["p_required"], label="p_required", **plot_styles[0])
    # axes[0].plot(time / 3600, parameters["t_amb"], label="t_amb", **plot_styles[0])
    # axes[0].legend()
    #
    # # axes[0].plot(time / 3600, parameters["daily_prices"], label="daily_prices", **plot_styles[0])
    # # axes[0].legend()
    #
    # axes[1].plot(time / 3600, parameters["w_solar_per_w_installed"], label="w_solar_per_w_installed", **plot_styles[1])
    # # axes[1].plot(time / 3600, parameters["daily_prices"], label="daily_prices", **plot_styles[0])
    # # axes[1].legend()
    # # axes[1].set_xlabel("hour", fontsize=14)
    #
    # plt.grid(True)
    # plt.show()

    parameters["p_required"] = parameters["p_required"] / 1000 # W to kW
    parameters["t_amb"] = parameters["t_amb"] - 273
    # plt.plot(time / 3600 / 24, parameters["p_required"], **plot_styles[0], linewidth=0.5)
    # plt.plot(time / 3600 / 24, parameters["w_solar_per_w_installed"], **plot_styles[0], linewidth=0.5)
    plt.plot(time / 3600 / 24, parameters["t_amb"], **plot_styles[0], linewidth=0.5)
    # plt.ylabel("kW")
    plt.ylabel("Temperatura [ºC]")
    plt.xlabel("días")
    plt.grid(True)
    plt.show()


def plot_data_regressions():
    tank_volume_data, tank_cost_data = get_tank_data()
    battery_watts_second_data, battery_cost_data = get_batteries_data()
    hp_power_w_data, hp_cost_data, hp_heat_w_data = get_hp_data()
    generator_power_w_data, generator_cost_data = get_generator_data()

    # # Water tank
    # tank_cost0, tank_slope = linear_regression(tank_volume_data, tank_cost_data)
    # print(f"tank y0: {tank_cost0}, slope: {tank_slope}")
    # tank_volume = np.linspace(0, 4, 1000)
    # tank_cost = tank_cost0 + tank_slope * tank_volume
    # plt.scatter(tank_volume_data, tank_cost_data, **plot_styles[0])
    # plt.plot(tank_volume, tank_cost, **plot_styles[1])
    # # plt.title("Water tank cost")
    # plt.ylabel("€")
    # plt.xlabel(r"$m^3$")
    # plt.show()

    # # Water tank scaled from m3 to liters
    # tank_volume_data  = tank_volume_data * 1000
    # tank_volume = np.linspace(0, 4000, 1000)
    # tank_cost0, tank_slope = linear_regression(tank_volume_data, tank_cost_data)
    # tank_cost = tank_cost0 + tank_slope * tank_volume
    # plt.scatter(tank_volume_data, tank_cost_data, **plot_styles[0])
    # plt.plot(tank_volume, tank_cost, **plot_styles[1])
    # plt.title("Water tank cost")
    # plt.ylabel("€")
    # plt.xlabel(r"$l$")
    # plt.show()

    # # Batteries
    # battery_cost0, battery_slope = linear_regression(battery_watts_second_data, battery_cost_data)
    # print(f"battery y0: {battery_cost0}, slope: {battery_slope}")
    # battery_watts_second = np.linspace(0, 90000000)
    # battery_cost = battery_cost0 + battery_slope * battery_watts_second
    # plt.scatter(battery_watts_second_data, battery_cost_data, **plot_styles[0])
    # plt.plot(battery_watts_second, battery_cost, **plot_styles[1])
    # plt.title("Battery cost")
    # plt.xlabel(r"$W \cdot s$")
    # plt.ylabel("€")
    # plt.show()

    # # Batteries scaled from Ws to Wh
    # battery_kwatts_hour_data  = battery_watts_second_data / 3600 / 1000
    # battery_cost0, battery_slope = linear_regression(battery_kwatts_hour_data, battery_cost_data)
    # battery_kwatts_hour = np.linspace(0, 90000000 / 3600 / 1000, 1000)
    # battery_cost = battery_cost0 + battery_slope * battery_kwatts_hour
    # plt.scatter(battery_kwatts_hour_data, battery_cost_data, **plot_styles[0])
    # plt.plot(battery_kwatts_hour, battery_cost, **plot_styles[1])
    # # plt.title("Battery cost")
    # plt.xlabel(r"$kW \cdot h$")
    # plt.ylabel("€")
    # plt.show()

    # # Heat pumps
    # hp_cost0, hp_slope = linear_regression(hp_power_w_data, hp_cost_data)
    # print(f"hp y0: {hp_cost0}, slope: {hp_slope}")
    # hp_power_w = np.linspace(1000, 5100, 1000)
    # hp_cost = hp_cost0 + hp_slope * hp_power_w
    # plt.scatter(hp_power_w_data, hp_cost_data, **plot_styles[0])
    # plt.plot(hp_power_w, hp_cost, **plot_styles[1])
    # # plt.title("Heat pump cost")
    # plt.xlabel(r"$W$")
    # plt.ylabel("€")
    # plt.show()

    # # Heat pumps scaled from W to kW
    # hp_power_kw_data = hp_power_w_data / 1000
    # hp_cost0, hp_slope = linear_regression(hp_power_kw_data, hp_cost_data)
    # hp_power_kw = np.linspace(1, 5.1, 1000)
    # hp_cost = hp_cost0 + hp_slope * hp_power_kw
    # plt.scatter(hp_power_kw_data, hp_cost_data, **plot_styles[0])
    # plt.plot(hp_power_kw, hp_cost, **plot_styles[1])
    # # plt.title("Heat pump cost")
    # plt.xlabel(r"$kW$")
    # plt.ylabel("€")
    # plt.show()

    # # Solar
    # solar_power_kw = np.linspace(0, 10, 1000)
    # solar_panels_price_per_kw = 1161
    # solar_cost = solar_panels_price_per_kw * solar_power_kw
    # plt.plot(solar_power_kw, solar_cost, **plot_styles[0])
    # plt.title("Solar panels cost")
    # plt.xlabel(r"$kW$")
    # plt.ylabel("€")
    # plt.show()

    # # Generators
    # generator_cost0, generator_slope = linear_regression(generator_power_w_data, generator_cost_data)
    # print(f"generator y0: {generator_cost0}, slope: {generator_slope}")
    # generator_power_w = np.linspace(4000, 170000, 1000)
    # generator_cost = generator_cost0 + generator_slope * generator_power_w
    # plt.scatter(generator_power_w_data, generator_cost_data, **plot_styles[0])
    # plt.plot(generator_power_w, generator_cost, **plot_styles[1])
    # plt.title("Diesel generator cost")
    # plt.xlabel(r"$W$")
    # plt.ylabel("€")
    # plt.show()

    # Generators scaled from W to kW
    generator_power_kw_data = generator_power_w_data / 1000
    generator_cost0, generator_slope = linear_regression(generator_power_kw_data, generator_cost_data)
    print(f"generator y0: {generator_cost0}, slope: {generator_slope}")
    generator_power_kw = np.linspace(4, 170, 1000)
    generator_cost = generator_cost0 + generator_slope * generator_power_kw
    plt.scatter(generator_power_kw_data, generator_cost_data, **plot_styles[0])
    plt.plot(generator_power_kw, generator_cost, **plot_styles[1])
    # plt.title("Diesel generator cost")
    plt.xlabel(r"$kW$")
    plt.ylabel("€")
    plt.show()


def plot_cop():
    t = np.linspace(253, 420, 1000)
    cop_arr = cop(t)
    dcop_values = get_dcopdT(t)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.plot(t - 273, cop_arr, **plot_styles[0])
    ax.set_ylabel("COP")
    ax.set_xlabel("T [ºC]")
    ax0 = ax.twinx()
    ax0.plot(t - 273, dcop_values, **plot_styles[1])
    ax0.set_ylabel(r"$\frac{dCOP}{dT}$")
    plt.show()


if __name__ == "__main__":
    plot_data_regressions()
    # plot_prices()
    # plot_dynamic_parameters()
    # plot_cop()
