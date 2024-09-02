import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from parameters import PARAMS
from pyoptsparse import History
from utils import (
    get_dynamic_parameters,
    plot_styles,
    plot_film,
    get_fixed_energy_cost_by_second,
    get_battery_depreciation_by_joule,
    get_battery_depreciation_by_second,
    get_solar_panels_depreciation_by_second,
    get_hp_depreciation_by_joule,
    get_hp_depreciation_by_second,
    get_tank_depreciation_by_second,
    get_generator_depreciation_by_joule,
    get_generator_depreciation_by_second,
    load_dict_from_file,
)


def get_costs(histories, parameters):
    p_compressor = histories["p_compressor"][-1]
    solar_size = histories["solar_size"][-1]
    p_bat = histories["p_bat"][-1]
    p_grid_max = histories["p_grid_max"][-1]
    e_bat_max = histories["e_bat_max"][-1]
    tank_volume = histories["tank_volume"][-1]
    p_compressor_max = histories["p_compressor_max"][-1]
    t_room = histories["t_room"][-1]

    h = np.ones(p_compressor.shape[0]) * parameters["H"]
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]
    p_required = parameters["p_required"]
    t_target = parameters["T_TARGET"]
    p_solar = parameters["w_solar_per_w_installed"] * solar_size
    p_grid = -p_solar + p_compressor + p_bat + p_required

    print("p_grid max:", np.max(p_grid))

    return {
        "variable_energy_cost": np.sum(h * np.maximum(pvpc_prices * p_grid, excess_prices * p_grid)),
        "fixed_energy_cost": np.sum(h * get_fixed_energy_cost_by_second(p_grid_max)),
        "battery drep": np.maximum(
            np.sum(h * np.abs(p_bat) * get_battery_depreciation_by_joule(e_bat_max)),
            np.sum(h * get_battery_depreciation_by_second(e_bat_max)),
        ),
        "solar drep": np.sum(h * get_solar_panels_depreciation_by_second(solar_size)),
        "HP drep": np.sum(h * get_hp_depreciation_by_second(p_compressor_max)),
        "tank drep": np.sum(h * get_tank_depreciation_by_second(tank_volume)),
        "generator drep": np.maximum(
            np.sum(h * np.abs(p_grid) * get_generator_depreciation_by_joule(p_grid_max)),
            np.sum(h * get_generator_depreciation_by_second(p_grid_max)),
        ),
    }


def save_plot(fig, ax, filename):
    fig.savefig(filename, format="svg")
    plt.close(fig)


def save_plots(i, histories, parameters, title=None, show=True, block=True, save=True):
    # Design variables
    t_cond = histories["t_cond"][i] - 273  # K to ºC
    t_tank = histories["t_tank"][i] - 273  # K to ºC
    t_floor = histories["t_floor"][i] - 273  # K to ºC
    t_room = histories["t_room"][i] - 273  # K to ºC
    m_dot_cond = histories["m_dot_cond"][i]
    m_dot_heating = histories["m_dot_heating"][i]
    p_compressor = histories["p_compressor"][i] / 1000  # to kW
    e_bat = histories["e_bat"][i]
    e_bat_max = histories["e_bat_max"][i]
    e_bat_max_kwh = histories["e_bat_max"][i].item() / (1000 * 3600)
    p_bat = histories["p_bat"][i] / 1000  # to kW
    p_waste = histories["p_waste"][i] / 1000  # to kW
    solar_size = histories["solar_size"][i]
    tank_volume = histories["tank_volume"][i]

    # Set initial values for controls to the value at t=1
    # Since we can't optimize for t=0, and it may have an arbitrary value
    m_dot_cond[0] = m_dot_cond[1]
    m_dot_heating[0] = m_dot_heating[1]
    p_compressor[0] = p_compressor[1]
    p_bat[0] = p_bat[1]

    # Parameters
    h = parameters["H"]
    t_amb = parameters["t_amb"] - 273  # K to ºC
    n_steps = parameters["t_amb"].shape[0]
    t = np.linspace(0, n_steps * h, n_steps)
    t = t / 3600  # seconds to hours
    t = t / 24  # hours to days
    pvpc_prices = parameters["pvpc_prices"] * (1000 * 3600)  # $/(Ws) to $/(kWh)
    excess_prices = parameters["excess_prices"] * (1000 * 3600)  # $/(Ws) to $/(kWh)
    p_required = parameters["p_required"] / 1000  # to kW
    p_solar = parameters["w_solar_per_w_installed"] * solar_size / 1000  # to kW
    p_grid = -p_solar + p_compressor + p_bat + p_required

    # Plot: prices
    fig, ax = plt.subplots(figsize=(8.27, 2))  # A4: (8.27, 11.69)
    ax.plot(t, pvpc_prices, label="PVPC", **plot_styles[0])
    ax.plot(t, excess_prices, label="Diario", **plot_styles[1])
    ax.set_ylabel(r"€$/(kW \cdot h)$")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_prices.svg")

    # Plot: solar
    fig, ax = plt.subplots(figsize=(8.27, 2))  # Half of A4 height
    ax.plot(t, p_solar, label="Solar", **plot_styles[0])
    ax.plot(t, p_required, label="Consumo", **plot_styles[1])
    ax.set_ylabel("Potencia [kW]")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_generated_consumed.svg")

    # Plot: Temperatures
    fig, ax = plt.subplots(figsize=(8.27, 2))
    ax.plot(t, t_cond, label="Salida bomba calor", **plot_styles[0])
    ax.plot(t, t_tank, label="Tanque", **plot_styles[1])
    ax.plot(t, t_floor, label="Suelo", **plot_styles[2])
    ax.plot(t, t_room, label="Habitación", **plot_styles[3])
    ax.plot(t, t_amb, label="Ambiente", **plot_styles[4])
    ax.set_ylabel("Temperatura [ºC]")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_temperatures.svg")

    # Plot: battery energy
    fig, ax = plt.subplots(figsize=(8.27, 2))
    ax.plot(t, e_bat / e_bat_max, label=f"{e_bat_max_kwh:.2f}kWh", **plot_styles[0])
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.set_ylabel("SOC Batería")
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_battery_soc.svg")

    # Plot: Controls
    fig, ax = plt.subplots(figsize=(8.27, 2))
    ax.plot(t, p_grid, label="Red", **plot_styles[0])
    ax.plot(t, p_compressor, label="Compresor", **plot_styles[1])
    ax.plot(t, p_bat, label="Batería", **plot_styles[2])
    ax.plot(t, p_waste, label="Sobrante", **plot_styles[3])
    ax.set_ylabel("Potencia [kW]")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    ax.set_xticklabels([])  # Hide x-axis labels
    ax.set_xlabel("")  # Remove x-axis label
    save_plot(fig, ax, f"saves/plot_{title}_controls.svg")

    # Plot: flows
    fig, ax = plt.subplots(figsize=(8.27, 2))
    ax.plot(t, m_dot_cond, label="Bomba de calor", **plot_styles[0])
    ax.plot(t, m_dot_heating, label="Suelo radiante", **plot_styles[1])
    ax.set_ylabel(r"Caudal de agua $[kg \cdot s^{-1}]$")
    ax.set_xlabel("días")
    ax.legend(fontsize=8, frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True)
    save_plot(fig, ax, f"saves/plot_{title}_flows.svg")


fig = None


def plot_full(i, histories, parameters, title=None, show=True, block=True, save=True):
    print(f"plotting...{title}")
    global fig
    # Close the previous figure if it exists
    if fig:
        plt.close(fig)

    # Design variables
    t_cond = histories["t_cond"][i]
    t_tank = histories["t_tank"][i]
    t_floor = histories["t_floor"][i]
    t_room = histories["t_room"][i]
    m_dot_cond = histories["m_dot_cond"][i]
    m_dot_heating = histories["m_dot_heating"][i]
    p_compressor = histories["p_compressor"][i]
    p_waste = histories["p_waste"][i]
    e_bat = histories["e_bat"][i]
    p_bat = histories["p_bat"][i]
    solar_size = histories["solar_size"][i]
    tank_volume = histories["tank_volume"][i]

    # Parameters
    h = parameters["H"]
    t_amb = parameters["t_amb"]
    n_steps = parameters["t_amb"].shape[0]
    t = np.linspace(0, n_steps * h, n_steps)
    t = t / 3600
    pvpc_prices = parameters["pvpc_prices"]
    excess_prices = parameters["excess_prices"]
    p_required = parameters["p_required"]
    p_solar = parameters["w_solar_per_w_installed"] * solar_size
    p_grid = -p_solar + p_compressor + p_bat + p_required + p_waste

    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), sharex=True)
    fig.subplots_adjust(hspace=0.5)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # First subplot for inputs
    axes[0].plot(t, pvpc_prices, label="pvpc_price", **plot_styles[0])
    axes[0].plot(t, excess_prices, label="excess_price", **plot_styles[1])
    axes[0].set_ylabel(r"€$/(kW \cdot h)$")
    axes[0].legend()
    axes[0].grid(True)
    ax0 = axes[0].twinx()
    ax0.plot(t, p_required, label="p_required", **plot_styles[2])
    ax0.plot(t, -p_solar, label="p_solar", **plot_styles[3])
    ax0.set_ylabel("W")
    ax0.legend()
    if title:
        axes[0].set_title(title)

    # Second subplot for temperatures and e_bat
    axes[1].plot(t, t_tank, label="t_tank", **plot_styles[0])
    axes[1].plot(t, t_floor, label="t_floor", **plot_styles[1])
    axes[1].plot(t, t_room, label="t_room", **plot_styles[2])
    axes[1].plot(t, t_amb, label="t_amb", **plot_styles[3])
    axes[1].set_ylabel("Temperature (K)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)
    ax1 = axes[1].twinx()
    ax1.plot(t, e_bat, label="E_bat", **plot_styles[4])
    ax1.set_ylabel("Ws")
    ax1.legend(loc="upper right")

    # Third subplot for controls
    axes[2].plot(t, p_grid, label="p_grid", **plot_styles[0])
    axes[2].plot(t, p_compressor, label="p_comp", **plot_styles[1])
    axes[2].plot(t, p_bat, label="p_bat", **plot_styles[2])
    axes[2].plot(t, p_waste, label="p_waste", **plot_styles[3])
    axes[2].set_ylabel("Power[W]")
    axes[2].legend(loc="upper left")
    axes[2].grid(True)
    axes[2].set_xlabel("Time [h]")
    ax2 = axes[2].twinx()
    ax2.plot(t, m_dot_cond, label="m_dot_cond", **plot_styles[4])
    ax2.plot(t, m_dot_heating, label="m_dot_heating", **plot_styles[5])
    ax2.set_ylabel("Mass flow rates")
    ax2.legend(loc="upper right")

    # Show the plots
    if show:
        # Save and close plot
        plt.show(block=block)  # Draw the figure
        plt.pause(0.3)  # Time for the figure to load

    if save:
        plt.savefig(f"tmp/frame_{time.time()}.png")


def plot_history(hist, only_last=True):
    # Get inputs
    storeHistory = History(hist)
    histories = storeHistory.getValues()

    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    t0 = PARAMS["T0"]

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["daily_prices"] = dynamic_parameters["daily_prices"]
    parameters["pvpc_prices"] = dynamic_parameters["pvpc_prices"]
    parameters["excess_prices"] = dynamic_parameters["excess_prices"]

    if only_last:
        indices = [-1]  # Only take the last index
    else:
        # Loop through every x opt results
        x = 10
        indices = list(range(0, len(histories["p_compressor"]), x))

    # loop through histories
    for iter, i in enumerate(indices):
        if only_last:
            title = hist.replace(".hst", "")
            title = title.replace("saves/", "")
            save_plots(i, histories, parameters, title=title, save=False, show=True)
            plot_full(i, histories, parameters, save=False, show=True)

            # Print statistics
            costs = get_costs(histories, parameters)
            print(costs)
            print("p_compressor max:", np.max(histories["p_compressor"][-1]))
            print("p_compressor_max:", histories["p_compressor_max"][-1])
            print("p_grid_max:", histories["p_grid_max"][-1])
            print("e_bat_max[Ws]:", histories["e_bat_max"][-1])
            print("e_bat_max[kWh]:", histories["e_bat_max"][-1] / (1000 * 3600))
            print("tank_volume:", histories["tank_volume"][-1])
            print("solar_size:", histories["solar_size"][-1])
            print("p_bat_max[w]: ", (histories["e_bat_max"][-1] * parameters["C_RATE_BAT"] / 3600))
            return
        else:
            title = f"iter: {iter}/{len(indices)}"
            plot_full(i, histories, parameters, title=title, show=False)

    # create animation with pictures from tmp folder
    filename_gif = hist.replace(".hst", ".gif")
    plot_film(filename_gif)


def plot_dict(dictfile):
    # Get inputs
    dict = load_dict_from_file(dictfile)
    histories = {
        "t_cond": [dict["t_cond"]],
        "t_tank": [dict["t_tank"]],
        "t_floor": [dict["t_floor"]],
        "t_room": [dict["t_room"]],
        "p_bat": [dict["p_bat"]],
        "e_bat": [dict["e_bat"]],
        "p_compressor": [dict["p_compressor"]],
        "p_waste": [dict["p_waste"]],
        "m_dot_cond": [dict["m_dot_cond"]],
        "m_dot_heating": [dict["m_dot_heating"]],
        "e_bat_max": [dict["e_bat_max"]],
        "solar_size": [dict["solar_size"]],
        "p_compressor_max": [dict["p_compressor_max"]],
        "p_grid_max": [dict["p_grid_max"]],
        "tank_volume": [dict["tank_volume"]],
    }

    print("p_grid_max dict", dict["p_grid_max"])

    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    t0 = PARAMS["T0"]

    dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
    parameters = PARAMS
    parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
    parameters["p_required"] = dynamic_parameters["p_required"]
    parameters["t_amb"] = dynamic_parameters["t_amb"]
    parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
    parameters["daily_prices"] = dynamic_parameters["daily_prices"]
    parameters["pvpc_prices"] = dynamic_parameters["pvpc_prices"]
    parameters["excess_prices"] = dynamic_parameters["excess_prices"]

    i = 0
    title = dictfile.replace(".hst", "")
    title = title.replace("saves/", "")
    save_plots(i, histories, parameters, title=title, save=False, show=True)
    plot_full(i, histories, parameters, save=False, show=True)

    # Print statistics
    costs = get_costs(histories, parameters)
    print(costs)
    print("p_compressor max:", np.max(histories["p_compressor"][-1]))
    print("p_compressor_max:", histories["p_compressor_max"][-1])
    print("p_grid_max:", np.max(histories["p_grid_max"][-1]))
    print("e_bat_max[Ws]:", histories["e_bat_max"][-1])
    print("e_bat_max[kWh]:", histories["e_bat_max"][-1] / (1000 * 3600))
    print("tank_volume:", histories["tank_volume"][-1])
    print("solar_size:", histories["solar_size"][-1])
    print("p_bat_max[w]: ", (histories["e_bat_max"][-1] * parameters["C_RATE_BAT"] / 3600))
    return


if __name__ == "__main__":
    # plot_history(hist="saves/sizing_regulated.hst", only_last=True)
    plot_dict("saves/sizing_regulated.pkl")
    # plot_history(hist="saves/sizing_free_market.hst", only_last=True)
    # plot_dict("saves/sizing_free_market.pkl")
    # plot_history(hist="saves/sizing_off_grid.hst", only_last=True)
    # plot_dict(dictfile="saves/sizing_off_grid.pkl")
