import numpy as np
from parameters import PARAMS
from utils import get_dynamic_parameters
from full_system.full_w_sizing import run_optimization


def main():
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    # full_horizon = 365 * 24 * 3600
    full_horizon = horizon * 10
    starArr = []
    parameters = PARAMS
    y0 = {
        "t_cond": 330,
        "t_tank": 310,
        "t_out_heating": 337.13903948,
        "t_floor": 303.15567351,
        "t_room": 300.65425203,
        "e_bat": PARAMS["E_BAT_MAX"] * PARAMS["SOC_MAX"],
        "p_bat": 1e-2,
    }
    parameters["y0"] = y0

    for t0 in range(0, full_horizon, h):
        print(t0)
        continue
        dynamic_parameters = get_dynamic_parameters(t0, h, horizon, year=2022)
        parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
        parameters["p_required"] = dynamic_parameters["p_required"]
        parameters["t_amb"] = dynamic_parameters["t_amb"]
        parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
        parameters["pvpc_prices"] = dynamic_parameters["pvpc_prices"]
        parameters["excess_prices"] = dynamic_parameters["excess_prices"]

        # Run
        xStar, objStar = run_optimization(parameters, plot=True)
        starArr.append(
            {
                "xStar": {
                    "e_bat_max": xStar["e_bat_max"],
                    "solar_size": xStar["solar_size"],
                    "p_compressor": xStar["p_compressor"],
                    "p_grid_max": xStar["p_grid_max"],
                    "tank_volume": xStar["tank_volume"],
                },
                "objStar": objStar,
            }
        )

        # Set next iteration initial conditions
        y0 = {
            "t_cond": xStar["t_cond"][1],
            "t_tank": xStar["t_tank"][1],
            "t_out_heating": xStar["t_out_heating"][1],
            "t_floor": xStar["t_floor"][1],
            "t_room": xStar["t_room"][1],
            "e_bat": xStar["e_bat"][1],
            "p_bat": xStar["p_bat"][1],
        }
        parameters["y0"] = y0

    print(starArr)


if __name__ == "__main__":
    main()
