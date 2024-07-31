import numpy as np
from parameters import PARAMS
from utils import get_dynamic_parameters
from mdf_wo_finn import run_optimization


def main():
    n_hours = PARAMS["N_HOURS"]
    n_seconds = n_hours * 3600
    h = PARAMS["H"]
    horizon = PARAMS["HORIZON"]
    n_steps = int(horizon / h)
    starArr = []
    y0 = [298.34089176, 309.70395426]  # T_tank, T_cond

    for t0 in range(0, n_seconds, h):
        dynamic_parameters = get_dynamic_parameters(t0, h, horizon)
        parameters = PARAMS
        parameters["q_dot_required"] = dynamic_parameters["q_dot_required"]
        parameters["p_required"] = dynamic_parameters["p_required"]
        parameters["t_amb"] = dynamic_parameters["t_amb"]
        parameters["w_solar_per_w_installed"] = dynamic_parameters["w_solar_per_w_installed"]
        parameters["daily_prices"] = dynamic_parameters["daily_prices"]
        parameters["pvpc_prices"] = dynamic_parameters["pvpc_prices"]
        parameters["excess_prices"] = dynamic_parameters["excess_prices"]
        parameters["y0"] = y0

        xStar, objStar = run_optimization(parameters)
        exit(0)
        starArr.append({"xStar": xStar, "objStar": objStar})

        # Get next y0
        u = [
            xStar["p_compressor"],
            xStar["m_dot_cond"],
            xStar["m_dot_load"]
        ]
        dae_p = [
            parameters["CP_WATER"],
            parameters["TANK_VOLUME"] * parameters["RHO_WATER"],  # tank mass
            parameters["U"],
            6 * np.pi * (parameters["TANK_VOLUME"] / (2 * np.pi)) ** (2 / 3),  # tank surface area (m2)
            parameters["t_amb"],
            parameters["LOAD_HX_EFF"]
        ]
        n_steps = 1
        next_y = dae_forward(y0, dae_p, u, n_steps)
        y0 = next_y


if __name__ == "__main__":
    main()
