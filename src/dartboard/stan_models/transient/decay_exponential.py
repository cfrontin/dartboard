import os.path

import numpy as np

import dartboard.stan_models.base as base


class DecayExponential(base.BaseStanModel):
    _file: str = os.path.abspath(__file__)
    _basename: str = os.path.basename(os.path.split(_file)[-1])
    _prior_keys: list[str] = [
        "mu_J",
        "std_J",
        "mu_sigma",
        "std_sigma",
        "std_A",
        "mu_T",
        "std_T",
    ]
    _param_keys: list[str] = [
        "Jinf",
        "sigma",
        "T_lambda",
        "A",
    ]

    @classmethod
    def model_function(cls, params: dict[str, int | float] | None):
        cls.validate_params(params)

        def f_out(t: np.ndarray, z: float = 0.0) -> np.ndarray:
            return (
                params["Jinf"]
                + params["A"] * np.exp(-t / params["T_lambda"])
                + z * params["sigma"]
            )

        return f_out
