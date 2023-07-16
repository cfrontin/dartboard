import os
import json

import numpy as np

import cmdstanpy
import logging

from .utils import package_paths
from .stan_models.transient.decay_exponential import DecayExponential

# configure logger to prevent output
_logger = logging.getLogger("cmdstanpy")
_logger.addHandler(logging.NullHandler())
_logger.propagate = False
_logger.setLevel(logging.WARNING)

# get the key locations for the package
_fn_module, _dir_package, _key_dirs = package_paths(__file__)

# filename for the stan transient models
dir_stan_transient = os.path.join(_key_dirs["stan"], "transient")

# private lists of required values
_models_implemented = {
    "decay_exponential": DecayExponential,
}


class TransientFitter(object):
    """
    this is a class to specify and run a transient fit

    parameters
    ----------
    model_type: str, optional
        a specification of the transient model (exponential decay by default)
    name: str, optional
        name for the cmdstan model (defaults to :model_type:)
    prior: dict, optional
        specifications of prior model parameters

    attributes
    ----------
    model_type: str
        the type of transient model to be fit
    fn_stan: str
        the filename of the stan model being used
    model: CmdStanModel
        a cmdstanpy model object for the fit
    optim: CmdStanMLE
        the cmdstanpy MLE result
    sampling: CmdStanMCMC
        the cmdstanpy MCMC results

    """

    def __init__(
        self,
        model_type: str = "decay_exponential",
        name: str = None,
        prior: dict[str, int | float] = None,
    ):
        if model_type not in _models_implemented:
            raise NotImplementedError(
                f"model type '{model_type}' not implemented in stan.\nknown models: {_models_implemented}."
            )
        else:
            self.model_type = _models_implemented[model_type]
            self.fn_stan = self.model_type.get_filename_stan_model()

        self.model = cmdstanpy.CmdStanModel(
            model_type if name is None else name, stan_file=self.fn_stan
        )

        if prior is not None:
            self.model_type.validate_prior(prior, model_type)
        else:
            self.prior = None

        self.optim = None
        self.sampling = None

    def fit(
        self,
        t_data: np.ndarray,
        g_data: np.ndarray,
        prior: dict[str, int | float] | None = None,
    ):
        """
        fit a transient model using the maximum a posteriori estimate by MLE

        parameters:
        - t_data (np.ndarray): sequence of sampling times for transient fit
        - g_data (np.ndarray): sequence of sample values for transient fit
        - prior (optional, dict): definitions for the prior model

        returns: none
        """
        assert len(g_data) == len(t_data), "g(t) and t must match"
        N_data = len(t_data)

        if (prior is None) and (self.prior is None):
            raise RuntimeError("user must specify a prior guess.")
        elif prior is not None:
            self.model_type.validate_prior(prior)
        else:  # use the default one
            prior = self.prior

        stan_data = (
            {
                "N": N_data,
                "t": t_data.tolist(),
                "g": g_data.tolist(),
                "mu_J": prior["mu_J"],
                "std_J": prior["std_J"],
                "mu_sigma": prior["mu_sigma"],
                "std_sigma": prior["std_sigma"],
                "std_A": prior["std_A"],
                "mu_T": prior["mu_T"],
                "std_T": prior["std_T"],
            },
        )[0]

        with open("stan_data.json", "w") as iofile_json:
            json.dump(stan_data, iofile_json, indent=2)

        self.optim = self.model.optimize(data="stan_data.json")
