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
                f"model type '{model_type}' not implemented in stan.\n"
                + f"known models: {_models_implemented}."
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

    def validate_prior_settings(self, prior):
        """
        make sure that the prior settings are either input correctly or set
        correctly in the class already
        """
        if (prior is None) and (self.prior is None):
            raise RuntimeError("user must specify a prior guess.")
        elif prior is not None:
            self.model_type.validate_prior(prior)
        else:  # use the default one
            prior = self.prior

    @staticmethod
    def validate_Ndata(t_in: np.ndarray, g_in: np.ndarray):
        """
        assert that the independent and dependent variables are the same length,
        return that length if it matches
        """
        assert len(g_in) == len(t_in), "g(t) and t must match"
        N_data = len(t_in)
        return N_data

    @staticmethod
    def pack_and_stash_stan_data(
        N_in: np.ndarray,
        t_in: np.ndarray,
        g_in: np.ndarray,
        prior_in: dict[str, int | float],
        fn_in: str,
    ):
        """
        pack up the stan data for this case and save it as a json in the
        working directory for access by cmdstan
        """

        stan_data = (
            {
                "N": N_in,
                "t": t_in.tolist(),
                "g": g_in.tolist(),
                "mu_J": prior_in["mu_J"],
                "std_J": prior_in["std_J"],
                "mu_sigma": prior_in["mu_sigma"],
                "std_sigma": prior_in["std_sigma"],
                "std_A": prior_in["std_A"],
                "mu_T": prior_in["mu_T"],
                "std_T": prior_in["std_T"],
            },
        )[
            0
        ]  # black is killing me with these

        with open(fn_in, "w") as iofile_json:
            json.dump(stan_data, iofile_json, indent=2)

    def fit(
        self,
        t_data: np.ndarray,
        g_data: np.ndarray,
        prior: dict[str, int | float] | None = None,
        optimizer_args: dict[str, any] = dict(),
    ):
        """
        fit a transient model using the maximum a posteriori estimate by MLE

        parameters
        ----------
        t_data: np.ndarray
            sequence of sampling times for transient fit
        g_data: np.ndarray
            sequence of sample values for transient fit
        prior: dict, optional
            definitions for the prior model
        optimizer_args: dict, optional
            arguments to be passed through to cmdstanpy.CmdStanModel.optimize

        """

        # perform validations
        N_data = self.validate_Ndata(t_data, g_data)
        self.validate_prior_settings(prior)

        # pack and run stan model
        fn_stan_data = "stan_data.json"
        self.pack_and_stash_stan_data(N_data, t_data, g_data, prior, fn_stan_data)
        self.optim = self.model.optimize(data=fn_stan_data, **optimizer_args)

    def sample(
        self,
        t_data: np.ndarray,
        g_data: np.ndarray,
        prior: dict[str, int | float] | None = None,
        sample_args: dict[str, any] = dict(),
    ):
        """
        sample from the posterior of transient models using HMC

        parameters
        ----------
        t_data: np.ndarray
            sequence of sampling times for transient fit
        g_data: np.ndarray
            sequence of sample values for transient fit
        prior: dict, optional
            definitions for the prior model
        sample_args: dict, optional
            arguments to be passed through to cmdstanpy.CmdStanModel.sample

        """

        # perform validations
        N_data = self.validate_Ndata(t_data, g_data)
        self.validate_prior_settings(prior)

        # pack and run stan model
        fn_stan_data = "stan_data.json"
        self.pack_and_stash_stan_data(N_data, t_data, g_data, prior, fn_stan_data)
        self.sampling = self.model.sample(data=fn_stan_data, **sample_args)
