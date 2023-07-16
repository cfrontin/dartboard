import os.path

import abc


class BaseStanModel(abc.ABC):
    """
    the base class to hold a stan model and represent its form in python


    """

    _file: str = os.path.abspath(__file__)
    _basename: str = os.path.basename(os.path.split(_file)[-1])
    _prior_keys: list[str]
    _param_keys: list[str]

    @classmethod
    def get_filename_stan_model(cls) -> str:
        """
        get the filename of the stan model for this class
        """
        fn_here = os.path.abspath(cls._file)
        fn_stan = os.path.splitext(fn_here)[0] + ".stan"
        return fn_stan

    @classmethod
    def get_prior_keys(cls) -> list[str]:
        """
        get the keys for the prior for this problem
        """
        return cls._prior_keys

    @classmethod
    def validate_prior(cls, prior_dict: dict[str, any]):
        """
        make sure a prior specification dictionary is valid, i.e. has the right keys
        """
        for key in cls.get_prior_keys():
            assert (
                key in prior_dict
            ), f"key {key} must be in priors to fit the {cls._basename} model."

    @classmethod
    def get_param_keys(cls) -> list[str]:
        """
        get the keys for the params for this problem
        """
        return cls._param_keys

    @classmethod
    def validate_params(cls, param_dict: dict[str, any]):
        """
        make sure a param specification dictionary is valid, i.e. has the right keys
        """
        for key in cls.get_param_keys():
            assert (
                key in param_dict
            ), f"key {key} must be in params to fit the {cls._basename} model."

    @classmethod
    @abc.abstractmethod
    def model_function(cls, params: dict[str, int | float] | None):
        """
        take a parameterization of the inferred model and return a function to
        show the model at that parameterization as a function of some
        (problem-dependent) independent variables and a given z-score
        """

        def fun0(*var_indep, z: float = 0.0):
            pass

        return fun0
