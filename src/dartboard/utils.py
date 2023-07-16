import os


def package_paths(file_in):
    fn_module = os.path.abspath(file_in)
    dir_package = os.path.abspath(os.path.join(os.path.dirname(fn_module), "../.."))

    return (
        fn_module,
        dir_package,
        {
            "src": os.path.join(dir_package, "src", "dartboard"),
            "stan": os.path.join(dir_package, "src", "dartboard", "stan_models"),
        },
    )
