datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    dataset = datasets[name](config)
    return dataset


from . import nerfactor_relight, shiny_relight # synthetic dataset
from . import spheric # generate spheric trace for visualization