models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    model = models[name](config)
    return model

from . import nerf, neus, neuspil, neuspir, neuspir_indirect, model_utils
