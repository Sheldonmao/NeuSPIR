systems = {}


def register(name):
    def decorator(cls):
        systems[name] = cls
        return cls
    return decorator


def make(name, config, load_from_checkpoint=None):
    if load_from_checkpoint is None:
        system = systems[name](config)
    else:
        print("loading from ckpt:",load_from_checkpoint)
        system = systems[name].load_from_checkpoint(load_from_checkpoint, strict=False, config=config)
    return system


from . import  neus, neusPIL, neusPIR,neusPIR_indirect
