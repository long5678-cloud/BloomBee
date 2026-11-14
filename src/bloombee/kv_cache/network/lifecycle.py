"""Visible multi-stage failures without replacing the actual model outcome."""

class LifecycleError(RuntimeError):
    def __init__(self, errors, *, terminal=None):
        self.errors = list(errors)
        self.terminal = terminal
        super().__init__('; '.join(f"{e['stage']}: {e['type']}: {e['message']}" for e in self.errors))


def failure(stage, error):
    return dict(stage=stage, type=type(error).__name__, message=str(error))


def boolean(config, name, default=False):
    value = config.get(name, default)
    if type(value) is not bool:
        raise TypeError(name+' must be an explicit boolean')
    return value
