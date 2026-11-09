from collections.abc import Callable

from torch import nn

LOSSES: dict[str, type[nn.Module]] = {}


def register_loss(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        LOSSES[name] = cls

        return cls

    return decorator


def build_loss(name: str, **kwargs: float) -> nn.Module:
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}' ({' | '.join(sorted(LOSSES))})")

    return LOSSES[name](**kwargs)


from losses import infonce, margin  # noqa: F401
