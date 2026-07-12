"""EMA helper for diffusion model checkpoints."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use EMA") from exc


class ExponentialMovingAverage:
    """Power-ramped EMA shadow state."""

    def __init__(self, model: torch.nn.Module, power: float = 0.75, max_decay: float = 0.9999) -> None:
        self.power = float(power)
        self.max_decay = float(max_decay)
        self.num_updates = 0
        self.shadow = {name: param.detach().clone() for name, param in model.state_dict().items() if torch.is_floating_point(param)}

    def decay(self) -> float:
        value = 1.0 - (1.0 + self.num_updates) ** (-self.power)
        return float(min(self.max_decay, max(0.0, value)))

    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        decay = self.decay()
        state = model.state_dict()
        for name, shadow_value in self.shadow.items():
            shadow_value.mul_(decay).add_(state[name].detach(), alpha=1.0 - decay)

    def state_dict(self) -> dict[str, object]:
        return {"power": self.power, "max_decay": self.max_decay, "num_updates": self.num_updates, "shadow": self.shadow}
