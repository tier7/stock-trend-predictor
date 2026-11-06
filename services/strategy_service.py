from dataclasses import dataclass

from math import isfinite


DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def validate_confidence_threshold(value):
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError("Próg pewności musi być liczbą od 50% do 100%.")
    if not isfinite(threshold) or not 0.5 <= threshold <= 1:
        raise ValueError("Próg pewności musi być liczbą od 50% do 100%.")
    return threshold


@dataclass(frozen=True)
class StrategyConfig:
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    min_net_return: float = 0.0
    position_fraction: float = 1.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0

    def __post_init__(self):
        for name in ("confidence_threshold", "min_net_return", "position_fraction", "fee_rate", "slippage_rate"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")

        validate_confidence_threshold(self.confidence_threshold)

        if self.min_net_return < 0:
            raise ValueError("Minimum net return cannot be negative")

        if not 0 < self.position_fraction <= 1:
            raise ValueError("Position fraction must be greater than 0 and at most 1")

        if not 0 <= self.fee_rate < 1:
            raise ValueError("Fee rate must be between 0 and 1")

        if not 0 <= self.slippage_rate < 1:
            raise ValueError("Slippage rate must be between 0 and 1")


def should_enter_position(prediction, model_config, strategy_config):
    if model_config.task == "classification":
        probability_up = float(prediction["probability_up"])

        if not isfinite(probability_up) or not 0 <= probability_up <= 1:
            raise ValueError("Probability up must be between 0 and 1")

        if probability_up > 0.5 and probability_up >= strategy_config.confidence_threshold:
            return True
        return False

    elif model_config.task == "regression":
        predicted_return =  float(prediction["predicted_return"])

        if not isfinite(predicted_return):
            raise ValueError("Predicted return must be finite")

        fee = strategy_config.fee_rate
        slippage = strategy_config.slippage_rate

        entry_cost = (1+slippage) * (1+fee)
        exit_proceeds = (1 - slippage) * (1-fee)
        predicted_net_return = ((1+predicted_return) * exit_proceeds / entry_cost - 1)

        if predicted_net_return > strategy_config.min_net_return:
            return True
        return False

    raise ValueError("Unsupported task")