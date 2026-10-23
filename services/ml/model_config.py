from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class ModelConfig:
    algorithm: str
    task: str
    scope: str = "global"
    horizon: int = 1
    feature_groups: tuple[str,...] = ("technical",)
    random_state: int = 26


    def __post_init__(self):
        if self.algorithm not in ("random_forest", "mlp", "lightgbm"):
            raise ValueError("Unsupported algorithm")

        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("Horizon must be a positive integer")

        if self.task not in ("classification", "regression"):
            raise ValueError("Unsupported task")

        if self.scope not in ("global", "local"):
            raise ValueError("Unsupported scope")

        if not isinstance(self.feature_groups, tuple) or not self.feature_groups:
            raise ValueError("feature_groups must be a non-empty tuple")

        allowed_groups = ("technical", "news", "market", "sector", "calendar")

        for group in self.feature_groups:
            if not isinstance(group, str) or group not in allowed_groups:
                raise ValueError(f"Unsupported feature group: {group}")

        if len(set(self.feature_groups)) != len(self.feature_groups):
            raise ValueError("Duplicate feature groups")

        if type(self.random_state) is not int or not 0 <= self.random_state < 2 ** 32:
            raise ValueError("random_state must be an integer in [0, 2**32)")

    @property
    def target_column(self):
        if self.task == "classification":
            return "target"
        else:
            return "target_return"

    def to_dict(self):
        values = asdict(self)
        values["feature_groups"] = list(self.feature_groups)
        return values

    @classmethod
    def from_dict(cls, data):
        values = data.copy()

        if "feature_groups" in values:
            values["feature_groups"] = tuple(values[f"feature_groups"])

        return cls(**values)
