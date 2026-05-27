import torch
from dataclasses import dataclass, field


@dataclass
class KANConfig:
    """
    KAN model configuration
    """
    hidden_dim: int = 256
    history_depth: int = 8


@dataclass
class SolverConfig:
    """
    Solver configuration
    """
    lr: float = 0.0001
    weight_decay: float = 0.0001
    step_size: int = 500
    gamma: float = 0.5


@dataclass
class ModelConfig:
    """
    Model configuration
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    kan: KANConfig = field(default_factory=KANConfig)


@dataclass
class Config:
    """
    Main predictor configuration class
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
