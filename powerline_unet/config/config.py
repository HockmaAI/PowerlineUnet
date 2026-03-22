"""
Configuration management for PowerlineUnet.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    name: str = "UNet"
    in_channels: int = 3
    out_channels: int = 1
    encoder_channels: list = None
    use_attention: bool = True
    pretrained: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    bce_weight: float = 0.7
    dice_weight: float = 0.3
    early_stopping_patience: int = 10
    save_best_only: bool = True


@dataclass
class PathConfig:
    data_root: str = "data"
    train_images: str = "data/train/train_imgs"
    train_masks: str = "data/train/train_gt"
    val_images: str = "data/val/val_imgs"
    val_masks: str = "data/val/val_gt"
    output_dir: str = "outputs"
    model_save_dir: str = "outputs/models"
    inference_input: str = "data/inference/in"
    inference_output: str = "data/inference/out"


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("configs/default.yaml")
        self.config = self._load_config()
        
        self.paths = PathConfig(**self.config.get('paths', {}))
        self.model = ModelConfig(**self.config.get('model', {}))
        self.training = TrainingConfig(**self.config.get('training', {}))
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    @property
    def project_root(self) -> Path:
        """Return project root directory."""
        return self.config_path.parent


# Global config instance
config = Config()
