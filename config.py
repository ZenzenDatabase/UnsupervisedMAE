# config.py

import os
from pathlib import Path

class BaseConfig:
    
    # Common Hyperparameters
    AUDIO_INPUT_SIZE = 128
    VISUAL_INPUT_SIZE = 1024
    HIDDEN_SIZE = 1024
    NUM_EPOCHS: int = 100
    BATCH_SIZE: int = 512
    OUTPUT_SIZE: int = 64
    MASK_RATIO: int = 0.3
    
    # Loss/Regularization Parameters
    ALPHA: float = 1e-3
    BETA: float = 1e-1
    
    # Feature Names
    AUDIO_FEATURE: str = "vggish_Features"
    VISUAL_FEATURE: str = "inception_Features"
    
    SEED: int = 42 
    
    # Root directory 
    ROOT_DIR: Path = Path("/home/user/hsc_mae")


# --- Dataset Specific Configurations ---
class AVEConfig(BaseConfig):    
    # Dataset Path
    DATA_PATH: Path = Path("/home/user/hsc_mae/data/ave/AVE.h5")
    
    # Dataset Specific Parameters
    NUM_CLASSES: int = 15


class VEGASConfig(BaseConfig):    
    # Dataset Path
    DATA_PATH: Path = Path("/home/user/hsc_mae/data/vegas/VEGAS.h5")
    
    NUM_CLASSES: int = 10

def get_config(dataset_name: str) -> BaseConfig:

    if dataset_name.lower() == 'ave':
        return AVEConfig()
    elif dataset_name.lower() == 'vegas':
        return VEGASConfig()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Must be 'ave', 'vegas'")