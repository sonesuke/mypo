"""Parameters for runner."""
from dataclasses import dataclass
from typing import Final

import numpy as np


@dataclass
class Settings:
    """Settings for runner."""

    tax_rate: np.float64
    fee_rate: np.float64


DEFAULT_SETTINGS: Final[Settings] = Settings(tax_rate=np.float64(0.20), fee_rate=np.float64(0.005))
