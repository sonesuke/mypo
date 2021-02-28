from dataclasses import dataclass
import numpy as np

@dataclass
class Settings:
    spending: np.float64
    tax_rate: np.float64
    fee_rate: np.float64