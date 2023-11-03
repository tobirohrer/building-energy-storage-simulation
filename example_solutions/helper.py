import pandas as pd
import numpy as np
from typing import Tuple

# Start and end Index of data used for testing
TEST_INDEX_START = 4380
TEST_INDEX_END = 8500

BATTERY_CAPACITY = 400
BATTERY_POWER = 100


def read_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    load = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/electricity_load_profile.csv')[
        'Load [kWh]']
    price = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/electricity_price_profile.csv')[
        'Day Ahead Auction']
    generation = pd.read_csv('../building_energy_storage_simulation/data/preprocessed/solar_generation_profile.csv')[
        'Generation [kWh]']
    return np.array(load), np.array(price), np.array(generation)


