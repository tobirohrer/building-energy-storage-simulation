from pathlib import Path

import pandas as pd
import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

# Start and end Index of data used for testing
TEST_INDEX_START = 4380
TEST_INDEX_END = 8500

BATTERY_CAPACITY = 400
BATTERY_POWER = 100


def read_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_path = Path(__file__).parent
    folder_path = (base_path / "../building_energy_storage_simulation/data/preprocessed/").resolve()

    load = pd.read_csv(folder_path / 'electricity_load_profile.csv')[
        'Load [kWh]']
    price = pd.read_csv(folder_path / 'electricity_price_profile.csv')[
        'Day Ahead Auction']
    generation = pd.read_csv(folder_path / 'solar_generation_profile.csv')[
        'Generation [kWh]']
    return np.array(load), np.array(price), np.array(generation)


def plot_control_trajectory(residual_load, augmented_load, price, battery_power) -> None:
    ax = plt.subplot()
    ax.plot(residual_load, label='Residual Load')
    ax.plot(augmented_load, label='Augmented Load')
    ax.plot(price, '--', label='Price')
    ax.plot(battery_power, label='Battery Power')
    plt.ylabel('Load and Battery Power Applied (kW) & Price (Cent per kWh)')
    plt.xlabel('Time Step')
    ax.legend()
    ax.grid()
    plt.show()
