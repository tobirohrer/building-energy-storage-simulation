import pkg_resources
import pandas as pd


def load_profile(filename: str, column_name: str) -> pd.Series:
    stream = pkg_resources.resource_stream(__name__, 'data/preprocessed/' + filename)
    return pd.read_csv(stream)[column_name]
