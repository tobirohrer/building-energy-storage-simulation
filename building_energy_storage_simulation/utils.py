import pkg_resources
import pandas as pd


def load_profile(filename, column_name):
    stream = pkg_resources.resource_stream(__name__, 'data/preprocessed/' + filename)
    return pd.read_csv(stream)[column_name]
