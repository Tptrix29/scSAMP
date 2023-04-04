from typing import Union
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyreadr as renv
from anndata import AnnData


def timer(func):
    def wrapper(*args, **kwargs):
        st = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"Time consumption of running '{func.__name__}': {end - st}")

    return wrapper


class Setting():
    def __init__(self):
        self.random_state: int = 0
        self.reset_seed()

    def reset_seed(self):
        self.random_state = np.random.randint(1000)


settings = Setting()


@timer
def to_rds(
        data: AnnData,
        output_file: str,
        obs_feature: Union[str, list] = "cell_type"
) -> None:
    rds: pd.DataFrame = pd.concat([data.to_df(), data.obs[obs_feature]], axis=1)
    renv.write_rds(output_file, rds)

@timer
def to_hdf5(
        source_file: Union[str, list],
        result_dir: str,
        type_label: str,
        source_format: str,
) -> list:
    """
    Convert csv/tab table to h5ad format
    :param source_file: raw file path
    :param result_dir: output directory
    :param type_label: column name
    :param source_format: raw file type
    :return: output file location list
    """
    result_dir = result_dir if result_dir[-1] == '/' else result_dir + '/'
    results_file = []

    if isinstance(source_file, str):
        source_file = [source_file]
    for file in source_file:
        re_file = result_dir + file.split('/')[-1].split('.')[0] + '.h5ad'
        print(f'Loading Data from {file}...')
        if source_format == 'csv':
            data = pd.read_csv(file, header=0)
        elif source_format == 'tab':
            data = pd.read_table(file, header=0)
        else:
            raise ValueError('Invalid source file format.')
        print(f'Data Loaded from {file}.')
        data.index = [str(i) for i in data.index]
        cell_type = data[type_label]
        data.drop(columns=[type_label], inplace=True)
        adata = AnnData(data, dtype=np.float64)
        adata.obs['cell_type'] = pd.Categorical(cell_type)
        adata.write(Path(re_file))
        print(f'HDF5 File saved in {re_file}.')
        results_file.append(re_file)
    return results_file


def _check_obs_key(
        adata: AnnData,
        key: str
) -> bool:
    if key not in adata.obs.columns:
        raise (KeyError(f'Could not find key "{key}" in .obs.columns'))
    if type(adata.obs[key].dtype) != pd.CategoricalDtype:
        raise (KeyError(f'.obs["{key}"] is not pandas.Categorical'))
    return True


def _check_ratio(ratio: float) -> bool:
    if ratio <= 0 or ratio > 1:
        raise ValueError("Invalid ratio: ration range should be (0, 1).")
    return True
