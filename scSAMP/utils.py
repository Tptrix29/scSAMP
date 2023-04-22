import os
import numpy as np
import pandas as pd
import pyreadr as renv
from typing import Union
from pathlib import Path
from anndata import AnnData

from .decorator import time_logging


class Setting:
    def __init__(self):
        self.random_state: int = 0
        self.reset_seed()

    def reset_seed(self):
        self.random_state = np.random.randint(1000)


settings = Setting()


def sampling_stream(
        data: AnnData,
        output_dir: str,
        prefix: str,
        suffix: str,
        ratio_range: np.array,
) -> list:
    """
    IO stream of sampling evaluation.
    Parameters
    ----------
    data: :class:`anndata.AnnData`
        Raw dataset.
    output_dir: str
        Output directory.
    prefix: str
        Output file prefix.
    suffix: str
        Output file suffix.
    ratio_range: :class:`numpy.array`
        Sampling ratio range.

    Returns
    -------
        List of output filename.
    """
    if os.path.exists(output_dir):
        raise FileExistsError(f"'{output_dir}' is not empty, please choose an empty directory")
    else:
        os.mkdir(output_dir)
    from .processing.sampler import SamplingProcessor, SamplingStrategy
    sampler = SamplingProcessor(reference=data, cluster_col="cell_type")
    files = []
    for r in ratio_range:
        sampler.set_size(ratio=r)
        for s in SamplingStrategy:
            s = str(s)
            sampled_data: AnnData = sampler.sampling(strategy=s)
            filename = "_".join([prefix, str(r), s, suffix]) + ".rds"
            files.append(filename)
            output = output_dir + '/' + filename
            print(f"Written {filename} at {output_dir}")
            to_rds(sampled_data, output)

    return files


@time_logging(mode="rds file stream")
def to_rds(
        data: AnnData,
        output_file: str,
        obs_feature: Union[str, list] = "cell_type"
) -> None:
    rds: pd.DataFrame = pd.concat([data.to_df(), data.obs[obs_feature]], axis=1)
    renv.write_rds(output_file, rds)


@time_logging(mode="hdf5 file stream")
def to_hdf5(
        source_file: Union[str, list],
        result_dir: str,
        type_label: str,
        source_format: str,
) -> list:
    """
    Convert `csv`/`tab` table to `h5ad` format.

    Parameters
    ----------
    source_file
        Raw file path.
    result_dir
        Output directory.
    type_label
        Column name of cell type label.
    source_format
        Source file type, chosen from `csv` or `tab`
    Returns
    -------
        Written filepath list.
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
        raise ValueError("Invalid ratio: ration range should be [0, 1).")
    return True
