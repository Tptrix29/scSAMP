import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from typing import Union
from abc import abstractmethod, ABCMeta

"""
Logging: BasePreprocessor
- Basic Preprocessing
- HVG Preprocessor
- PCA Preprocessor
"""


class BasePreprocessor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.gene_index: Union[pd.Index, None] = None
        self.is_prior = False
        self.basic_params: dict = {
            'cells_threshold':  3,
            'genes_threshold': 200,
            'target_threshold': 1e4,
            'max_threshold': 10
        }

    def reset_params(self,
                     cells_threshold: int = 3,
                     genes_threshold: int = 200,
                     target_threshold: int = 1e4,
                     max_threshold: int = 10
                     ) -> None:
        self.basic_params['cells_threshold'] = cells_threshold
        self.basic_params['cells_threshold'] = genes_threshold
        self.basic_params['cells_threshold'] = target_threshold
        self.basic_params['cells_threshold'] = max_threshold

    @abstractmethod
    def display_params(self):
        for k, v in self.basic_params.items():
            print(f"{k}: {v}")

    @abstractmethod
    def refPreprocessing(self, ref: AnnData) -> AnnData:
        pass

    @abstractmethod
    def queryPreprocessor(self, query: AnnData) -> AnnData:
        pass

    def basicProcessing(self, adata) -> None:
        """
        Basic Preprocessing Steps, including:
        1. Filtering cells
        2. Normalization by counts per cell, every cell has the same total count after normalization
        3. Logarithm Transformation
        :param adata: reference data
        :return: None (Default inplace = True)
        """
        sc.pp.filter_cells(adata, min_genes=self.basic_params['genes_threshold'])
        sc.pp.normalize_total(adata, target_sum=self.basic_params['target_threshold'])
        sc.pp.log1p(adata)

    def test_prior(self) -> None:
        if not self.is_prior:
            raise (ValueError("Use 'refPreprocessing' to get index first."))


class BasicPreprocessor(BasePreprocessor):
    """
    Non feature selection.
    """
    def __init__(self):
        super().__init__()

    def display_params(self):
        super().display_params()

    def refPreprocessing(self, ref: AnnData) -> AnnData:
        sc.pp.filter_genes(ref, min_cells=self.basic_params['cells_threshold'])
        super().basicProcessing(adata=ref)
        sc.pp.scale(ref, max_value=self.basic_params['max_threshold'])
        self.gene_index = ref.var_names
        self.is_prior = True
        return ref

    def queryPreprocessor(self, query: AnnData) -> AnnData:
        super().test_prior()
        new_query = query[:, self.gene_index]
        super().basicProcessing(adata=new_query)
        sc.pp.scale(new_query, max_value=self.basic_params['max_threshold'])
        return new_query


class HVGPreprocessor(BasePreprocessor):
    """
    Highly Variable Genes (HVGs) for feature selection.
    Steps:
    1. Basic preprocessing
    2. HVG selection
    """
    def __init__(self, n_hvg: int = 1000):
        super().__init__()
        self.n_hvg: int = n_hvg

    def display_params(self):
        super().display_params()
        print(f'HVG Number: {self.n_hvg}')

    def reset_n_hvg(self, n_hvg: int) -> None:
        self.n_hvg = n_hvg

    def refPreprocessing(self, ref: AnnData) -> AnnData:
        super().basicProcessing(adata=ref)
        sc.pp.highly_variable_genes(ref, n_top_genes=self.n_hvg, inplace=True)
        self.gene_index = ref.var[ref.var['highly_variable']].index
        self.is_prior = True
        ref = ref[:, self.gene_index]
        sc.pp.scale(ref, max_value=self.basic_params['max_threshold'])
        return ref

    def queryPreprocessor(self, query: AnnData) -> AnnData:
        super().test_prior()
        super().basicProcessing(adata=query)
        new_query = query[:, self.gene_index]
        sc.pp.scale(new_query, max_value=self.basic_params['max_threshold'])
        return new_query


class PCAPreprocessor(BasePreprocessor):
    """
    Principle Components Analysis for feature selection.
    Steps:
    1. Basic preprocessing
    2. HVG selection
    3. PC selection
    """
    def __init__(self, n_pc: int = 40):
        super().__init__()
        self.n_pc: int = n_pc
        self.trans_matrix: np.array = None

    def display_params(self):
        super().display_params()
        print(f'PC Number: {self.n_pc}')
        print(f'Transformation Matirx Shape: {self.trans_matrix.shape}')

    def refPreprocessing(self, ref: AnnData) -> AnnData:
        # sc.tl.pca(adata, svd_solver='arpack')
        pass

    def queryPreprocessor(self, query: AnnData) -> AnnData:
        pass


