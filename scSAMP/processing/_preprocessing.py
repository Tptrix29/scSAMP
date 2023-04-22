import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Union
from abc import abstractmethod, ABCMeta


class BasePreprocessor(metaclass=ABCMeta):
    """
    Base abstract class for preprocessing.

    Attributes
    ------------
    gene_index: :class:`pandas.Index`
        Selected genes' index
    is_prior: bool
        Weather reference is preprocessed
    basic_params: dict
        Preprocessing parameters

    Notes
    --------

    """
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
        """
        Preprocessing reference dataset.
        Parameters
        ----------
        ref :class:`anndata.AnnData`
            reference dataset
        Returns
        -------
        Preprocessed reference dataset
        """
        pass

    @abstractmethod
    def queryPreprocessor(self, query: AnnData) -> AnnData:
        """
        Preprocessing query dataset.

        Parameters
        ----------
        query :class:`anndata.AnnData`
            query dataset
        Returns
        -------
        Preprocessed query dataset
        """
        pass

    def basicProcessing(self, adata) -> None:
        """
        Basic Preprocessing

        Parameters
        ----------
        adata :class:`anndata.AnnData`
            reference data

        Returns
        -------

        """
        sc.pp.filter_cells(adata, min_genes=self.basic_params['genes_threshold'])
        sc.pp.normalize_total(adata, target_sum=self.basic_params['target_threshold'])
        sc.pp.log1p(adata)

    def test_prior(self) -> None:
        """
        Ensure the reference dataset is preprocessed.

        Returns
        -------

        """
        if not self.is_prior:
            raise (ValueError("Use 'refPreprocessing' to get index first."))


class BasicPreprocessor(BasePreprocessor):
    """
    Basic preprocessing class.

    Attributes
    ------------
    gene_index: :class:`pandas.Index`
        Selected genes' index
    is_prior: bool
        Weather reference is preprocessed
    basic_params: dict
        Preprocessing parameters

    Notes
    --------
    Steps:
        1. Filtering cells
        2. Normalization by counts per cell, every cell has the same total count after normalization
        3. Logarithm Transformation

    Examples
    ---------

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
    Preprocessing class for HVG (Highly Variable Gene) selection.
    Attributes
    ------------
    gene_index: :class:`pandas.Index`
        Selected genes' index
    is_prior: bool
        Weather reference is preprocessed
    basic_params: dict
        Preprocessing parameters
    n_hvg: int
        HVG count

    Notes
    --------
    Steps:
        1. Basic preprocessing
        2. HVG selection

    Examples
    ---------


    """
    def __init__(self, n_hvg: int = 1000):
        super().__init__()
        self.n_hvg: int = n_hvg

    def display_params(self):
        super().display_params()
        print(f'HVG Number: {self.n_hvg}')

    def reset_n_hvg(self, n_hvg: int) -> None:
        self.n_hvg = n_hvg
        self.is_prior = False

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
    Preprocessing class for PCA (Principle Components Analysis).

    Attributes
    ------------
    gene_index: :class:`pandas.Index`
        Selected genes' index
    is_prior: bool
        Weather reference is preprocessed
    basic_params: dict
        Preprocessing parameters
    n_hvg: int
        HVG count
    n_pc: int
        PC count

    Notes
    --------
    Steps:
        1. Basic preprocessing
        2. HVG selection
        3. PC selection

    Examples
    ---------


    """
    def __init__(self, n_hvg: int = 1000, n_pc: int = 50):
        super().__init__()
        self.n_hvg: int = n_hvg
        self.n_pc: int = n_pc
        self.trans_matrix: np.array = None

    def display_params(self):
        super().display_params()
        print(f'PC Number: {self.n_pc}')
        print(f'Transformation Matirx Shape: {self.trans_matrix.shape}')

    def reset_n_hvg(self, n_hvg: int) -> None:
        self.n_hvg = n_hvg
        self.is_prior = False

    def reset_n_pc(self, n_pc) -> None:
        self.n_pc = n_pc
        self.is_prior = False

    def refPreprocessing(self, ref: AnnData) -> AnnData:
        super().basicProcessing(adata=ref)
        sc.pp.highly_variable_genes(ref, n_top_genes=self.n_hvg, inplace=True)
        self.gene_index = ref.var[ref.var['highly_variable']].index
        ref = ref[:, self.gene_index]
        sc.pp.scale(ref, max_value=self.basic_params['max_threshold'])
        sc.tl.pca(ref, svd_solver='arpack')
        self.is_prior = True
        self.trans_matrix = ref.varm["PCs"][:, :self.n_pc]
        new_ref = ref[:, :self.n_pc]
        new_ref.var.index = ["PC" + str(i+1) for i in range(self.n_pc)]
        new_ref.X = ref.obsm["X_pca"][:, :self.n_pc]
        return new_ref

    def queryPreprocessor(self, query: AnnData) -> AnnData:
        super().test_prior()
        super().basicProcessing(adata=query)
        new_query = query[:, self.gene_index]
        sc.pp.scale(new_query, max_value=self.basic_params['max_threshold'])
        target_query = new_query[:, :self.n_pc]
        target_query.var.index = ["PC" + str(i+1) for i in range(self.n_pc)]
        target_query.X = new_query.X.dot(self.trans_matrix)
        return target_query


