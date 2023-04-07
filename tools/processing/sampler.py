import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData
from typing import Optional
from imblearn.over_sampling import SMOTE

from tools.utils import settings, _check_obs_key, _check_ratio
from tools.processing.balance import balance_power
import tools.processing.factors as fc
from tools.config import SamplingStrategy
from tools.decorator import time_logging


class SamplingProcessor:
    """
    Sampling Processing Class.

    Attributes
    ----------
    reference: :class:`anndata.AnnData`
        References dataset.
    sampling_ratio: float, default=None
        Temporary sampling ratio, range in [0, 1).
    cluster_label: str
        Column name of cell type label.
    raw_count: int
        Cell count of raw dataset.
    latest_label: str
        Sampling selection label of specific method and ratio (latest applied).
    oversampling_dataset: :class:`anndata.AnnData`
        Oversampling dataset, generated by :class:`imblearn.over_sampling.SMOTE`
    metadata: :class:`pandas.DataFrame`
        Metadata of cluster-specific factors.
    random_state: int
        Random state.

    Notes
    -----
    This is a beta version.

    Examples
    --------
    >>> import scanpy as sc
    >>> data = sc.read_h5ad("path/to/h5ad/file")
    >>> sampler = SamplingProcessor(reference=data, cluster_col="cell_type", random_state=0)
    >>> sampler.sampling("balance")
    """

    def __init__(
            self,
            reference: AnnData,
            cluster_col: str,
            ratio: Optional[float] = None,
            random_state: int = settings.random_state,
    ):
        if ratio:
            _check_ratio(ratio)
        self.sampling_ratio: Optional[float] = ratio
        self.reference: AnnData = reference
        self.cluster_label: str = cluster_col
        self.random_state: int = random_state

        self.raw_count: int = self.reference.obs.shape[0]

        self.latest_label: str = ''
        self.oversampling_dataset: Optional[AnnData] = None
        self.target_ratio: float = 0

        # change index
        self.reference.obs.index = pd.RangeIndex(self.raw_count).astype(str)
        # count and ratio analysis
        self.metadata: pd.DataFrame = self.reference.obs.groupby(self.cluster_label).apply(
            lambda x: pd.Series([x.shape[0], x.shape[0] / self.reference.shape[0]],
                                index=['count', 'ratio']))
        self.metadata['ratio_balanced'] = balance_power(self.metadata['ratio'])

    def sampling(
            self,
            strategy: str,
    ) -> AnnData:
        """
        Sampling function with strategy parameter.

        Parameters
        ----------
        strategy: str, {'stratify', 'balance', 'compactness', 'complexity', 'concave', 'convex', 'entropy'}
            Sampling strategy chosen from :class:`tools.config.SamplingStrategy`.

        Returns
        -------
            Sampled dataset as :class:`anndata.AnnData` object.
        """

        _check_obs_key(self.reference, self.cluster_label)
        if strategy not in [val for val in SamplingStrategy]:
            raise(ValueError(f"Invalid Sampling Strategy '{strategy}'."))

        if self.latest_label:
            self.oversampling_dataset.obs[self.latest_label] = False
        self.latest_label = self.cluster_label + '_' + strategy
        if strategy == SamplingStrategy.STRATIFY:
            return self._stratify()
        if strategy == SamplingStrategy.BALANCE:
            return self._imbalance_sampling(balance_power(self.metadata['ratio']))
        else:
            self._check_factor(strategy)
            return self._imbalance_sampling(self.metadata[strategy+'_balanced'])

    def _reset_ratio(self, ratio: float) -> None:
        _check_ratio(ratio)
        self.sampling_ratio = ratio

    def get_target_dataset(self, col: Optional[str] = None) -> AnnData:
        """
        Get sampled dataset.
        """
        col = self.latest_label if not col else col
        part1 = self.reference[self.reference.obs[col] == True]
        part1 = AnnData(X=part1.X, obs=part1.obs[[self.cluster_label]],
                        var=pd.DataFrame(index=self.reference.var.index), dtype=np.float64)
        part1.obs["isRaw"] = True
        if self.oversampling_dataset:
            if col in self.oversampling_dataset.obs.columns:
                part2 = self.oversampling_dataset[self.oversampling_dataset.obs[col] == True]
                part2 = AnnData(X=part2.X, obs=part2.obs[[self.cluster_label]],
                                var=pd.DataFrame(index=self.reference.var.index), dtype=np.float64)
                part2.obs["isRaw"] = False
                concatenate = ad.concat([part1, part2])
                concatenate.var.index = self.reference.var.index
                return concatenate
        return part1

    def generate_factors(self) -> None:
        """
        Generate cluster-specific factors.
        """
        for s in SamplingStrategy:
            if s != SamplingStrategy.BALANCE and s!= SamplingStrategy.STRATIFY:
                self._check_factor(str(s))

    def _check_factor(self, factor_type: str) -> None:
        """
        Optimize generation of specific factor.

        Parameters
        ----------
        factor_type
            Type of factor chosen from :class:`tools.config.SamplingStrategy`
        """
        factor_type = str(factor_type)
        if factor_type not in self.metadata.columns:
            if factor_type == SamplingStrategy.COMPACTNESS:
                _, self.metadata[str(SamplingStrategy.COMPACTNESS)] = \
                    fc.compactness_factor(self.reference.obs, self.reference.X, self.cluster_label)
            elif factor_type == SamplingStrategy.COMPLEXITY:
                _, self.metadata[str(SamplingStrategy.COMPLEXITY)] = \
                    fc.complexity_factor(self.reference.obs, self.reference.X, self.cluster_label)
            else:
                self._check_factor(SamplingStrategy.COMPACTNESS)
                self._check_factor(SamplingStrategy.COMPLEXITY)
                if factor_type == SamplingStrategy.CONCAVE:
                    self.metadata[str(SamplingStrategy.CONCAVE)] = fc.concave_2var(
                        self.metadata[str(SamplingStrategy.COMPACTNESS)],
                        self.metadata[str(SamplingStrategy.COMPLEXITY)])
                elif factor_type == SamplingStrategy.CONVEX:
                    self.metadata[str(SamplingStrategy.CONVEX)] = fc.convex_2var(
                        self.metadata[str(SamplingStrategy.COMPACTNESS)],
                        self.metadata[str(SamplingStrategy.COMPLEXITY)])
                elif factor_type == SamplingStrategy.ENTROPY:
                    self.metadata[str(SamplingStrategy.ENTROPY)] = fc.entropy_2var(
                        self.metadata[str(SamplingStrategy.COMPACTNESS)],
                        self.metadata[str(SamplingStrategy.COMPLEXITY)])
            self.metadata[factor_type + '_balanced'] = balance_power(self.metadata[factor_type])

    @time_logging(mode="oversampling")
    def _generate_oversampling(self) -> None:
        """
        Generate oversampling dataset by SMOTE.

        SMOTE is the most stable oversampling method (Generate fully-balanced dataset)
        """
        X_resampled, y_resampled = SMOTE(random_state=self.random_state) \
            .fit_resample(self.reference.X, self.reference.obs[self.cluster_label])
        X_resampled, y_resampled = X_resampled[self.raw_count:, :], y_resampled[self.raw_count:]
        print(y_resampled.shape[0])
        self.oversampling_dataset = AnnData(X=X_resampled,
                                            obs=pd.DataFrame({
                                                self.cluster_label: y_resampled,
                                                self.latest_label: True,
                                            }))

    def _stratify(self) -> AnnData:
        """ Classical stratified sampling method. """
        ind = pd.DataFrame(self.reference.obs[self.cluster_label])\
            .groupby(self.cluster_label).apply(
                lambda x: x.sample(frac=self.sampling_ratio,
                                   random_state=self.random_state)
            ).index.droplevel(0)

        self.reference.obs[self.latest_label] = self.reference.obs.index
        self.reference.obs[self.latest_label] = pd.Categorical(self.reference.obs.apply(
            lambda x: True if x[self.latest_label] in ind else False, axis=1
        ))
        return self.get_target_dataset()

    def _imbalance_sampling(self, target_ratio: np.array) -> AnnData:
        """
        Imbalanced sampling operation.

        Parameters
        ----------
        target_ratio
            target sampling ratio for each cluster

        Returns
        -------
            sampled dataset as :class:`anndata.AnnData` object
        """

        ref_target_index: pd.Index = pd.Index([])
        over_target_index: pd.Index = pd.Index([])

        target_count = self.raw_count * self.sampling_ratio
        for i in range(self.metadata.shape[0]):
            clst = self.reference.obs[self.reference.obs[self.cluster_label] == self.metadata.index[i]]
            sample_count = int(np.round(target_ratio[i] * target_count))
            if sample_count <= clst.shape[0]:
                ref_target_index = ref_target_index.union(clst.sample(n=sample_count, random_state=self.random_state).index)
            # Oversampling Operation
            else:
                ref_target_index = ref_target_index.union(clst.index)
                if self.oversampling_dataset is None:
                    self._generate_oversampling()
                partial_clst = self.oversampling_dataset.obs[
                    self.oversampling_dataset.obs[self.cluster_label] == self.metadata.index[i]]
                # print(f"Parameters: {sample_count-clst.shape[0]}")
                # print(f"partial shape: {partial_clst.shape}, label: {self.metadata.index[i]}")
                over_target_index = over_target_index.union(partial_clst.sample(
                        n=sample_count-clst.shape[0], random_state=self.random_state).index)

        self.reference.obs[self.latest_label] = self.reference.obs.index
        self.reference.obs[self.latest_label] = pd.Categorical(self.reference.obs.apply(
            lambda x: True if x[self.latest_label] in ref_target_index else False, axis=1
        ))
        if self.oversampling_dataset is not None:
            self.oversampling_dataset.obs[self.latest_label] = self.oversampling_dataset.obs.index
            self.oversampling_dataset.obs[self.latest_label] = pd.Categorical(self.oversampling_dataset.obs.apply(
                lambda x: True if x[self.latest_label] in over_target_index else False, axis=1
            ))

        return self.get_target_dataset()

    # def to_hdf5(self, strategy: str, format: str, prefix: Optional[str], suffix: Optional[str]) -> None:

    def suggest_ratio(self, strategy: str, quantile: float = 0.99) -> float:
        """
        Generate suggested sampling ratio.

        Parameters
        ----------
        strategy
            Type of sampling strategy chosen from :class:`tools.config.SamplingStrategy`
        quantile
            Customized quantile for hypothesis test.
        """
        import scipy.stats as sts
        mini_prob: float = min(self.metadata[str(strategy)+"_balanced"])
        size: float = sts.nbinom.ppf(q=quantile, p=mini_prob, n=1)
        return size / self.raw_count