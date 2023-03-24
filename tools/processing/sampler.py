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

    Parameters
    -------------

    """
    def __init__(
            self,
            reference: AnnData,
            cluster_col: str,
            ratio: Optional[float],
            random_state: int = settings.random_state,
    ):
        """
        Initialize class with necessary parameters.
        :param reference: reference dataset prepared for sampling
        :param cluster_col: cell cluster label name
        :param ratio: target sampling ratio
        :param random_state: random state
        """
        if ratio:
            _check_ratio(ratio)
        self.sampling_ratio: float = ratio
        self.reference: AnnData = reference
        self.cluster_label: str = cluster_col
        self.random_state: int = random_state

        self.raw_count: int = self.reference.obs.shape[0]

        self.latest_label: str = ''
        self.oversampling_dataset: Optional[AnnData] = None

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
        :param strategy: sampling strategy
            Sampling Strategy Option:
            ``stratify``:
                Classical stratified sampling
            ``balance``:
                Stratified sampling with balanced ratio
            ``compactness``:
                Adjusted sampling with intra-cluster compactness factor
            ``complexity``:
                Adjusted sampling with inter-cluster complexity factor
            ``concave``:
                Adjusted sampling with concave integration of compactness factor and complexity factor
            ``convex``:
                Adjusted sampling with convex integration of compactness factor and complexity factor
            ``entropy``:
                Adjusted sampling with entropy weight integration of compactness factor and complexity factor
        :return: sampled dataset as AnnData object
        """
        _check_obs_key(self.reference, self.cluster_label)
        if strategy not in [val for val in SamplingStrategy]:
            raise(ValueError(f"Invalid Sampling Strategy '{strategy}'."))

        self.latest_label = self.cluster_label + '_' + strategy
        if strategy == SamplingStrategy.STRATIFY:
            return self._stratify()
        if strategy == SamplingStrategy.BALANCE:
            return self._imbalance_sampling(balance_power(self.metadata['ratio']))
        else:
            self._check_factor(strategy)
            return self._imbalance_sampling(self.metadata[strategy+'_balanced'])

    def reset_ratio(self, ratio: float) -> None:
        _check_ratio(ratio)
        self.sampling_ratio = ratio

    def get_target_dataset(self, col: Optional[str] = None) -> AnnData:
        """
        Get sampled dataset.
        :return: sampled dataset as AnnData object
        """
        col = self.latest_label if not col else col
        part1 = self.reference[self.reference.obs[col] == True]
        part1 = AnnData(X=part1.X, obs=part1.obs[[self.cluster_label]])
        if self.oversampling_dataset:
            if col in self.oversampling_dataset.obs.columns:
                part2 = self.oversampling_dataset[self.oversampling_dataset.obs[col] == True]
                part2 = AnnData(X=part2.X, obs=part2.obs[[self.cluster_label]])
                part1.obs["isRaw"] = True
                part2.obs["isRaw"] = False
                concatenate = ad.concat([part1, part2])
                concatenate.var.index = self.reference.var.index
                return concatenate
        return part1

    # def _calculate_boundary(self, strategy: str) -> (int, float):
    #     self._check_factor(strategy)
    #     choices: pd.Series = (self.metadata['count'] / self.metadata[strategy+'_balanced']).sort_values(ascending=True)
    #     i = 0
    #     while int(choices[i]) < int(len(choices) * self.metadata.loc[choices.index[i], 'count']):
    #         i += 1
    #         if i == len(choices):
    #             raise RuntimeError(
    #                 "This dataset is not suitable for optimizing sample amount, "
    #                 "please manually set up sampling ratio")
    #     choice = self.metadata.loc[choices.index[i], :]
    #     return int(choices[i]), int(np.abs(choice[strategy+"_balanced"] - choice["ratio_balanced"]) * 100)
    #
    # def optimize_amount(self, strategy: str, over_acceptance: Optional[int] = None) -> int:
    #     low, acceptance = self._calculate_boundary(strategy)
    #     if not over_acceptance:
    #         over_acceptance = acceptance
    #     if over_acceptance < 0 or over_acceptance >= 50:
    #         raise ValueError(f"Invalid oversampling acceptance '{over_acceptance}', valid range: [0, 50)")
    #     return int(low / (1 - over_acceptance / 100))

    def generate_factors(self):
        for s in SamplingStrategy:
            if s != SamplingStrategy.BALANCE and s!= SamplingStrategy.STRATIFY:
                self._check_factor(str(s))

    def _check_factor(self, factor_type: str):
        """
        Check the existence of specific factor
        :param factor_type: factor name
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
        Generate oversampling dataset.
        SMOTE is the most stable oversampling method (Generate fully-balanced dataset)
        Other choices: BorderlineSMOTE, ADASYN, SMOTESVM, SMOTEKmeans
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
        :param target_ratio: target sampling ratio for each cluster
        :return: sampled dataset as AnnData object
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

    def to_hdf5(self, strategy: str, prefix: str):
        pass


# def _stratify_sampling(
#         adata: AnnData,
#         ratio: float,
#         group_label: str,
#         random_state: int = settings.random_state
# ) -> (str, AnnData):
#     ind = pd.DataFrame(adata.obs[group_label]).groupby(group_label).apply(
#         lambda x: x.sample(frac=ratio, random_state=random_state)
#     ).index.droplevel(0)
#
#     col_label = group_label + '_stratify'
#     adata.obs[col_label] = adata.obs.index
#     adata.obs[col_label] = pd.Categorical(adata.obs.apply(
#         lambda x: True if x[col_label] in ind else False, axis=1
#     ))
#     return col_label, adata
#
#
# def _balance_sampling(
#         adata: AnnData,
#         ratio: float,
#         group_label: str,
# ) -> (str, AnnData):
#     # normal initial ratio
#     clst_summary = adata.obs.groupby(group_label).apply(
#         lambda x: x.shape[0]
#     ) / adata.obs.shape[0]
#     ratio_arr = balance_power(np.array(clst_summary))
#     return _imbalance_processing(adata, group_label, ratio,
#                                  group_label + '_balance',
#                                  ratio_arr, clst_summary.index)
#
#
# def _compactness_sampling(
#         adata: AnnData,
#         ratio: float,
#         group_label: str,
# ) -> (str, AnnData):
#     type_info, compactness_vec = fc.compactness_factor(adata.obs, adata.X, group_label)
#     ratio_arr = balance_power(np.array(compactness_vec))
#     return _imbalance_processing(adata, group_label, ratio,
#                                  group_label + '_compactness',
#                                  ratio_arr, type_info)
#
#
# def _complexity_sampling(
#         adata: AnnData,
#         ratio: float,
#         group_label: str,
# ) -> (str, AnnData):
#     type_info, complexity_vec = fc.complexity_factor(adata.obs, adata.X, group_label)
#     ratio_arr = balance_power(np.array(complexity_vec))
#     return _imbalance_processing(adata, group_label, ratio,
#                                  group_label + '_complexity',
#                                  ratio_arr, type_info)
#
#
# def _concave_sampling(
#         adata: AnnData,
#         ratio: float,
#         group_label: str,
# ) -> (str, AnnData):
#     type_info1, compactness_vec = fc.compactness_factor(adata.obs, adata.X, group_label)
#     type_info2, complexity_vec = fc.complexity_factor(adata.obs, adata.X, group_label)
#     ratio_arr = balance_power(fc.concave_2var(compactness_vec, complexity_vec))
#     return _imbalance_processing(adata, group_label, ratio,
#                                  group_label + '_concave',
#                                  ratio_arr, type_info1)
#
#
# def _convex_sampling(
#         adata: AnnData,
#         ratio: float,
#         group_label: str,
# ) -> (str, AnnData):
#     type_info1, compactness_vec = fc.compactness_factor(adata.obs, adata.X, group_label)
#     type_info2, complexity_vec = fc.complexity_factor(adata.obs, adata.X, group_label)
#     ratio_arr = balance_power(fc.convex_2var(compactness_vec, complexity_vec))
#     return _imbalance_processing(adata, group_label, ratio,
#                                  group_label + '_convex',
#                                  ratio_arr, type_info1)
#
#
# def _imbalance_processing(
#         adata: AnnData,
#         group_label: str,
#         ratio: float,
#         col_label: str,
#         ratio_arr: np.array,
#         types: pd.Index,
#         random_state: int = settings.random_state,
# ) -> (str, AnnData):
#     """
#     Processing sampling ratio with specific strategy.
#     When sampling ratio > cluster ratio: Oversampling
#     Oversampling Strategy:
#     1. BorderlineSMOTE *
#     2. ADASYN
#     :param adata: AnnData
#     :param group_label: cell type label
#     :param ratio: sampling ratio
#     :param col_label: sampling selection label
#     :param ratio_arr: mathematically balanced sampling ratio array for each cluster
#     :param clst_summary: sampling ratio array for each cluster
#     :param random_state:
#     :return:
#     """
#     total = adata.obs.shape[0] * ratio
#     data = pd.DataFrame()
#     resampled_part, addition = None, pd.DataFrame()
#     for i in range(len(types)):
#         clst = adata.obs[adata.obs[group_label] == types[i]]
#         sample_count = int(np.round(ratio_arr[i] * total))
#         if sample_count <= clst.shape[0]:
#             data = pd.concat([data,
#                               clst.sample(
#                                   n=sample_count, random_state=random_state
#                               )])
#         # Oversampling
#         else:
#             # print("Oversampling")
#             if resampled_part is None:
#                 X_resampled, y_resampled = BorderlineSMOTE(random_state=random_state) \
#                     .fit_resample(adata.X, adata.obs[group_label])
#                 resampled_part = AnnData(X=X_resampled,
#                                          obs=pd.DataFrame({
#                                              group_label: y_resampled,
#                                              col_label: [True for i in range(y_resampled.shape[0])]
#                                          }))[adata.shape[0]:, :]
#
#             partial_clst = resampled_part.obs[resampled_part.obs[group_label] == types[i]]
#             # print(f"{types[i]}: {partial_clst}")
#             data = pd.concat([data, clst])
#             addition = pd.concat([addition,
#                                   partial_clst.sample(n=sample_count - clst.shape[0], random_state=random_state)])
#
#     ind = data.index
#
#     adata.obs[col_label] = adata.obs.index
#     adata.obs[col_label] = pd.Categorical(adata.obs.apply(
#         lambda x: True if x[col_label] in ind else False, axis=1
#     ))
#     if resampled_part is None:
#         return col_label, adata
#     else:
#         new_adata = AnnData(X=adata.X, obs=adata.obs[[group_label, col_label]])
#         new_adata = ad.concat([new_adata, resampled_part[addition.index, :]])
#         new_adata.obs[group_label] = pd.Categorical(new_adata.obs[group_label])
#         new_adata.obs[col_label] = pd.Categorical(new_adata.obs[col_label])
#         return col_label, new_adata
