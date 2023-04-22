from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.svm import SVC

from .._config import SamplingStrategy, EvaluationStrategy
from .._decorator import eval_metrics, time_logging
from ..evaluation.model._actinn import ACTINN
from ..processing._sampler import SamplingProcessor
from .._utils import _check_obs_key

font = {'family': 'serif',
        'serif': 'Helvetica',
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)


class EvaluationProcessor:
    """
    Evaluation-delivering class.

    Parameters
    -----------


    """
    def __init__(
            self,
            ref: AnnData,
            query: AnnData,
            col: str,
            up: Optional[float],
            step: int,
    ):
        self.raw: AnnData = ref
        self.query: AnnData = query
        self.ratio_range: np.array = np.linspace(0, up, step+1)[1:]
        self.pred_col: str = col

        self.train: Optional[AnnData] = None
        self.records: pd.DataFrame = pd.DataFrame()
        self.model: Optional[ACTINN, SVC] = None

        self.total_iter = self.ratio_range.shape[0] * len(SamplingStrategy)

        self.sampler: SamplingProcessor = SamplingProcessor(reference=ref, cluster_col=col, ratio=1)

    def eval(
            self,
            classifier: str,
            **kwargs
    ) -> None:
        """
        Evaluation with certain classifier.

        Parameters
        ----------
        classifier :class:`scSAMP.config.EvaluationStrategy`
            classification model name

        Returns
        -------

        """
        _check_obs_key(self.query, self.pred_col)

        if classifier not in [val for val in EvaluationStrategy]:
            raise(ValueError(f"Invalid Evaluation Strategy '{classifier}'."))

        X_test = self.query.X
        y_test = self.query.obs[self.pred_col]
        i = 1
        for r in self.ratio_range:
            for sampling in SamplingStrategy:
                print(f"{i} / {self.total_iter} ----------------------------------")
                i += 1
                sampling = str(sampling)
                self.sampler._reset_ratio(r)
                self.train = self.sampler.sampling(strategy=sampling)
                X_train = self.train.X
                y_train = self.train.obs[self.pred_col]

                # Evaluation
                if classifier == EvaluationStrategy.SVM:
                    training_time: float = self._svm_train(X_train, y_train, **kwargs)
                    record: dict = self._svm_eval(X_test, y_test)
                elif classifier == EvaluationStrategy.ACTINN:
                    training_time: float = self._actinn_train(X_train, y_train, **kwargs)
                    record: dict = self._actinn_eval(X_test, y_test)
                else:
                    raise ValueError(f"Invalid Classifier '{classifier}'")

                # Set up record properties
                record["train_time"] = training_time
                record["model"] = classifier
                record["method"] = sampling
                record["ratio"] = r
                record["n"] = int(r * self.raw.shape[0])
                for k, v in record.items():
                    record[k] = [v]

                self.records = pd.concat([self.records,
                                          pd.DataFrame(record,
                                                       index=[sampling+"-"+classifier+"-"+str(record["n"])])])

    @time_logging(mode="training")
    def _svm_train(self, X1, y1, **kwargs):
        self.model = SVC(**kwargs)
        self.model.fit(X1, y1)

    @eval_metrics
    def _svm_eval(self, X, y):
        return y, self.model.predict(X)

    @time_logging(mode="training")
    def _actinn_train(self, X1, y1, **kwargs):
        self.model = ACTINN(**kwargs)
        self.model.fit(X1, y1)

    @eval_metrics
    def _actinn_eval(self, X, y):
        return y, self.model.predict(X)

    @time_logging(mode="predicting")
    def predict(self, X):
        return self.model.predict(X)

    def get_records(self) -> pd.DataFrame:
        return self.records

    def lastest_status(self) -> Optional[pd.DataFrame]:
        if self.records.shape[0]:
            return self.records.iloc[-1, :]
        return None

    def deposit_records(self, filepath: str) -> None:
        self.records.to_csv(filepath)

    def fig_panel(
            self,
            model: str,
            metrics: Union[tuple, list] = ("accuracy", "precision", "recall", "F1", "kappa"),
            n_rows: int = 2,
            width: int = 3,
            height: int = 3,
            legend: bool = True,
            score_lim: bool = True
    ):
        methods = [str(i) for i in SamplingStrategy]
        est = len(methods) / n_rows
        n_cols = int(est) + 1 if est > int(est) else int(est)
        layout = str(n_rows) + str(n_cols)

        df = self.records[self.records["model"] == model]
        x = list(set(df["n"]))
        x_limit = int(max(x) / 1000 + 1) * 1000
        x.sort()

        plt.figure(int(layout), figsize=(width * n_cols, height * n_rows))
        i = 1
        for m in methods:
            plt.subplot(int(layout + str(i)))
            i += 1
            frag = df[df["method"] == m]
            for metric in metrics:
                plt.plot(x, frag[metric], "o-")
            plt.title(m)
            if score_lim:
                plt.ylim([0, 1])
            plt.xlim([0, x_limit])
        plt.tight_layout()
        if legend:
            plt.legend(metrics, loc="lower left", bbox_to_anchor=(1, 0))
        plt.show()

    def cluster_f1_cols(self) -> list:
        res = []
        for col in self.records.columns:
            if str(col).endswith("_F1"):
               res.append(col)
        return res
