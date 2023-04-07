from enum import Enum


class SamplingStrategy(Enum):
    """
    Sample strategy collection.

    :Options:
        - ``stratify``: Classical stratified sampling
        - ``balance``: Stratified sampling with balanced ratio
        - ``compactness``: Adjusted sampling with intra-cluster compactness factor
        - ``complexity``: Adjusted sampling with inter-cluster complexity factor
        - ``concave``: Adjusted sampling with concave integration of compactness factor and complexity factor
        - ``convex``: Adjusted sampling with convex integration of compactness factor and complexity factor
        - ``entropy``: Adjusted sampling with entropy weight integration of compactness factor and complexity factor

    Notes
    -----


    """
    STRATIFY: str = "stratify"
    BALANCE: str = "balance"
    COMPACTNESS: str = "compactness"
    COMPLEXITY: str = "complexity"
    CONCAVE: str = "concave"
    CONVEX: str = "convex"
    ENTROPY: str = "entropy"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __len__(self):
        return len([i for i in self])

    def __iter__(self):
        return self.value


class EvaluationStrategy(Enum):
    """
    Evaluation strategy collection.

    :Options:
        - ``SVM``: Support vector machine model
        - ``ACTINN``: Neural network model
    """

    SVM: str = "svm"
    ACTINN: str = 'actinn'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __iter__(self):
        return self.value

    def __len__(self):
        return len([i for i in self])





