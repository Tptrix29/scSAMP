from enum import Enum


class SamplingStrategy(Enum):
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


