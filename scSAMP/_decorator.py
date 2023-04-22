import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from .evaluation._score import specific_scores


def eval_metrics(eval_func):
    def wrapper(*args, **kwargs) -> dict:
        y, y_pred = eval_func(*args, **kwargs)
        scores = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y, y_pred, average='macro', zero_division=0),
            "F1": f1_score(y, y_pred, average='macro', zero_division=0),
            "kappa": cohen_kappa_score(y, y_pred)
        }
        for i in set(y):
            _, _, clst_f1 = specific_scores(y=y, y_pred=y_pred, target=i)
            scores[i+"_F1"] = clst_f1
        return scores
    return wrapper


def time_logging(mode: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            st = time.time()
            func(*args, **kwargs)
            end = time.time()
            t = end - st
            print(f"[{mode.upper()}] Time Consumption: {t} s")
            return t
        return wrapper
    return decorator




