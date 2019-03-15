
import math
# https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/metrics


def sort_and_couple(labels, scores):
    """Zip the `labels` with `scores` into a single list."""
    couple = list(zip(labels, scores))
    return sorted(couple, key=lambda x: x[1], reverse=True)


def map(y_true, y_pred, _threshold=0):
    """
    Calculate mean average precision.
    Example:
        >>> y_true = [0, 1, 0, 0]
        >>> y_pred = [0.1, 0.6, 0.2, 0.3]
        >>> MeanAveragePrecision()(y_true, y_pred)
        1.0
    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :return: Mean average precision.
    """
    result = 0.
    pos = 0
    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, score) in enumerate(coupled_pair):
        if label > _threshold:
            pos += 1.
            result += pos / (idx + 1.)
    if pos == 0:
        return 0.
    else:
        return result / pos


def mrr(y_true, y_pred, _threshold=0):
    """
    Calculate reciprocal of the rank of the first relevant item.
    Example:
        >>> import numpy as np
        >>> y_pred = np.asarray([0.2, 0.3, 0.7, 1.0])
        >>> y_true = np.asarray([1, 0, 0, 0])
        >>> MeanReciprocalRank()(y_true, y_pred)
        0.25
    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :return: Mean reciprocal rank.
    """
    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, pred) in enumerate(coupled_pair):
        if label > _threshold:
            return 1. / (idx + 1)
    return 0.


def dcg(y_true, y_pred, _k=1, _threshold=0) -> float:
    """
    Calculate discounted cumulative gain (dcg).
    Relevance is positive real values or binary values.
    Example:
        >>> y_true = [0, 1, 2, 0]
        >>> y_pred = [0.4, 0.2, 0.5, 0.7]
        >>> DiscountedCumulativeGain(1)(y_true, y_pred)
        0.0
        >>> round(DiscountedCumulativeGain(k=-1)(y_true, y_pred), 2)
        0.0
        >>> round(DiscountedCumulativeGain(k=2)(y_true, y_pred), 2)
        2.73
        >>> round(DiscountedCumulativeGain(k=3)(y_true, y_pred), 2)
        2.73
        >>> type(DiscountedCumulativeGain(k=1)(y_true, y_pred))
        <class 'float'>
    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :return: Discounted cumulative gain.
    """
    if _k <= 0:
        return 0.
    coupled_pair = sort_and_couple(y_true, y_pred)
    result = 0.
    for i, (label, score) in enumerate(coupled_pair):
        if i >= _k:
            break
        if label > _threshold:
            result += (math.pow(2., label) - 1.) / math.log(2. + i)
    return result


def ndcg(y_true, y_pred):
    """
    Calculate normalized discounted cumulative gain (ndcg).
    Relevance is positive real values or binary values.
    Example:
        >>> y_true = [0, 1, 2, 0]
        >>> y_pred = [0.4, 0.2, 0.5, 0.7]
        >>> ndcg = NormalizedDiscountedCumulativeGain
        >>> ndcg(k=1)(y_true, y_pred)
        0.0
        >>> round(ndcg(k=2)(y_true, y_pred), 2)
        0.52
        >>> round(ndcg(k=3)(y_true, y_pred), 2)
        0.52
        >>> type(ndcg()(y_true, y_pred))
        <class 'float'>
    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :return: Normalized discounted cumulative gain.
    """
    idcg_val = dcg(y_true, y_true)
    dcg_val = dcg(y_true, y_pred)
    return dcg_val / idcg_val if idcg_val != 0 else 0
